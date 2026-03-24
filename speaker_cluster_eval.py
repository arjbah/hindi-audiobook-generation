"""
speaker_cluster_eval.py
=======================
Evaluates how well Resemblyzer k-means speaker clustering aligns with
ground-truth character labels from a Pocket FM segment JSON.

Assumes one audio file per segment, named by segment_id, e.g.:
    ./audio/8000_story120_0000.wav
    ./audio/8000_story120_0001.wav
    ...

Usage
-----
python speaker_cluster_eval.py \
    --json     story120_segments.json \
    --audio_dir ./audio \
    --out_dir   ./eval_results \
    [--ext wav] \
    [--window 4.0] [--anchor_n 5]

If a segment's audio file is missing it is skipped with a warning.
"""

import argparse
import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import librosa
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize


# ──────────────────────────────────────────────────────────────────────
# Palette (matches your existing drift script)
# ──────────────────────────────────────────────────────────────────────
BLUE   = "#2563EB"
RED    = "#DC2626"
PURPLE = "#7C3AED"
GREEN  = "#059669"
AMBER  = "#D97706"
NARRATOR_LABEL = "narrator"


# ──────────────────────────────────────────────────────────────────────
# 1. Load & normalise ground-truth labels from JSON
# ──────────────────────────────────────────────────────────────────────

def load_gt_segments(json_path: str) -> list[dict]:
    """
    Returns a list of dicts, each with:
        segment_id : str
        text       : str
        character  : str   (empty → NARRATOR_LABEL)
        gender     : str
        age        : str
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    segments = []
    for entry in raw:
        char = entry.get("character", "").strip()
        segments.append({
            "segment_id": entry["segment_id"],
            "text":       entry.get("text", ""),
            "character":  char if char else NARRATOR_LABEL,
            "gender":     entry.get("gender", ""),
            "age":        entry.get("age", ""),
        })
    return segments


def unique_characters(segments: list[dict]) -> list[str]:
    """Sorted unique character names, narrator always first."""
    chars = sorted({s["character"] for s in segments})
    if NARRATOR_LABEL in chars:
        chars = [NARRATOR_LABEL] + [c for c in chars if c != NARRATOR_LABEL]
    return chars


# ──────────────────────────────────────────────────────────────────────
# 2. Embed each segment
# ──────────────────────────────────────────────────────────────────────

def load_encoder(device: str) -> VoiceEncoder:
    print(f"  Loading Resemblyzer GE2E encoder on {device}…")
    return VoiceEncoder(device=device)


def embed_segment_file(audio_path: str, encoder: VoiceEncoder,
                       min_sec: float = 0.5) -> np.ndarray | None:
    """
    Load one segment audio file, return L2-normalised 256-d embedding.
    Returns None if the file is too short or unreadable.
    """
    try:
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        if len(wav) / 16000 < min_sec:
            return None
        processed = preprocess_wav(wav, source_sr=16000)
        emb = encoder.embed_utterance(processed)
        return emb / (np.linalg.norm(emb) + 1e-8)
    except Exception as e:
        warnings.warn(f"    Could not embed {audio_path}: {e}")
        return None


def build_embedding_matrix(segments: list[dict], audio_dir: str,
                            encoder: VoiceEncoder, ext: str = "wav"
                            ) -> tuple[np.ndarray, list[dict]]:
    """
    Embeds every segment that has a corresponding audio file.
    Returns:
        embeddings : (N, 256) float32 array
        valid_segs : list of segment dicts that were successfully embedded
    """
    embeddings  = []
    valid_segs  = []
    missing     = 0

    for seg in segments:
        path = os.path.join(audio_dir, f"{seg['segment_id']}.{ext}")
        if not os.path.exists(path):
            missing += 1
            continue
        emb = embed_segment_file(path, encoder)
        if emb is not None:
            embeddings.append(emb)
            valid_segs.append(seg)

    if missing:
        print(f"  [warn] {missing} segment audio files not found — skipped")
    print(f"  Embedded {len(valid_segs)} / {len(segments)} segments")
    return np.stack(embeddings).astype(np.float32), valid_segs


# ──────────────────────────────────────────────────────────────────────
# 3. K-means clustering
# ──────────────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray,
                       n_clusters: int) -> np.ndarray:
    """L2-normalise then k-means. Returns cluster label array."""
    normed = normalize(embeddings)
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    return km.fit_predict(normed)


# ──────────────────────────────────────────────────────────────────────
# 4. Hungarian matching  (predicted cluster → GT character)
# ──────────────────────────────────────────────────────────────────────

def hungarian_match(cluster_labels: np.ndarray,
                    gt_labels: np.ndarray,
                    n_clusters: int,
                    n_gt: int) -> dict[int, int]:
    """
    Finds the optimal 1-to-1 assignment of predicted cluster IDs
    to GT character IDs that maximises total overlap.

    Returns a dict: {cluster_id → gt_id}
    """
    # Cost matrix: -overlap (we minimise, so negate for max)
    cost = np.zeros((n_clusters, n_gt), dtype=np.float64)
    for c in range(n_clusters):
        for g in range(n_gt):
            cost[c, g] = -np.sum((cluster_labels == c) & (gt_labels == g))

    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


# ──────────────────────────────────────────────────────────────────────
# 5. Evaluation metrics
# ──────────────────────────────────────────────────────────────────────

def compute_eval_metrics(cluster_labels: np.ndarray,
                         gt_int: np.ndarray,
                         mapping: dict[int, int],
                         char_names: list[str]) -> dict:
    """
    Returns a dict with:
        overall_accuracy    : fraction correctly assigned
        cluster_purity      : mean purity across all clusters
        per_char_precision  : {char_name: precision}
        per_char_recall     : {char_name: recall}
        per_char_f1         : {char_name: f1}
        confusion           : (n_gt × n_gt) confusion matrix
                              rows=GT, cols=predicted-mapped
    """
    n = len(cluster_labels)
    n_gt = len(char_names)

    # Map cluster IDs → GT IDs using Hungarian assignment
    mapped = np.array([mapping.get(int(c), -1) for c in cluster_labels])

    overall_accuracy = float(np.mean(mapped == gt_int))

    # Purity per cluster: fraction of dominant GT class
    purities = []
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        gt_in_cluster = gt_int[mask]
        dominant_frac = np.max(np.bincount(gt_in_cluster, minlength=n_gt)) / mask.sum()
        purities.append(float(dominant_frac))
    cluster_purity = float(np.mean(purities))

    # Per-character precision / recall / F1
    per_char_precision, per_char_recall, per_char_f1 = {}, {}, {}
    for g, char in enumerate(char_names):
        tp = int(np.sum((mapped == g) & (gt_int == g)))
        fp = int(np.sum((mapped == g) & (gt_int != g)))
        fn = int(np.sum((mapped != g) & (gt_int == g)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_char_precision[char] = round(prec, 4)
        per_char_recall[char]    = round(rec, 4)
        per_char_f1[char]        = round(f1, 4)

    # Confusion matrix: rows = GT, cols = mapped prediction
    cm = confusion_matrix(gt_int, mapped, labels=list(range(n_gt)))

    return {
        "overall_accuracy":   overall_accuracy,
        "cluster_purity":     cluster_purity,
        "per_char_precision": per_char_precision,
        "per_char_recall":    per_char_recall,
        "per_char_f1":        per_char_f1,
        "confusion":          cm,
        "mapped_labels":      mapped,
    }


def compute_embedding_separation(embeddings: np.ndarray,
                                  gt_int: np.ndarray,
                                  char_names: list[str]) -> dict:
    """
    Intra-character consistency and inter-character distance
    computed on GT labels (ground-truth grouping, not clusters).
    """
    char_embs  = {}
    char_means = {}
    intra      = {}

    for g, char in enumerate(char_names):
        mask = gt_int == g
        embs = embeddings[mask]
        char_embs[char] = embs
        mean = embs.mean(axis=0)
        char_means[char] = mean / (np.linalg.norm(mean) + 1e-8)

        if len(embs) < 2:
            intra[char] = float("nan")
        else:
            sims = [1 - cosine(embs[i], embs[j])
                    for i, j in combinations(range(len(embs)), 2)]
            intra[char] = float(np.mean(sims))

    inter = {}
    for ca, cb in combinations(char_names, 2):
        d = float(cosine(char_means[ca], char_means[cb]))
        inter[(ca, cb)] = inter[(cb, ca)] = d

    return {"intra": intra, "inter": inter,
            "char_means": char_means, "char_embs": char_embs}


# ──────────────────────────────────────────────────────────────────────
# 6. Visualisation
# ──────────────────────────────────────────────────────────────────────

def _reduce_dims(embeddings: np.ndarray):
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42,
                       n_neighbors=min(15, len(embeddings) - 1))
        return reducer.fit_transform(embeddings), "UMAP"
    except ImportError:
        pass
    try:
        from sklearn.manifold import TSNE
        return (TSNE(n_components=2, random_state=42,
                     perplexity=min(30, len(embeddings) - 1))
                .fit_transform(embeddings), "t-SNE")
    except Exception:
        pass
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(embeddings), "PCA"


def plot_eval_panel(embeddings: np.ndarray,
                    gt_int: np.ndarray,
                    cluster_labels: np.ndarray,
                    mapped_labels: np.ndarray,
                    eval_metrics: dict,
                    sep_metrics: dict,
                    char_names: list[str],
                    mapping: dict,
                    stem: str,
                    out_dir: str):

    n_chars  = len(char_names)
    palette  = plt.cm.tab10(np.linspace(0, 1, n_chars))
    embs_2d, proj = _reduce_dims(normalize(embeddings))

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Speaker Cluster Eval vs Ground Truth  —  {stem}",
                 fontsize=13, fontweight="bold", y=1.0)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── Panel A: GT labels in projection space
    ax_gt = fig.add_subplot(gs[0, 0])
    for g, char in enumerate(char_names):
        mask = gt_int == g
        ax_gt.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                      color=palette[g], s=28, alpha=0.7, label=char,
                      edgecolors="none")
    ax_gt.set_title(f"Ground Truth Labels ({proj})")
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_gt.legend(fontsize=7, title="GT char", title_fontsize=7,
                 loc="best", framealpha=0.6)

    # ── Panel B: Predicted cluster → GT-mapped colours
    ax_pred = fig.add_subplot(gs[0, 1])
    for g, char in enumerate(char_names):
        mask = mapped_labels == g
        if mask.sum() == 0:
            continue
        ax_pred.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                        color=palette[g], s=28, alpha=0.7,
                        label=char, edgecolors="none")
    # Mark misclassified segments with a red X
    wrong = mapped_labels != gt_int
    ax_pred.scatter(embs_2d[wrong, 0], embs_2d[wrong, 1],
                    marker="x", color="red", s=40, linewidths=0.8,
                    alpha=0.6, label="misclassified", zorder=5)
    ax_pred.set_title(f"Predicted (Hungarian-mapped) — acc={eval_metrics['overall_accuracy']:.3f}")
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    ax_pred.legend(fontsize=7, title="Pred char", title_fontsize=7,
                   loc="best", framealpha=0.6)

    # ── Panel C: Confusion matrix
    ax_cm = fig.add_subplot(gs[0, 2])
    cm = eval_metrics["confusion"]
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_xticks(range(n_chars))
    ax_cm.set_yticks(range(n_chars))
    short = [c[:10] for c in char_names]
    ax_cm.set_xticklabels(short, rotation=35, ha="right", fontsize=7)
    ax_cm.set_yticklabels(short, fontsize=7)
    ax_cm.set_xlabel("Predicted character")
    ax_cm.set_ylabel("Ground truth character")
    ax_cm.set_title("Confusion Matrix\n(rows=GT, cols=Pred)")
    for i in range(n_chars):
        for j in range(n_chars):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=7,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    # ── Panel D: Per-character F1 bar chart
    ax_f1 = fig.add_subplot(gs[1, 0])
    f1_vals = [eval_metrics["per_char_f1"][c] for c in char_names]
    bars = ax_f1.barh(char_names, f1_vals,
                      color=[palette[g] for g in range(n_chars)],
                      edgecolor="white", height=0.55)
    ax_f1.set_xlim(0, 1.1)
    ax_f1.axvline(0.75, color="gray", linestyle="--", linewidth=1)
    ax_f1.set_xlabel("F1 score")
    ax_f1.set_title("Per-Character F1\n(Precision × Recall harmonic mean)")
    for bar, char in zip(bars, char_names):
        ax_f1.text(bar.get_width() + 0.01,
                   bar.get_y() + bar.get_height() / 2,
                   f"{eval_metrics['per_char_f1'][char]:.3f}",
                   va="center", fontsize=8)

    # ── Panel E: Intra-char consistency (GT grouping)
    ax_intra = fig.add_subplot(gs[1, 1])
    intra_vals = [sep_metrics["intra"].get(c, float("nan")) for c in char_names]
    valid_chars = [c for c, v in zip(char_names, intra_vals) if not np.isnan(v)]
    valid_vals  = [v for v in intra_vals if not np.isnan(v)]
    valid_colors = [palette[char_names.index(c)] for c in valid_chars]
    bars2 = ax_intra.barh(valid_chars, valid_vals,
                           color=valid_colors, edgecolor="white", height=0.55)
    ax_intra.set_xlim(0, 1.1)
    ax_intra.axvline(0.75, color="gray", linestyle="--", linewidth=1,
                     label="0.75 ref")
    ax_intra.set_xlabel("Mean pairwise cosine similarity")
    ax_intra.set_title("GT Intra-Char Consistency\n(higher = voice is stable within GT class)")
    ax_intra.legend(fontsize=7)
    for bar, char in zip(bars2, valid_chars):
        ax_intra.text(bar.get_width() + 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"{sep_metrics['intra'][char]:.3f}",
                      va="center", fontsize=8)

    # ── Panel F: Summary table
    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    rows = [
        ["Overall accuracy",    f"{eval_metrics['overall_accuracy']:.4f}"],
        ["Cluster purity",      f"{eval_metrics['cluster_purity']:.4f}"],
        ["# GT characters",     str(n_chars)],
        ["# segments embedded", str(len(gt_int))],
        ["", ""],
        ["Character", "Prec / Rec / F1"],
    ]
    for char in char_names:
        p  = eval_metrics["per_char_precision"][char]
        r  = eval_metrics["per_char_recall"][char]
        f1 = eval_metrics["per_char_f1"][char]
        rows.append([char[:16], f"{p:.2f} / {r:.2f} / {f1:.2f}"])

    tbl = ax_tbl.table(cellText=rows, cellLoc="left", loc="center",
                        colWidths=[0.62, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax_tbl.set_title("Evaluation Summary", pad=8)

    out_path = os.path.join(out_dir, f"{stem}_cluster_eval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")


# ──────────────────────────────────────────────────────────────────────
# 7. Console report
# ──────────────────────────────────────────────────────────────────────

def print_report(eval_metrics: dict, sep_metrics: dict,
                 char_names: list[str], mapping: dict):
    print("\n" + "═" * 60)
    print("  SPEAKER CLUSTER EVALUATION REPORT")
    print("═" * 60)
    print(f"  Overall accuracy  : {eval_metrics['overall_accuracy']:.4f}")
    print(f"  Cluster purity    : {eval_metrics['cluster_purity']:.4f}")
    print()
    print(f"  {'Character':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'GT-Intra':>10} {'#segs':>6}")
    print("  " + "-" * 58)
    for char in char_names:
        p     = eval_metrics["per_char_precision"][char]
        r     = eval_metrics["per_char_recall"][char]
        f1    = eval_metrics["per_char_f1"][char]
        intra = sep_metrics["intra"].get(char, float("nan"))
        intra_str = f"{intra:.3f}" if not np.isnan(intra) else "  n/a"
        print(f"  {char:<20} {p:>6.3f} {r:>6.3f} {f1:>6.3f} "
              f"{intra_str:>10}")

    print()
    print("  Inter-character cosine distance (GT centroids):")
    for (ca, cb), d in sep_metrics["inter"].items():
        if ca < cb:  # avoid duplicate pairs
            print(f"    {ca:<18} <-> {cb:<18}  {d:.4f}")

    print()
    print("  Hungarian cluster → GT character mapping:")
    for cluster_id, gt_id in mapping.items():
        print(f"    cluster_{cluster_id}  →  {char_names[gt_id]}")
    print("═" * 60)


# ──────────────────────────────────────────────────────────────────────
# 8. Save JSON results
# ──────────────────────────────────────────────────────────────────────

def save_results_json(eval_metrics: dict, sep_metrics: dict,
                      char_names: list[str], mapping: dict,
                      stem: str, out_dir: str):
    output = {
        "stem":             stem,
        "overall_accuracy": eval_metrics["overall_accuracy"],
        "cluster_purity":   eval_metrics["cluster_purity"],
        "characters":       char_names,
        "hungarian_mapping": {str(k): char_names[v] for k, v in mapping.items()},
        "per_character": {
            char: {
                "precision":        eval_metrics["per_char_precision"][char],
                "recall":           eval_metrics["per_char_recall"][char],
                "f1":               eval_metrics["per_char_f1"][char],
                "gt_intra_consistency": (
                    None if np.isnan(sep_metrics["intra"].get(char, float("nan")))
                    else sep_metrics["intra"][char]
                ),
            }
            for char in char_names
        },
        "inter_character_distances": {
            f"{ca}__vs__{cb}": d
            for (ca, cb), d in sep_metrics["inter"].items()
            if ca < cb
        },
        "confusion_matrix": eval_metrics["confusion"].tolist(),
    }
    path = os.path.join(out_dir, f"{stem}_cluster_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Resemblyzer speaker clustering against JSON ground truth"
    )
    parser.add_argument("--json",       required=True,
                        help="Path to segment JSON (e.g. story120.json)")
    parser.add_argument("--audio_dir",  required=True,
                        help="Directory containing per-segment audio files")
    parser.add_argument("--ext",        default="wav",
                        help="Audio file extension (default: wav)")
    parser.add_argument("--out_dir",    default="./eval_results")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Override k-means k. Defaults to #unique GT characters.")
    parser.add_argument("--anchor_n",   type=int, default=5,
                        help="Anchor segments for drift (not used here but kept for API parity)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.json).stem

    print(f"\n{'═'*60}")
    print(f"  Story: {stem}")
    print(f"{'═'*60}")

    # ── Load GT
    segments   = load_gt_segments(args.json)
    char_names = unique_characters(segments)
    n_gt       = len(char_names)
    char_to_id = {c: i for i, c in enumerate(char_names)}

    print(f"\n  Ground-truth characters ({n_gt}):")
    for char in char_names:
        count = sum(1 for s in segments if s["character"] == char)
        print(f"    {char:<24} {count:>4} segments")

    # ── Embed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_encoder(device)

    print("\n  Extracting per-segment embeddings…")
    embeddings, valid_segs = build_embedding_matrix(
        segments, args.audio_dir, encoder, ext=args.ext
    )

    gt_int = np.array([char_to_id[s["character"]] for s in valid_segs])

    # ── Cluster
    n_clusters = args.n_clusters or n_gt
    print(f"\n  Running k-means with k={n_clusters}…")
    cluster_labels = cluster_embeddings(embeddings, n_clusters)

    # ── Hungarian match
    mapping = hungarian_match(cluster_labels, gt_int, n_clusters, n_gt)

    # ── Eval metrics
    eval_metrics = compute_eval_metrics(
        cluster_labels, gt_int, mapping, char_names
    )
    sep_metrics = compute_embedding_separation(embeddings, gt_int, char_names)

    # ── Report
    print_report(eval_metrics, sep_metrics, char_names, mapping)

    # ── Plots + JSON
    plot_eval_panel(
        embeddings, gt_int, cluster_labels,
        eval_metrics["mapped_labels"],
        eval_metrics, sep_metrics,
        char_names, mapping, stem, args.out_dir
    )
    save_results_json(eval_metrics, sep_metrics, char_names, mapping, stem, args.out_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
