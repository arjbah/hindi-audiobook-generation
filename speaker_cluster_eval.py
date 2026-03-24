"""
speaker_cluster_eval_compiled.py
=================================
Evaluates Resemblyzer speaker clustering against ground-truth character
labels from a Pocket FM segment JSON, when only a single compiled audio
file is available (no per-segment WAVs).

Strategy
--------
1. Load the compiled audio.
2. Split into speech segments using silence detection (librosa) or
   Silero-VAD if available (more accurate).
3. Match detected segments to JSON entries 1-to-1 by order
   (valid because the audio was synthesised sequentially from the JSON).
4. Embed each speech segment with Resemblyzer.
5. Run k-means clustering and evaluate against GT labels via
   Hungarian assignment.

Usage
-----
python speaker_cluster_eval.py `
    --json     C:/Users/prana/source/repos/hindi-audiobook-generation/data/story120_multi_captions.json `
    --audio    C:/Users/prana/source/repos/hindi-audiobook-generation/Audios/story120_multi.wav `
    --out_dir  ./eval_results

Optional flags
    --min_silence_ms   300     silence gap to split on (ms)
    --silence_thresh   -40     dBFS threshold for silence
    --n_clusters       5       override k (defaults to #GT characters)
    --save_segments            dump split WAVs to out_dir/segments/ for inspection
"""

import argparse
import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import librosa
import librosa.effects
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import soundfile as sf
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

NARRATOR_LABEL = "narrator"
BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#059669"
AMBER  = "#D97706"
PURPLE = "#7C3AED"


# ──────────────────────────────────────────────────────────────────────
# 1.  Ground-truth loading
# ──────────────────────────────────────────────────────────────────────

def load_gt_segments(json_path: str) -> list[dict]:
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
    chars = sorted({s["character"] for s in segments})
    if NARRATOR_LABEL in chars:
        chars = [NARRATOR_LABEL] + [c for c in chars if c != NARRATOR_LABEL]
    return chars


# ──────────────────────────────────────────────────────────────────────
# 2.  Audio splitting
# ──────────────────────────────────────────────────────────────────────

def split_on_silence_librosa(wav: np.ndarray, sr: int,
                              top_db: float = 40,
                              min_silence_ms: int = 300,
                              min_seg_ms: int = 300) -> list[tuple[float, float, np.ndarray]]:
    """
    Uses librosa.effects.split (energy-based) to find non-silent intervals.
    Returns list of (start_sec, end_sec, audio_array).
    """
    frame_length = 512
    hop_length   = 128
    intervals    = librosa.effects.split(
        wav,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    min_silence_samples = int(min_silence_ms / 1000 * sr)
    min_seg_samples     = int(min_seg_ms / 1000 * sr)

    # Merge intervals that are separated by less than min_silence_ms
    merged = []
    for start, end in intervals:
        if merged and (start - merged[-1][1]) < min_silence_samples:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append([start, end])

    segments = []
    for start, end in merged:
        if (end - start) < min_seg_samples:
            continue
        segments.append((start / sr, end / sr, wav[start:end]))

    return segments


def split_with_silero(wav: np.ndarray, sr: int,
                      min_seg_ms: int = 300) -> list[tuple[float, float, np.ndarray]]:
    """
    Uses Silero-VAD for more accurate speech/silence boundaries.
    Falls back to librosa if silero is unavailable.
    """
    try:
        # Silero requires 16kHz mono
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        (get_speech_timestamps, _, _, _, _) = utils

        wav_tensor = torch.from_numpy(wav).float()
        speech_ts  = get_speech_timestamps(
            wav_tensor, model,
            sampling_rate=sr,
            min_silence_duration_ms=300,
            min_speech_duration_ms=min_seg_ms,
        )

        min_samples = int(min_seg_ms / 1000 * sr)
        segments = []
        for ts in speech_ts:
            s, e = ts["start"], ts["end"]
            if (e - s) < min_samples:
                continue
            segments.append((s / sr, e / sr, wav[s:e]))

        print(f"  [Silero-VAD] detected {len(segments)} speech segments")
        return segments

    except Exception as ex:
        print(f"  [Silero-VAD] not available ({ex}), falling back to librosa silence split")
        return None


def split_audio(wav: np.ndarray, sr: int,
                top_db: float = 40,
                min_silence_ms: int = 300) -> list[tuple[float, float, np.ndarray]]:
    """Try Silero first, fall back to librosa."""
    result = split_with_silero(wav, sr, min_seg_ms=300)
    if result is not None:
        return result
    segs = split_on_silence_librosa(wav, sr, top_db=top_db,
                                     min_silence_ms=min_silence_ms)
    print(f"  [librosa split] detected {len(segs)} speech segments")
    return segs


# ──────────────────────────────────────────────────────────────────────
# 3.  Match audio segments → JSON entries 1-to-1
# ──────────────────────────────────────────────────────────────────────

def align_segments_to_json(audio_segs: list[tuple],
                            gt_segs: list[dict],
                            verbose: bool = True
                            ) -> tuple[list[np.ndarray], list[dict]]:
    """
    Simple ordered alignment: audio_segs[i] corresponds to gt_segs[i].

    If counts differ we warn and truncate to the shorter list.
    This is valid because the compiled audio was synthesised in JSON order.
    """
    n_audio = len(audio_segs)
    n_json  = len(gt_segs)

    if n_audio != n_json:
        warnings.warn(
            f"\n  [alignment] detected {n_audio} audio segments but JSON has "
            f"{n_json} entries.\n"
            f"  Truncating to {min(n_audio, n_json)}.\n"
            f"  If the mismatch is large, try adjusting --min_silence_ms or "
            f"--silence_thresh to get a better split."
        )

    n = min(n_audio, n_json)
    wavs     = [audio_segs[i][2] for i in range(n)]
    matched  = [gt_segs[i]       for i in range(n)]

    if verbose:
        print(f"\n  Alignment: {n_audio} audio segs  ↔  {n_json} JSON entries "
              f"→ using {n}")

    return wavs, matched


# ──────────────────────────────────────────────────────────────────────
# 4.  Embedding
# ──────────────────────────────────────────────────────────────────────

def load_encoder(device: str) -> VoiceEncoder:
    print(f"  Loading Resemblyzer GE2E encoder on {device}…")
    return VoiceEncoder(device=device)


def embed_wavs(wavs: list[np.ndarray],
               encoder: VoiceEncoder) -> tuple[np.ndarray, list[int]]:
    """
    Embed a list of raw waveforms.  Returns:
        embeddings : (M, 256) — only segments that embedded successfully
        valid_idx  : indices into input list that succeeded
    """
    embeddings = []
    valid_idx  = []

    for i, wav in enumerate(wavs):
        try:
            processed = preprocess_wav(wav, source_sr=16000)
            emb = encoder.embed_utterance(processed)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
            valid_idx.append(i)
        except Exception as e:
            warnings.warn(f"  [embed] segment {i} failed: {e}")

    print(f"  Embedded {len(embeddings)} / {len(wavs)} segments")
    return np.stack(embeddings).astype(np.float32), valid_idx


# ──────────────────────────────────────────────────────────────────────
# 5.  Clustering + Hungarian (identical to previous script)
# ──────────────────────────────────────────────────────────────────────

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    normed = normalize(embeddings)
    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    return km.fit_predict(normed)


def hungarian_match(cluster_labels, gt_int, n_clusters, n_gt):
    cost = np.zeros((n_clusters, n_gt))
    for c in range(n_clusters):
        for g in range(n_gt):
            cost[c, g] = -np.sum((cluster_labels == c) & (gt_int == g))
    row, col = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row, col)}


def compute_eval_metrics(cluster_labels, gt_int, mapping, char_names):
    n_gt   = len(char_names)
    mapped = np.array([mapping.get(int(c), -1) for c in cluster_labels])
    acc    = float(np.mean(mapped == gt_int))

    purities = []
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        dom  = np.max(np.bincount(gt_int[mask], minlength=n_gt)) / mask.sum()
        purities.append(float(dom))

    prec_d, rec_d, f1_d = {}, {}, {}
    for g, char in enumerate(char_names):
        tp = int(np.sum((mapped == g) & (gt_int == g)))
        fp = int(np.sum((mapped == g) & (gt_int != g)))
        fn = int(np.sum((mapped != g) & (gt_int == g)))
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec_d[char] = round(p, 4)
        rec_d[char]  = round(r, 4)
        f1_d[char]   = round(f1, 4)

    cm = confusion_matrix(gt_int, mapped, labels=list(range(n_gt)))
    return {
        "overall_accuracy":   acc,
        "cluster_purity":     float(np.mean(purities)),
        "per_char_precision": prec_d,
        "per_char_recall":    rec_d,
        "per_char_f1":        f1_d,
        "confusion":          cm,
        "mapped_labels":      mapped,
    }


def compute_separation(embeddings, gt_int, char_names):
    char_means, intra, inter = {}, {}, {}
    for g, char in enumerate(char_names):
        embs = embeddings[gt_int == g]
        mean = embs.mean(axis=0)
        char_means[char] = mean / (np.linalg.norm(mean) + 1e-8)
        if len(embs) < 2:
            intra[char] = float("nan")
        else:
            sims = [1 - cosine(embs[i], embs[j])
                    for i, j in combinations(range(len(embs)), 2)]
            intra[char] = float(np.mean(sims))
    for ca, cb in combinations(char_names, 2):
        d = float(cosine(char_means[ca], char_means[cb]))
        inter[(ca, cb)] = inter[(cb, ca)] = d
    return {"intra": intra, "inter": inter, "char_means": char_means}


# ──────────────────────────────────────────────────────────────────────
# 6.  Segment-level timeline plot  (new — shows GT vs predicted over time)
# ──────────────────────────────────────────────────────────────────────

def plot_timeline(audio_segs_used: list[tuple],
                  gt_int: np.ndarray,
                  mapped_labels: np.ndarray,
                  char_names: list[str],
                  palette,
                  out_dir: str, stem: str):
    """
    Two horizontal colour strips:
        top    = ground truth character per segment
        bottom = predicted (Hungarian-mapped) character per segment
    """
    starts = np.array([s[0] for s in audio_segs_used])
    ends   = np.array([s[1] for s in audio_segs_used])

    fig, axes = plt.subplots(3, 1, figsize=(18, 4),
                              gridspec_kw={"height_ratios": [1, 1, 0.4]})
    fig.suptitle(f"Segment-level GT vs Predicted  —  {stem}",
                 fontsize=11, fontweight="bold")

    for ax, labels, title in zip(
        axes[:2],
        [gt_int, mapped_labels],
        ["Ground Truth", "Predicted (Hungarian-mapped)"]
    ):
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(s, e, color=palette[labels[i]], alpha=0.85)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=8)
        ax.set_xlim(starts[0], ends[-1])

    # Legend strip
    axes[2].axis("off")
    for g, char in enumerate(char_names):
        axes[2].barh(0, 1, left=g, color=palette[g], height=0.5)
        axes[2].text(g + 0.5, 0, char[:12], ha="center", va="center",
                     fontsize=7, color="white" if g < 8 else "black")
    axes[2].set_xlim(0, len(char_names))
    axes[2].set_title("Character legend", fontsize=8, pad=2)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────
# 7.  Main eval panel (same structure as previous script)
# ──────────────────────────────────────────────────────────────────────

def _reduce_dims(embeddings):
    try:
        from umap import UMAP
        r = UMAP(n_components=2, random_state=42,
                 n_neighbors=min(15, len(embeddings) - 1))
        return r.fit_transform(embeddings), "UMAP"
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


def plot_eval_panel(embeddings, gt_int, mapped_labels,
                    eval_metrics, sep_metrics,
                    char_names, stem, out_dir):

    n      = len(char_names)
    pal    = plt.cm.tab10(np.linspace(0, 1, n))
    e2d, proj = _reduce_dims(normalize(embeddings))

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Cluster Eval vs Ground Truth  —  {stem}",
                 fontsize=13, fontweight="bold", y=1.0)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    for ax, labels, title_str in zip(
        [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
        [gt_int, mapped_labels],
        [f"Ground Truth ({proj})",
         f"Predicted (Hungarian-mapped) — acc={eval_metrics['overall_accuracy']:.3f}"]
    ):
        for g, char in enumerate(char_names):
            mask = labels == g
            ax.scatter(e2d[mask, 0], e2d[mask, 1],
                       color=pal[g], s=28, alpha=0.7,
                       label=char, edgecolors="none")
        if "Predicted" in title_str:
            wrong = mapped_labels != gt_int
            ax.scatter(e2d[wrong, 0], e2d[wrong, 1],
                       marker="x", color="red", s=40, linewidths=0.8,
                       alpha=0.6, label="misclassified", zorder=5)
        ax.set_title(title_str, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, loc="best", framealpha=0.6)

    ax_cm = fig.add_subplot(gs[0, 2])
    cm = eval_metrics["confusion"]
    im = ax_cm.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    short = [c[:10] for c in char_names]
    ax_cm.set_xticks(range(n)); ax_cm.set_yticks(range(n))
    ax_cm.set_xticklabels(short, rotation=35, ha="right", fontsize=7)
    ax_cm.set_yticklabels(short, fontsize=7)
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Ground Truth")
    ax_cm.set_title("Confusion Matrix")
    for i in range(n):
        for j in range(n):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       fontsize=7,
                       color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    ax_f1 = fig.add_subplot(gs[1, 0])
    f1v = [eval_metrics["per_char_f1"][c] for c in char_names]
    bars = ax_f1.barh(char_names, f1v, color=[pal[g] for g in range(n)],
                      edgecolor="white", height=0.55)
    ax_f1.set_xlim(0, 1.1)
    ax_f1.axvline(0.75, color="gray", linestyle="--", linewidth=1)
    ax_f1.set_title("Per-Character F1")
    for bar, char in zip(bars, char_names):
        ax_f1.text(bar.get_width() + 0.01,
                   bar.get_y() + bar.get_height() / 2,
                   f"{eval_metrics['per_char_f1'][char]:.3f}",
                   va="center", fontsize=8)

    ax_intra = fig.add_subplot(gs[1, 1])
    valid_c = [(c, sep_metrics["intra"][c]) for c in char_names
               if not np.isnan(sep_metrics["intra"].get(c, float("nan")))]
    if valid_c:
        vc_names, vc_vals = zip(*valid_c)
        vc_colors = [pal[char_names.index(c)] for c in vc_names]
        bars2 = ax_intra.barh(vc_names, vc_vals, color=vc_colors,
                               edgecolor="white", height=0.55)
        ax_intra.set_xlim(0, 1.1)
        ax_intra.axvline(0.75, color="gray", linestyle="--", linewidth=1)
        ax_intra.set_title("GT Intra-Char Consistency")
        for bar, char in zip(bars2, vc_names):
            ax_intra.text(bar.get_width() + 0.01,
                          bar.get_y() + bar.get_height() / 2,
                          f"{sep_metrics['intra'][char]:.3f}",
                          va="center", fontsize=8)

    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    rows = [
        ["Overall accuracy", f"{eval_metrics['overall_accuracy']:.4f}"],
        ["Cluster purity",   f"{eval_metrics['cluster_purity']:.4f}"],
        ["# GT characters",  str(n)],
        ["# segs embedded",  str(len(gt_int))],
        ["", ""],
        ["Character", "P / R / F1"],
    ] + [[c[:16], f"{eval_metrics['per_char_precision'][c]:.2f} / "
                  f"{eval_metrics['per_char_recall'][c]:.2f} / "
                  f"{eval_metrics['per_char_f1'][c]:.2f}"]
         for c in char_names]
    tbl = ax_tbl.table(cellText=rows, cellLoc="left", loc="center",
                        colWidths=[0.62, 0.38])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    ax_tbl.set_title("Summary", pad=8)

    out_path = os.path.join(out_dir, f"{stem}_cluster_eval.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out_path}")

    return pal  # return palette for timeline plot


# ──────────────────────────────────────────────────────────────────────
# 8.  Console report + JSON save
# ──────────────────────────────────────────────────────────────────────

def print_report(eval_m, sep_m, char_names, mapping):
    print("\n" + "═" * 62)
    print("  SPEAKER CLUSTER EVALUATION REPORT")
    print("═" * 62)
    print(f"  Overall accuracy : {eval_m['overall_accuracy']:.4f}")
    print(f"  Cluster purity   : {eval_m['cluster_purity']:.4f}")
    print()
    print(f"  {'Character':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Intra':>8}")
    print("  " + "-" * 52)
    for char in char_names:
        intra = sep_m["intra"].get(char, float("nan"))
        ist   = f"{intra:.3f}" if not np.isnan(intra) else "  n/a"
        print(f"  {char:<22} "
              f"{eval_m['per_char_precision'][char]:>6.3f} "
              f"{eval_m['per_char_recall'][char]:>6.3f} "
              f"{eval_m['per_char_f1'][char]:>6.3f} {ist:>8}")
    print()
    print("  Inter-character distances (GT centroids):")
    for (ca, cb), d in sep_m["inter"].items():
        if ca < cb:
            print(f"    {ca:<20} <-> {cb:<20}  {d:.4f}")
    print()
    print("  Hungarian mapping:")
    for cid, gid in mapping.items():
        print(f"    cluster_{cid}  →  {char_names[gid]}")
    print("═" * 62)


def save_json(eval_m, sep_m, char_names, mapping, n_audio, n_json, stem, out_dir):
    out = {
        "stem": stem,
        "audio_segments_detected": n_audio,
        "json_segments":           n_json,
        "segments_evaluated":      int(len(eval_m["mapped_labels"])),
        "overall_accuracy":        eval_m["overall_accuracy"],
        "cluster_purity":          eval_m["cluster_purity"],
        "characters":              char_names,
        "hungarian_mapping":       {str(k): char_names[v] for k, v in mapping.items()},
        "per_character": {
            c: {
                "precision":   eval_m["per_char_precision"][c],
                "recall":      eval_m["per_char_recall"][c],
                "f1":          eval_m["per_char_f1"][c],
                "intra_consistency": (
                    None if np.isnan(sep_m["intra"].get(c, float("nan")))
                    else sep_m["intra"][c]
                ),
            } for c in char_names
        },
        "inter_character_distances": {
            f"{ca}__vs__{cb}": d
            for (ca, cb), d in sep_m["inter"].items() if ca < cb
        },
        "confusion_matrix": eval_m["confusion"].tolist(),
    }
    path = os.path.join(out_dir, f"{stem}_cluster_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  [saved] {path}")


# ──────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate speaker clustering on a compiled audio file using JSON GT"
    )
    parser.add_argument("--json",             required=True)
    parser.add_argument("--audio",            required=True)
    parser.add_argument("--out_dir",          default="./eval_results")
    parser.add_argument("--n_clusters",       type=int,   default=None)
    parser.add_argument("--min_silence_ms",   type=int,   default=300,
                        help="Minimum silence gap between segments (ms)")
    parser.add_argument("--silence_thresh",   type=float, default=40,
                        help="top_db threshold for librosa silence split")
    parser.add_argument("--save_segments",    action="store_true",
                        help="Save split WAV segments for manual inspection")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = Path(args.audio).stem

    print(f"\n{'═'*62}")
    print(f"  Story  : {stem}")
    print(f"  Audio  : {args.audio}")
    print(f"  JSON   : {args.json}")
    print(f"{'═'*62}")

    # Load GT
    gt_segs    = load_gt_segments(args.json)
    char_names = unique_characters(gt_segs)
    char_to_id = {c: i for i, c in enumerate(char_names)}
    print(f"\n  GT characters ({len(char_names)}):")
    for c in char_names:
        cnt = sum(1 for s in gt_segs if s["character"] == c)
        print(f"    {c:<24} {cnt:>4} segments")

    # Load + split audio
    print(f"\n  Loading audio…")
    wav, sr = librosa.load(args.audio, sr=16000, mono=True)
    print(f"  Duration: {len(wav)/sr:.1f} s")

    print(f"  Splitting on silence (top_db={args.silence_thresh}, "
          f"min_silence={args.min_silence_ms} ms)…")
    audio_segs = split_audio(wav, sr,
                              top_db=args.silence_thresh,
                              min_silence_ms=args.min_silence_ms)

    # Optionally save splits for inspection
    if args.save_segments:
        seg_dir = os.path.join(args.out_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for i, (s, e, w) in enumerate(audio_segs):
            sf.write(os.path.join(seg_dir, f"seg_{i:04d}.wav"), w, sr)
        print(f"  [saved] {len(audio_segs)} segment WAVs → {seg_dir}/")

    # Align to JSON
    wavs, matched_segs = align_segments_to_json(audio_segs, gt_segs)
    gt_int = np.array([char_to_id[s["character"]] for s in matched_segs])

    # Embed
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_encoder(device)
    print("\n  Embedding segments…")
    embeddings, valid_idx = embed_wavs(wavs, encoder)
    gt_int   = gt_int[valid_idx]
    audio_segs_valid = [audio_segs[i] for i in valid_idx]

    # Cluster + evaluate
    n_clusters = args.n_clusters or len(char_names)
    print(f"\n  K-means clustering (k={n_clusters})…")
    cluster_labels = cluster_embeddings(embeddings, n_clusters)
    mapping        = hungarian_match(cluster_labels, gt_int, n_clusters, len(char_names))
    eval_m         = compute_eval_metrics(cluster_labels, gt_int, mapping, char_names)
    sep_m          = compute_separation(embeddings, gt_int, char_names)

    print_report(eval_m, sep_m, char_names, mapping)

    palette = plot_eval_panel(embeddings, gt_int, eval_m["mapped_labels"],
                               eval_m, sep_m, char_names, stem, args.out_dir)
    plot_timeline(audio_segs_valid, gt_int, eval_m["mapped_labels"],
                  char_names, palette, args.out_dir, stem)
    save_json(eval_m, sep_m, char_names, mapping,
              len(audio_segs), len(gt_segs), stem, args.out_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()