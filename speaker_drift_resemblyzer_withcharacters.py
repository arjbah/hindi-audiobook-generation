r"""
Speaker Drift & Character Identity Evaluation Pipeline
-------------------------------------------------------
Evaluates speaker identity consistency and character voice separation
in long-form TTS audio using Resemblyzer GE2E embeddings.

Designed for the StoriCo use case: a single narrator who modulates
their voice for multiple characters. Automatically detects character
clusters via k-means on speaker embeddings (no manual labels needed),
matching the methodology in Kalyan et al StoriCo paper.

MODES
-----
1. Drift-only (default):
   Segments audio and computes speaker drift from a fixed anchor.

2. Character analysis (with --n_characters N):
   Auto-clusters segments into N voice groups via k-means,
   computes per-character intra-consistency, inter-character
   separation, and an embedding projection scatter plot.
   Needs an accurate character count for best results defaults to 4 because that's common for StoriCo.

USAGE
-----
  # Drift only
  python speaker_drift_resemblyzer.py --audio story.wav

  # Character analysis - auto-detect 4 clusters (StoriCo default)
  python speaker_drift_resemblyzer.py --audio story.wav --n_characters 4

  # Multi-file comparison
  # for the eventual comparison with INdicParler output.
  python speaker_drift_resemblyzer.py --audio a.wav b.wav --compare
  
"""


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from itertools import combinations

import librosa
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


# ======================================================================
# Config
# ======================================================================

@dataclass
class EvalConfig:
    window_sec:  float = 4.0
    stride_sec:  float = 2.0
    anchor_n:    int   = 5
    min_seg_sec: float = 1.5
    device:      str   = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# Audio I/O
# ======================================================================

def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav, sr


def segment_audio(wav: np.ndarray, sr: int, window_sec: float,
                  stride_sec: float, min_seg_sec: float):
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    min_samples    = int(min_seg_sec * sr)
    segments = []
    start = 0
    while start < len(wav):
        end = start + window_samples
        seg = wav[start : min(end, len(wav))]
        if len(seg) < min_samples:
            break
        if len(seg) < window_samples:
            seg = np.pad(seg, (0, window_samples - len(seg)))
        segments.append((start / sr, seg))
        start += stride_samples
    return segments


# ======================================================================
# Embedding extraction
# ======================================================================

def load_encoder(device: str) -> VoiceEncoder:
    print(f"  Loading Resemblyzer GE2E encoder on {device}...")
    return VoiceEncoder(device=device)


def embed_segment(seg: np.ndarray, encoder: VoiceEncoder) -> np.ndarray:
    processed = preprocess_wav(seg, source_sr=16000)
    emb = encoder.embed_utterance(processed)
    return emb / (np.linalg.norm(emb) + 1e-8)


def embed_segments(segments: list, encoder: VoiceEncoder) -> tuple:
    timestamps = np.array([s[0] for s in segments])
    embeddings = np.stack([embed_segment(s[1], encoder) for s in segments])
    return timestamps, embeddings


# ======================================================================
# Drift metrics
# ======================================================================

def compute_drift_metrics(embeddings: np.ndarray, anchor_n: int) -> dict:
    anchor = embeddings[:anchor_n].mean(axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-8)

    anchor_dev = np.array([cosine(anchor, e) for e in embeddings])
    seq_drift  = np.array([cosine(embeddings[i], embeddings[i + 1])
                           for i in range(len(embeddings) - 1)])
    cum_drift  = np.cumsum(seq_drift) / (np.arange(len(seq_drift)) + 1)

    return {
        "anchor_deviation": anchor_dev,
        "sequential_drift": seq_drift,
        "cumulative_drift": cum_drift,
        "anchor_embedding": anchor,
    }


def drift_summary(metrics: dict, timestamps: np.ndarray) -> dict:
    ad    = metrics["anchor_deviation"]
    sd    = metrics["sequential_drift"]
    pidx  = int(np.argmax(ad))
    slope = float(np.polyfit(timestamps, ad, 1)[0])
    return {
        "mean_anchor_deviation": float(np.mean(ad)),
        "max_anchor_deviation":  float(np.max(ad)),
        "std_anchor_deviation":  float(np.std(ad)),
        "peak_drift_time_sec":   float(timestamps[pidx]),
        "drift_rate_per_sec":    slope,
        "mean_sequential_drift": float(np.mean(sd)),
        "max_sequential_drift":  float(np.max(sd)),
        "instability_score":     float(np.var(sd)),
        "n_segments":            int(len(ad)),
    }


# ======================================================================
# Auto character clustering
# ======================================================================

def auto_cluster_characters(embeddings: np.ndarray, timestamps: np.ndarray,
                             n_clusters: int) -> tuple:
    """
    K-means on L2-normalised embeddings, matching StoriCo methodology.
    Clusters renamed by first-appearance order so character_0 is the
    first voice heard (typically the narrator).
    """
    normed     = normalize(embeddings)
    km         = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = km.fit_predict(normed)

    seen = []
    for lbl in raw_labels:
        if lbl not in seen:
            seen.append(lbl)
    remap      = {old: new for new, old in enumerate(seen)}
    labels     = np.array([remap[l] for l in raw_labels])
    char_names = [f"character_{i}" for i in range(n_clusters)]

    print(f"\n  Auto-detected {n_clusters} voice clusters (k-means):")
    for i, name in enumerate(char_names):
        idx   = np.where(labels == i)[0]
        first = float(timestamps[idx[0]])
        print(f"    {name}: {len(idx):>4} segments  (first @ {first:.1f}s)")

    return labels, char_names


def group_embeddings_by_cluster(embeddings: np.ndarray, labels: np.ndarray,
                                 char_names: list) -> dict:
    return {
        char_names[i]: embeddings[labels == i]
        for i in range(len(char_names))
        if np.sum(labels == i) > 0
    }


# ======================================================================
# Character identity metrics
# ======================================================================

def compute_character_metrics(char_embeddings: dict) -> dict:
    char_names = sorted(char_embeddings.keys())
    char_means = {}
    intra      = {}

    for char in char_names:
        embs     = char_embeddings[char]
        mean_emb = embs.mean(axis=0)
        char_means[char] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        if len(embs) < 2:
            intra[char] = 1.0
        else:
            sims = [1 - cosine(embs[i], embs[j])
                    for i, j in combinations(range(len(embs)), 2)]
            intra[char] = float(np.mean(sims))

    inter_matrix = {}
    for ca, cb in combinations(char_names, 2):
        dist = float(cosine(char_means[ca], char_means[cb]))
        inter_matrix[(ca, cb)] = dist
        inter_matrix[(cb, ca)] = dist

    return {
        "char_means":   char_means,
        "intra":        intra,
        "inter_matrix": inter_matrix,
        "char_names":   char_names,
    }


# ======================================================================
# Visualization — drift plots
# ======================================================================

BLUE   = "#2563EB"
RED    = "#DC2626"
PURPLE = "#7C3AED"
GREEN  = "#059669"
AMBER  = "#D97706"
YELLOW = "#F59E0B"


def plot_drift_panel(timestamps, metrics, stats, title, out_dir, stem):
    ad  = metrics["anchor_deviation"]
    sd  = metrics["sequential_drift"]
    cd  = metrics["cumulative_drift"]
    seq_times = timestamps[1:]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(timestamps, ad, color=BLUE, linewidth=1.5, label="Anchor deviation", zorder=3)
    ax1.fill_between(timestamps, ad, alpha=0.08, color=BLUE)
    trend = np.poly1d(np.polyfit(timestamps, ad, 1))(timestamps)
    ax1.plot(timestamps, trend, color=RED, linestyle="--", linewidth=1.2,
             label=f"Trend  (slope = {stats['drift_rate_per_sec']:.5f} / s)")
    pidx = int(np.argmax(ad))
    ax1.axvline(timestamps[pidx], color=YELLOW, linestyle=":", linewidth=1.2,
                label=f"Peak @ {stats['peak_drift_time_sec']:.1f} s")
    ax1.scatter([timestamps[pidx]], [ad[pidx]], color=YELLOW, s=55, zorder=5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Cosine distance from anchor")
    ax1.set_title("Speaker Anchor Deviation Over Time")
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    rows = [
        ["Mean anchor deviation",  f"{stats['mean_anchor_deviation']:.4f}"],
        ["Max anchor deviation",   f"{stats['max_anchor_deviation']:.4f}"],
        ["Std anchor deviation",   f"{stats['std_anchor_deviation']:.4f}"],
        ["Peak drift time",        f"{stats['peak_drift_time_sec']:.1f} s"],
        ["Drift rate",             f"{stats['drift_rate_per_sec']:.5f} / s"],
        ["Mean sequential drift",  f"{stats['mean_sequential_drift']:.4f}"],
        ["Max sequential drift",   f"{stats['max_sequential_drift']:.4f}"],
        ["Instability score",      f"{stats['instability_score']:.6f}"],
        ["Segments evaluated",     str(stats["n_segments"])],
    ]
    tbl = ax2.table(cellText=rows, colLabels=["Metric", "Value"],
                    cellLoc="left", loc="center", colWidths=[0.68, 0.32])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)
    ax2.set_title("Summary Statistics", pad=10)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(seq_times, sd, color=PURPLE, linewidth=1.2, label="Sequential drift")
    ax3.axhline(stats["mean_sequential_drift"], color=AMBER, linestyle="--",
                linewidth=1, label=f"Mean = {stats['mean_sequential_drift']:.4f}")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cosine distance")
    ax3.set_title("Segment-to-Segment Drift")
    ax3.legend(fontsize=8)
    ax3.set_ylim(bottom=0)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(seq_times, cd, color=GREEN, linewidth=1.5, label="Cumulative mean drift")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Running mean cosine distance")
    ax4.set_title("Cumulative Drift (Running Mean)")
    ax4.legend(fontsize=8)
    ax4.set_ylim(bottom=0)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(ad, bins=20, color=BLUE, alpha=0.7, edgecolor="white")
    ax5.axvline(stats["mean_anchor_deviation"], color=RED, linestyle="--",
                linewidth=1.2, label=f"Mean = {stats['mean_anchor_deviation']:.4f}")
    ax5.set_xlabel("Cosine distance from anchor")
    ax5.set_ylabel("Segment count")
    ax5.set_title("Drift Distribution")
    ax5.legend(fontsize=8)

    path = os.path.join(out_dir, f"{stem}_drift.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ======================================================================
# Visualization — character projection scatter
# ======================================================================

def _reduce_dims(all_embs: np.ndarray, centroids: np.ndarray):
    """UMAP > t-SNE > PCA fallback. Returns (all_2d, centroid_2d, method_name)."""
    try:
        from umap import UMAP
        reducer     = UMAP(n_components=2, random_state=42,
                           n_neighbors=min(15, len(all_embs) - 1))
        all_2d      = reducer.fit_transform(all_embs)
        c_norm      = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        centroid_2d = reducer.transform(c_norm)
        return all_2d, centroid_2d, "UMAP"
    except ImportError:
        pass

    try:
        from sklearn.manifold import TSNE
        combined    = np.vstack([all_embs, centroids])
        combined_2d = TSNE(n_components=2, random_state=42,
                           perplexity=min(30, len(combined) - 1)).fit_transform(combined)
        return combined_2d[:len(all_embs)], combined_2d[len(all_embs):], "t-SNE"
    except ImportError:
        pass

    from sklearn.decomposition import PCA
    combined    = np.vstack([all_embs, centroids])
    combined_2d = PCA(n_components=2).fit_transform(combined)
    return combined_2d[:len(all_embs)], combined_2d[len(all_embs):], "PCA"


def plot_character_panel(char_metrics: dict, char_embeddings: dict,
                         labels: np.ndarray, timestamps: np.ndarray,
                         out_dir: str, stem: str):
    char_names = char_metrics["char_names"]
    intra      = char_metrics["intra"]
    inter      = char_metrics["inter_matrix"]
    n          = len(char_names)

    all_embs  = np.vstack([char_embeddings[c] for c in char_names])
    centroids = np.vstack([char_metrics["char_means"][c] for c in char_names])
    all_2d, centroid_2d, proj_method = _reduce_dims(all_embs, centroids)

    # label array aligned with all_embs stacking order
    seg_labels = np.concatenate([
        np.full(len(char_embeddings[c]), i) for i, c in enumerate(char_names)
    ])

    palette    = plt.cm.tab10(np.linspace(0, 1, n))
    char_color = {c: palette[i] for i, c in enumerate(char_names)}

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Character Voice Identity Analysis",
                 fontsize=13, fontweight="bold", y=1.0)
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.55,
                           height_ratios=[8, 1])

    # ── Projection scatter
    ax1 = fig.add_subplot(gs[0, 0])
    for i, char in enumerate(char_names):
        mask = seg_labels == i
        ax1.scatter(all_2d[mask, 0], all_2d[mask, 1],
                    color=char_color[char], s=30, alpha=0.65,
                    label=char, edgecolors="none")
    for i, char in enumerate(char_names):
        ax1.scatter(centroid_2d[i, 0], centroid_2d[i, 1],
                    color=char_color[char], s=220, marker="*",
                    edgecolors="black", linewidths=0.7, zorder=6)
    ax1.set_title(f"Embedding Projections ({proj_method})\nStars = character centroids")
    ax1.set_xlabel(f"{proj_method} dim 1")
    ax1.set_ylabel(f"{proj_method} dim 2")
    ax1.legend(title="Cluster", fontsize=8, title_fontsize=8,
               loc="best", framealpha=0.7)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # ── Timeline colour strip
    ax_strip = fig.add_subplot(gs[1, 0])
    t_range  = float(timestamps[-1] - timestamps[0]) or 1.0
    for i in range(len(timestamps)):
        t     = timestamps[i]
        width = (timestamps[i + 1] - t) if i + 1 < len(timestamps) else t_range / len(timestamps)
        ax_strip.axvspan(t, t + width, color=palette[labels[i]], alpha=0.9)
    ax_strip.set_xlim(timestamps[0], timestamps[-1])
    ax_strip.set_yticks([])
    ax_strip.set_xlabel("Time (s)")
    ax_strip.set_title("Cluster sequence over time", fontsize=8)

    # ── Intra-consistency bars
    ax2 = fig.add_subplot(gs[0, 1])
    colors_bar = [char_color[c] for c in char_names]
    bars = ax2.barh(char_names, [intra[c] for c in char_names],
                    color=colors_bar, edgecolor="white", height=0.55)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(0.75, color="gray", linestyle="--", linewidth=1, label="0.75 reference")
    ax2.set_xlabel("Mean pairwise cosine similarity (within cluster)")
    ax2.set_title("Intra-Character Voice Consistency\n(higher = more consistent across story)")
    ax2.legend(fontsize=8)
    for bar, char in zip(bars, char_names):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{intra[char]:.3f}", va="center", fontsize=8)

    # ── Inter-character distance table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    pair_rows = [[ca, cb, f"{inter[(ca, cb)]:.4f}"]
                 for ca, cb in combinations(char_names, 2)]
    if pair_rows:
        tbl = ax3.table(cellText=pair_rows,
                        colLabels=["Char A", "Char B", "Cosine dist"],
                        cellLoc="center", loc="center",
                        colWidths=[0.3, 0.3, 0.4])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)
    ax3.set_title("Inter-Character Distances\n(higher = more distinct)",
                  fontsize=8, pad=4)

    path = os.path.join(out_dir, f"{stem}_characters.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
    if proj_method == "PCA":
        print("  [note] pip install umap-learn for higher quality projections")


# ======================================================================
# Visualization — multi-file comparison
# ======================================================================

def plot_comparison(results: list, out_dir: str):
    colors = [BLUE, RED, GREEN, AMBER, PURPLE]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Speaker Drift — Multi-File Comparison", fontsize=13, fontweight="bold")

    for i, (label, timestamps, metrics) in enumerate(results):
        c = colors[i % len(colors)]
        axes[0].plot(timestamps, metrics["anchor_deviation"],
                     color=c, linewidth=1.5, label=label, alpha=0.85)
        axes[1].plot(timestamps[1:], metrics["cumulative_drift"],
                     color=c, linewidth=1.5, label=label, alpha=0.85)

    for ax, title, ylabel in zip(
        axes,
        ["Anchor Deviation Over Time", "Cumulative Drift (Running Mean)"],
        ["Cosine distance from anchor", "Running mean cosine distance"],
    ):
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_drift.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ======================================================================
# Console reporting
# ======================================================================

def print_character_report(char_metrics: dict):
    char_names = char_metrics["char_names"]
    intra      = char_metrics["intra"]
    inter      = char_metrics["inter_matrix"]

    print("\n  Character Identity Report")
    print(f"  {'Character':<20} {'Intra-consistency':>18}")
    print("  " + "-" * 42)
    for char in char_names:
        print(f"  {char:<20} {intra[char]:>18.4f}")

    print("\n  Inter-Character Distance (higher = more distinct)")
    for ca, cb in combinations(char_names, 2):
        print(f"  {ca:<18} <-> {cb:<18}  dist = {inter[(ca, cb)]:.4f}")


# ======================================================================
# Main pipeline
# ======================================================================

def run_eval(audio_path: str, cfg: EvalConfig, encoder: VoiceEncoder,
             out_dir: str, n_characters: Optional[int] = None):
    stem = Path(audio_path).stem
    print(f"\n-- Evaluating: {stem}")

    wav, sr  = load_audio(audio_path)
    duration = len(wav) / sr
    print(f"  Duration: {duration:.1f} s  |  Device: {cfg.device}")

    segments = segment_audio(wav, sr, cfg.window_sec, cfg.stride_sec, cfg.min_seg_sec)
    print(f"  Segments: {len(segments)} "
          f"(window={cfg.window_sec}s, stride={cfg.stride_sec}s)")

    print("  Extracting embeddings...")
    timestamps, embeddings = embed_segments(segments, encoder)

    drift_metrics = compute_drift_metrics(embeddings, cfg.anchor_n)
    stats         = drift_summary(drift_metrics, timestamps)

    print("\n  Drift Results:")
    for k, v in stats.items():
        print(f"    {k:<30} {v}")

    plot_drift_panel(timestamps, drift_metrics, stats,
                     title=f"Speaker Drift (Resemblyzer) -- {stem}",
                     out_dir=out_dir, stem=stem)

    if n_characters is not None:
        labels, char_names  = auto_cluster_characters(embeddings, timestamps, n_characters)
        char_embeddings     = group_embeddings_by_cluster(embeddings, labels, char_names)
        char_metrics        = compute_character_metrics(char_embeddings)
        print_character_report(char_metrics)
        plot_character_panel(char_metrics, char_embeddings,
                             labels, timestamps,
                             out_dir=out_dir, stem=stem)

    return stem, timestamps, drift_metrics, stats


def main():
    parser = argparse.ArgumentParser(
        description="Speaker drift & character identity evaluation (Resemblyzer)"
    )
    parser.add_argument("--audio",        nargs="+", required=True)
    parser.add_argument("--n_characters", type=int,  default=None,
                        help="Number of voice clusters to auto-detect (e.g. 4). "
                             "Omit to run drift-only.")
    parser.add_argument("--window",       type=float, default=4.0)
    parser.add_argument("--stride",       type=float, default=2.0)
    parser.add_argument("--anchor_n",     type=int,   default=5)
    parser.add_argument("--compare",      action="store_true")
    parser.add_argument("--out_dir",      type=str,   default="./drift_results")
    args = parser.parse_args()

    cfg = EvalConfig(window_sec=args.window, stride_sec=args.stride, anchor_n=args.anchor_n)
    os.makedirs(args.out_dir, exist_ok=True)
    encoder   = load_encoder(cfg.device)
    all_drift = []

    for audio_path in args.audio:
        stem, timestamps, drift_metrics, stats = run_eval(
            audio_path, cfg, encoder,
            out_dir=args.out_dir,
            n_characters=args.n_characters,
        )
        all_drift.append((stem, timestamps, drift_metrics))

    if args.compare and len(all_drift) > 1:
        plot_comparison(all_drift, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()