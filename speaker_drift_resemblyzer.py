import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import librosa
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class EvalConfig:
    window_sec: float = 4.0       # embedding window size in seconds
    stride_sec: float = 2.0       # hop between windows
    anchor_n: int = 5             # number of leading segments to average as anchor
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Audio segmentation
# ──────────────────────────────────────────────

def segment_audio(wav: np.ndarray, sr: int, window_sec: float, stride_sec: float):
    """
    Returns list of (start_time, segment_array) tuples.
    Discards trailing segments shorter than half a window.
    """
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    segments = []

    start = 0
    while start + window_samples // 2 < len(wav):
        end = min(start + window_samples, len(wav))
        seg = wav[start:end]
        # pad short trailing segment to full window with silence
        if len(seg) < window_samples:
            seg = np.pad(seg, (0, window_samples - len(seg)))
        segments.append((start / sr, seg))
        start += stride_samples

    return segments


# ──────────────────────────────────────────────
# Embedding extraction
# ──────────────────────────────────────────────

def extract_embeddings(segments, encoder: VoiceEncoder):
    """
    Extracts 256-dim d-vector per segment using Resemblyzer GE2E encoder.
    Returns (timestamps, embedding_matrix) where matrix is (N, 256).
    """
    timestamps = []
    embeddings = []

    for t, seg in segments:
        # Resemblyzer expects 16kHz mono float32
        emb = encoder.embed_utterance(seg)
        timestamps.append(t)
        embeddings.append(emb)

    return np.array(timestamps), np.stack(embeddings)  # (N,), (N, 256)


# ──────────────────────────────────────────────
# Drift metrics
# ──────────────────────────────────────────────

def compute_drift_metrics(embeddings: np.ndarray, anchor_n: int):
    """
    Returns dict with:
      - anchor_deviation: cosine distance from anchor per segment
      - sequential_drift: cosine distance between consecutive segments
      - anchor_embedding: the anchor vector used
    """
    # Anchor = mean of first anchor_n segments
    anchor = embeddings[:anchor_n].mean(axis=0)
    anchor = anchor / np.linalg.norm(anchor)

    anchor_dev = np.array([
        cosine(anchor, emb) for emb in embeddings
    ])

    seq_drift = np.array([
        cosine(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ])

    return {
        "anchor_deviation": anchor_dev,
        "sequential_drift": seq_drift,
        "anchor_embedding": anchor,
    }


def summary_stats(metrics: dict):
    ad = metrics["anchor_deviation"]
    sd = metrics["sequential_drift"]
    return {
        "mean_anchor_deviation":   float(np.mean(ad)),
        "max_anchor_deviation":    float(np.max(ad)),
        "std_anchor_deviation":    float(np.std(ad)),
        "mean_sequential_drift":   float(np.mean(sd)),
        "max_sequential_drift":    float(np.max(sd)),
        "instability_score":       float(np.var(sd)),  # variance of local drift
        "n_segments":              len(ad),
    }


# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────

def plot_drift(timestamps, metrics, title, save_path: Optional[str] = None):
    ad = metrics["anchor_deviation"]
    sd = metrics["sequential_drift"]
    stats = summary_stats(metrics)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Anchor deviation over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, ad, color="#2563EB", linewidth=1.5, label="Anchor deviation")
    ax1.axhline(stats["mean_anchor_deviation"], color="#DC2626", linestyle="--",
                linewidth=1, label=f"Mean = {stats['mean_anchor_deviation']:.4f}")
    ax1.fill_between(timestamps, ad, alpha=0.1, color="#2563EB")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Cosine distance from anchor")
    ax1.set_title("Speaker Anchor Deviation Over Time")
    ax1.legend(fontsize=9)
    ax1.set_ylim(bottom=0)

    # ── Sequential drift over time
    ax2 = fig.add_subplot(gs[1, 0])
    seq_times = timestamps[1:]  # one fewer point
    ax2.plot(seq_times, sd, color="#7C3AED", linewidth=1.2, label="Sequential drift")
    ax2.axhline(stats["mean_sequential_drift"], color="#D97706", linestyle="--",
                linewidth=1, label=f"Mean = {stats['mean_sequential_drift']:.4f}")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cosine distance")
    ax2.set_title("Segment-to-Segment Drift")
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0)

    # ── Summary stats table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Mean anchor deviation",  f"{stats['mean_anchor_deviation']:.4f}"],
        ["Max anchor deviation",   f"{stats['max_anchor_deviation']:.4f}"],
        ["Std anchor deviation",   f"{stats['std_anchor_deviation']:.4f}"],
        ["Mean sequential drift",  f"{stats['mean_sequential_drift']:.4f}"],
        ["Max sequential drift",   f"{stats['max_sequential_drift']:.4f}"],
        ["Instability score",      f"{stats['instability_score']:.6f}"],
        ["Segments evaluated",     str(stats["n_segments"])],
    ]
    table = ax3.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.65, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax3.set_title("Summary Statistics", pad=12)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    else:
        plt.show()

    plt.close()
    return stats


def plot_comparison(results: list, save_path: Optional[str] = None):
    """
    Overlay anchor deviation curves from multiple files on one plot.
    results: list of (label, timestamps, metrics) tuples
    """
    colors = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (label, timestamps, metrics) in enumerate(results):
        ad = metrics["anchor_deviation"]
        ax.plot(timestamps, ad, color=colors[i % len(colors)],
                linewidth=1.5, label=label, alpha=0.85)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cosine distance from anchor")
    ax.set_title("Speaker Anchor Deviation — Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")
    else:
        plt.show()
    plt.close()


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def run_eval(audio_path: str, cfg: EvalConfig, encoder: VoiceEncoder, out_dir: str):
    name = Path(audio_path).stem
    print(f"\n── Evaluating: {name}")

    # Load + resample to 16kHz mono (Resemblyzer requirement)
    print("  Loading audio...")
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(wav) / sr
    print(f"  Duration: {duration:.1f}s")

    # Segment
    segments = segment_audio(wav, sr=16000,
                             window_sec=cfg.window_sec,
                             stride_sec=cfg.stride_sec)
    print(f"  Segments: {len(segments)} "
          f"(window={cfg.window_sec}s, stride={cfg.stride_sec}s)")

    # Embeddings
    print("  Extracting embeddings...")
    segs_only = [s[1] for s in segments]
    # Resemblyzer preprocess_wav expects 16kHz, run per-segment
    timestamps = np.array([s[0] for s in segments])
    embeddings = np.stack([
        encoder.embed_utterance(preprocess_wav(seg, source_sr=16000))
        for seg in segs_only
    ])

    # Metrics
    metrics = compute_drift_metrics(embeddings, anchor_n=cfg.anchor_n)
    stats = summary_stats(metrics)

    print("\n  Results:")
    for k, v in stats.items():
        print(f"    {k:<28} {v}")

    # Plot
    plot_path = os.path.join(out_dir, f"{name}_drift.png")
    plot_drift(timestamps, metrics, title=f"Speaker Drift — {name}", save_path=plot_path)

    return name, timestamps, metrics, stats


def main():
    parser = argparse.ArgumentParser(description="Speaker drift evaluation for TTS audio")
    parser.add_argument("--audio", nargs="+", required=True,
                        help="Path(s) to .wav audio file(s)")
    parser.add_argument("--window", type=float, default=4.0,
                        help="Embedding window size in seconds (default: 4.0)")
    parser.add_argument("--stride", type=float, default=2.0,
                        help="Stride between windows in seconds (default: 2.0)")
    parser.add_argument("--anchor_n", type=int, default=5,
                        help="Number of leading segments to use as anchor (default: 5)")
    parser.add_argument("--compare", action="store_true",
                        help="Generate overlay comparison plot for multiple files")
    parser.add_argument("--out_dir", type=str, default="./drift_results",
                        help="Output directory for plots (default: ./drift_results)")
    args = parser.parse_args()

    cfg = EvalConfig(
        window_sec=args.window,
        stride_sec=args.stride,
        anchor_n=args.anchor_n,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {cfg.device}")
    print(f"Output dir: {args.out_dir}")

    print("\nLoading speaker encoder...")
    encoder = VoiceEncoder(device=cfg.device)

    all_results = []
    for audio_path in args.audio:
        name, timestamps, metrics, stats = run_eval(audio_path, cfg, encoder, args.out_dir)
        all_results.append((name, timestamps, metrics))

    if args.compare and len(all_results) > 1:
        compare_path = os.path.join(args.out_dir, "comparison_drift.png")
        plot_comparison(all_results, save_path=compare_path)

    print("\nDone.")


if __name__ == "__main__":
    main()