"""
paper_metrics.py
================
Generates publication-quality comparison figures for the paper:
  "BERT-driven dynamic emotion conditioning vs. constant emotion token
   baseline in IndicParlerTTS audiobook synthesis."

Four figures are produced:
  1. metric_comparison.pdf   — bar chart: Agreement, Cohen κ, Pearson r, Spearman ρ
  2. emotion_distribution.pdf — emotion token usage distribution (BERT pipeline)
  3. sentiment_timeline.pdf  — per-segment sentiment strip (BERT vs. baseline)
  4. scatter_comparison.pdf  — text score vs. audio valence, both conditions side-by-side

Usage
-----
With real sentiment_eval JSON outputs:
    python paper_metrics.py `
    --bert_json  C:/Users/prana/source/repos/hindi-audiobook-generation/sentiment_eval/results/story_narration_BERT_sentiment_eval.json `
    --baseline_json C:/Users/prana/source/repos/hindi-audiobook-generation/sentiment_eval/results/story_narration_sentiment_eval.json `
    --pipeline_log "C:/Users/prana/source/repos/hindi-audiobook-generation/Audios/Inference/Sentiment Output/HIN_sentiment/pipeline_log.json" `
    --out_dir paper_figures `
    --format png
Demo mode (no real data needed):
    python paper_metrics.py --demo --out_dir paper_figures

Optional flags
    --format   pdf|png  (default: pdf)
    --dpi      300      (default: 300, used for PNG)
    --style    paper|poster  (default: paper)
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────

BERT_COLOR     = "#2563EB"   # blue
BASE_COLOR     = "#DC2626"   # red
NEUTRAL_COLOR  = "#6B7280"   # grey

POLARITY_COLORS = {
    "positive": "#16A34A",
    "neutral":  "#9CA3AF",
    "negative": "#DC2626",
}

EMOTION_COLORS = [
    "#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED",
    "#0891B2", "#DB2777", "#65A30D", "#EA580C", "#0F766E",
    "#9333EA", "#B91C1C",
]

PAPER_RCPARAMS = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_eval_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_aggregate(data: dict) -> dict:
    """Pull aggregate stats from a sentiment_eval JSON."""
    agg = data.get("aggregate", data)
    return {
        "agreement":   agg.get("agreement",    agg.get("polarity_agreement", 0.0)),
        "kappa":       agg.get("cohen_kappa",  agg.get("kappa", 0.0)),
        "pearson_r":   agg.get("pearson_r",    agg.get("r",   0.0)),
        "spearman_r":  agg.get("spearman_r",   agg.get("rho", 0.0)),
        "expr_pearson_r":  agg.get("expressiveness_pearson_r",  0.0),
        "expr_spearman_r": agg.get("expressiveness_spearman_r", 0.0),
        "pos_match":   agg.get("positive_class_match", 0.0),
        "neg_match":   agg.get("negative_class_match", 0.0),
    }


def extract_segment_scores(data: dict):
    """Return (text_numeric, audio_valence, text_polarity, audio_polarity) lists."""
    segs = data.get("segments", [])
    text_scores, audio_scores, text_pol, audio_pol = [], [], [], []
    for s in segs:
        ts = s.get("text_sentiment", s.get("text", {}))
        au = s.get("audio_sentiment", s.get("audio", {}))
        text_scores.append(ts.get("numeric_score", 0.0))
        audio_scores.append(au.get("numeric_score", au.get("valence", 0.5)) * 2 - 1)
        text_pol.append(ts.get("polarity", "neutral"))
        audio_pol.append(au.get("polarity", "neutral"))
    return text_scores, audio_scores, text_pol, audio_pol


def load_emotion_distribution(pipeline_log_path: str) -> dict[str, int]:
    """Count emotion token usage from the pipeline_log.json."""
    with open(pipeline_log_path, encoding="utf-8") as f:
        log = json.load(f)
    counts: dict[str, int] = {}
    chunks = log if isinstance(log, list) else log.get("chunks", [])
    for chunk in chunks:
        em = chunk.get("emotion", "Narration")
        counts[em] = counts.get(em, 0) + 1
    return counts


# ── Demo / synthetic data ──────────────────────────────────────────────────────

def _rng(seed=42):
    return np.random.default_rng(seed)


def make_demo_data(n: int = 80):
    """
    Synthesise plausible per-segment data for a ~80-chunk audiobook.
    BERT pipeline: audio valence tracks text sentiment with noise.
    Baseline: audio valence clusters near neutral (constant Narration token
    produces near-neutral prosody regardless of text content).
    """
    rng = _rng()

    # Text sentiment distribution: skewed positive for a typical folk story
    text_scores = rng.normal(loc=0.18, scale=0.42, size=n).clip(-1, 1)

    # BERT pipeline audio: correlated with text + reasonable noise
    bert_audio = (text_scores * 0.68 + rng.normal(0, 0.22, n)).clip(-1, 1)

    # Constant-baseline audio: near neutral, weak correlation with text
    base_audio = (text_scores * 0.10 + rng.normal(0.04, 0.20, n)).clip(-1, 1)

    def polarity(scores, lo=-0.15, hi=0.15):
        return [
            "negative" if s < lo else ("positive" if s > hi else "neutral")
            for s in scores
        ]

    bert_t_pol = polarity(text_scores)
    bert_a_pol = polarity(bert_audio)
    base_t_pol = polarity(text_scores)
    base_a_pol = polarity(base_audio)

    def agreement_kappa(t, a):
        agree = np.mean([tp == ap for tp, ap in zip(t, a)])
        classes = ["negative", "neutral", "positive"]
        n_total = len(t)
        pe = sum(
            (t.count(c) / n_total) * (a.count(c) / n_total)
            for c in classes
        )
        kappa = (agree - pe) / (1 - pe) if pe < 1 else 0.0
        return float(agree), float(kappa)

    def pearson(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    def spearman(x, y):
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r)

    try:
        from scipy.stats import spearmanr as _sr
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    b_agree, b_kappa = agreement_kappa(bert_t_pol, bert_a_pol)
    c_agree, c_kappa = agreement_kappa(base_t_pol, base_a_pol)
    b_pear = pearson(text_scores, bert_audio)
    c_pear = pearson(text_scores, base_audio)

    if _have_scipy:
        b_spear = spearman(text_scores, bert_audio)
        c_spear = spearman(text_scores, base_audio)
    else:
        b_spear = b_pear * 0.97
        c_spear = c_pear * 0.97

    bert_agg = dict(agreement=b_agree, kappa=b_kappa, pearson_r=b_pear, spearman_r=b_spear,
                    expr_pearson_r=0.30, expr_spearman_r=0.28,
                    pos_match=0.78, neg_match=0.71)
    base_agg = dict(agreement=c_agree, kappa=c_kappa, pearson_r=c_pear, spearman_r=c_spear,
                    expr_pearson_r=0.04, expr_spearman_r=0.05,
                    pos_match=0.62, neg_match=0.58)

    # Emotion token distribution for BERT pipeline
    emotion_counts = {
        "Narration": 28, "Neutral": 14, "Happy": 11, "Sad": 9,
        "Fear": 6, "Anger": 5, "Surprise": 4, "Conversation": 3,
    }

    return (
        bert_agg, base_agg,
        list(text_scores), list(bert_audio), list(base_audio),
        bert_t_pol, bert_a_pol, base_a_pol,
        emotion_counts,
    )


# ── Figure 1: Metric comparison bar chart ─────────────────────────────────────

METRIC_META = [
    ("agreement",  "Polarity Agreement",  True,   "%"),
    ("kappa",      "Cohen's κ",           False,  ""),
    ("pearson_r",  "Pearson r",           False,  ""),
    ("spearman_r", "Spearman ρ",          False,  ""),
    ("pos_match",  "Pos-class Match",     True,   "%"),
    ("neg_match",  "Neg-class Match",     True,   "%"),
]

def plot_metric_comparison(bert_agg: dict, base_agg: dict, out_path: str):
    """
    Bar chart of polarity-alignment metrics on a unified [−0.2, 1.0] axis.
    Percentage-style metrics (agreement, per-class match) are shown on the
    same axis as correlation-style metrics (κ, r, ρ) — interpreted as the
    fraction of agreement / correlation strength on a [0, 1] scale.
    """
    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    n_metrics = len(METRIC_META)
    x = np.arange(n_metrics)
    width = 0.36

    bert_vals = np.array([bert_agg[k] for k, _, _, _ in METRIC_META])
    base_vals = np.array([base_agg[k] for k, _, _, _ in METRIC_META])

    bars_bert = ax.bar(x - width / 2, bert_vals, width, color=BERT_COLOR,
                       label="BERT Sentiment Pipeline", zorder=3)
    bars_base = ax.bar(x + width / 2, base_vals, width, color=BASE_COLOR,
                       label="Constant Emotion Baseline", zorder=3)

    def _fmt(v: float, pct: bool) -> str:
        return f"{v*100:.1f}%" if pct else f"{v:+.3f}"

    for bar, v, (_, _, pct, _) in zip(bars_bert, bert_vals, METRIC_META):
        offset = 0.012 if v >= 0 else -0.045
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset, _fmt(v, pct),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color=BERT_COLOR, fontweight="bold")
    for bar, v, (_, _, pct, _) in zip(bars_base, base_vals, METRIC_META):
        offset = 0.012 if v >= 0 else -0.045
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset, _fmt(v, pct),
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5, color=BASE_COLOR, fontweight="bold")

    # Improvement delta annotations
    for i, ((_, _, pct, _), bv, cv) in enumerate(zip(METRIC_META, bert_vals, base_vals)):
        delta = bv - cv
        sign = "+" if delta >= 0 else ""
        delta_txt = f"{sign}{delta*100:.1f} pp" if pct else f"{sign}{delta:.3f}"
        ax.annotate(
            delta_txt,
            xy=(x[i], max(bv, cv) + 0.07),
            ha="center", fontsize=7.0, color="#111827",
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _, _ in METRIC_META], fontsize=9)
    ax.set_ylabel("Score  (correlations on [−1,1]; percentages on [0,1])")
    ax.set_title("Sentiment Alignment Metrics: BERT Pipeline vs. Constant Emotion Baseline",
                 pad=10)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(-0.15, 1.0)
    ax.axhline(0.0, color="#374151", lw=0.7, zorder=1)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 2: Emotion token distribution ─────────────────────────────────────

def plot_emotion_distribution(emotion_counts: dict[str, int], out_path: str):
    labels = list(emotion_counts.keys())
    sizes  = list(emotion_counts.values())
    total  = sum(sizes)
    colors = EMOTION_COLORS[: len(labels)]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(9, 4))

    # Bar chart (left)
    x = np.arange(len(labels))
    bars = ax_bar.bar(x, sizes, color=colors, zorder=3)
    for bar, s in zip(bars, sizes):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{s / total:.0%}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=35, ha="right")
    ax_bar.set_ylabel("Chunk Count")
    ax_bar.set_title("Emotion Token Frequency")

    # Pie chart (right) — collapse small slices
    threshold = 0.03
    main_labels, main_sizes, other_size = [], [], 0
    for lbl, sz in zip(labels, sizes):
        if sz / total >= threshold:
            main_labels.append(lbl)
            main_sizes.append(sz)
        else:
            other_size += sz
    if other_size:
        main_labels.append("Other")
        main_sizes.append(other_size)

    pie_colors = EMOTION_COLORS[: len(main_labels)]
    wedges, texts, autotexts = ax_pie.pie(
        main_sizes, labels=main_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.78, labeldistance=1.08,
        wedgeprops={"linewidth": 0.6, "edgecolor": "white"},
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Emotion Token Distribution\n(BERT-driven pipeline)")

    # Shannon entropy annotation
    probs = np.array(sizes) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = math.log2(len(labels))
    ax_pie.text(0, -1.35,
                f"Entropy H = {entropy:.2f} bits  (max = {max_entropy:.2f} bits)\n"
                f"Normalised H = {entropy/max_entropy:.2f}",
                ha="center", fontsize=8, color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3F4F6", edgecolor="#D1D5DB"))

    plt.suptitle("Emotion Token Usage — BERT Sentiment Pipeline", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 3: Per-segment sentiment timeline ──────────────────────────────────

def plot_sentiment_timeline(
    text_scores: list,
    bert_audio: list,
    base_audio: list,
    out_path: str,
):
    n = len(text_scores)
    xs = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1]})

    def color_strip(ax, scores, lo=-0.15, hi=0.15, label=""):
        colors = [
            POLARITY_COLORS["negative"] if s < lo
            else (POLARITY_COLORS["positive"] if s > hi else POLARITY_COLORS["neutral"])
            for s in scores
        ]
        for i, (s, c) in enumerate(zip(scores, colors)):
            ax.bar(i, 1, color=c, width=1.0, align="edge", linewidth=0)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=60, fontsize=9)
        neg_p = mpatches.Patch(color=POLARITY_COLORS["negative"],  label="Negative")
        neu_p = mpatches.Patch(color=POLARITY_COLORS["neutral"],   label="Neutral")
        pos_p = mpatches.Patch(color=POLARITY_COLORS["positive"],  label="Positive")
        ax.legend(handles=[neg_p, neu_p, pos_p], loc="upper right",
                  fontsize=7, framealpha=0.85, ncol=3)

    color_strip(axes[0], text_scores,  label="Text\nSentiment")
    color_strip(axes[1], bert_audio,   label="Audio\n(BERT)")
    color_strip(axes[2], base_audio,   label="Audio\n(Baseline)")

    axes[2].set_xlabel("Segment Index")
    fig.suptitle(
        "Per-Segment Sentiment Polarity: Text vs. Synthesised Audio",
        y=1.01, fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure 4: Scatter comparison ──────────────────────────────────────────────

def plot_scatter_comparison(
    text_scores: list,
    bert_audio: list,
    base_audio: list,
    bert_agg: dict,
    base_agg: dict,
    out_path: str,
    base_text_scores: list | None = None,
):
    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    def _scatter(ax, xs, ys, agg, color, title):
        ax.scatter(xs, ys, alpha=0.55, s=22, color=color, zorder=3)
        # Regression line
        m, b_int = np.polyfit(xs, ys, 1)
        xl = np.linspace(-1, 1, 100)
        ax.plot(xl, m * xl + b_int, color=color, lw=1.8, ls="--", zorder=4)
        # Zero reference lines
        ax.axhline(0, color="#9CA3AF", lw=0.7, zorder=1)
        ax.axvline(0, color="#9CA3AF", lw=0.7, zorder=1)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Text Sentiment Score")
        ax.set_title(title)
        stats_str = (
            f"r = {agg['pearson_r']:.3f}\n"
            f"ρ = {agg['spearman_r']:.3f}\n"
            f"κ = {agg['kappa']:.3f}"
        )
        ax.text(0.04, 0.97, stats_str, transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9))

    _scatter(ax_b, text_scores, bert_audio, bert_agg, BERT_COLOR,
             "BERT Sentiment Pipeline")
    _scatter(ax_c, base_text_scores if base_text_scores is not None else text_scores,
             base_audio, base_agg, BASE_COLOR, "Constant Emotion Baseline")

    ax_b.set_ylabel("Audio Valence Score (normalised)")
    fig.suptitle("Text Sentiment vs. Audio Valence Correlation", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bert_json",     default="",
                        help="sentiment_eval JSON for BERT pipeline output")
    parser.add_argument("--baseline_json", default="",
                        help="sentiment_eval JSON for constant-emotion baseline output")
    parser.add_argument("--pipeline_log",  default="",
                        help="pipeline_log.json from sentiment_narrator_*.py")
    parser.add_argument("--out_dir",       default="paper_figures")
    parser.add_argument("--format",        choices=["pdf", "png"], default="pdf")
    parser.add_argument("--dpi",           type=int, default=300)
    parser.add_argument("--demo",          action="store_true",
                        help="Use synthetic demo data (no real JSON needed)")
    args = parser.parse_args()

    plt.rcParams.update(PAPER_RCPARAMS)
    if args.format == "png":
        plt.rcParams["savefig.dpi"] = args.dpi

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def out(name):
        return str(out_dir / f"{name}.{args.format}")

    # ── Load data ──────────────────────────────────────────────────────────────
    base_text_scores = None
    if args.demo or not (args.bert_json and args.baseline_json):
        print("  [demo] Generating synthetic data…")
        (bert_agg, base_agg,
         text_scores, bert_audio, base_audio,
         text_pol, bert_pol, base_pol,
         emotion_counts) = make_demo_data(n=80)
    else:
        bert_data = load_eval_json(args.bert_json)
        base_data = load_eval_json(args.baseline_json)
        bert_agg  = extract_aggregate(bert_data)
        base_agg  = extract_aggregate(base_data)

        text_scores, bert_audio, text_pol, bert_pol = extract_segment_scores(bert_data)
        base_text_scores, base_audio, _, base_pol   = extract_segment_scores(base_data)

        # Truncate all arrays to the shorter run so lengths always match
        n_min = min(len(text_scores), len(base_audio))
        text_scores, bert_audio = text_scores[:n_min], bert_audio[:n_min]
        text_pol,    bert_pol   = text_pol[:n_min],    bert_pol[:n_min]
        base_text_scores = base_text_scores[:n_min]
        base_audio, base_pol = base_audio[:n_min], base_pol[:n_min]

        if args.pipeline_log:
            emotion_counts = load_emotion_distribution(args.pipeline_log)
        else:
            emotion_counts = {"Narration": 1}  # placeholder

    # ── Print aggregate table ──────────────────────────────────────────────────
    print("\n  ┌──────────────────────────────────────────────────────────────┐")
    print("  │  Metric                  BERT Pipeline   Constant Baseline   │")
    print("  ├──────────────────────────────────────────────────────────────┤")
    for key, label, pct, unit in METRIC_META:
        bv = bert_agg[key] * (100 if pct else 1)
        cv = base_agg[key] * (100 if pct else 1)
        delta = bv - cv
        sign = "+" if delta >= 0 else ""
        suffix = "%" if pct else ""
        print(f"  │  {label:<22} {bv:>8.2f}{suffix:<2}     {cv:>8.2f}{suffix:<2}   ({sign}{delta:.2f}) │")
    # Bonus: expressiveness correlation alongside the table.
    if "expr_pearson_r" in bert_agg:
        print("  ├──────────────────────────────────────────────────────────────┤")
        print(f"  │  Expressiveness  r     {bert_agg['expr_pearson_r']:>+8.3f}      "
              f"{base_agg['expr_pearson_r']:>+8.3f}                │")
        print(f"  │  Expressiveness  ρ     {bert_agg['expr_spearman_r']:>+8.3f}      "
              f"{base_agg['expr_spearman_r']:>+8.3f}                │")
    print("  └──────────────────────────────────────────────────────────────┘\n")

    # ── Generate figures ───────────────────────────────────────────────────────
    print("  Generating figures…")
    plot_metric_comparison(bert_agg, base_agg, out("fig1_metric_comparison"))
    plot_emotion_distribution(emotion_counts,   out("fig2_emotion_distribution"))
    plot_sentiment_timeline(text_scores, bert_audio, base_audio,
                            out("fig3_sentiment_timeline"))
    base_ts = base_text_scores if not args.demo and args.baseline_json else None
    plot_scatter_comparison(text_scores, bert_audio, base_audio,
                            bert_agg, base_agg,
                            out("fig4_scatter_comparison"),
                            base_text_scores=base_ts)

    print(f"\n  All figures written to: {out_dir}/\n")


if __name__ == "__main__":
    main()
