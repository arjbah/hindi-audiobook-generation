"""
paper_pipeline_diagram.py
=========================
Generates a publication-quality block diagram of the BERT-driven sentiment
narrator pipeline (sentiment_narrator_hin.py) for the methodology section.

Saves to paper_figures/fig0_pipeline_diagram.{pdf,png}

Usage
-----
    python paper_pipeline_diagram.py --format png
    python paper_pipeline_diagram.py --format pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── Style constants  (kept consistent with paper_metrics.py) ──────────────────

PAPER_RCPARAMS = {
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     11,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
}

# Colour palette — block fill colours are pastel so coloured outlines stand out.
COL_INPUT     = "#E0E7FF"
COL_INPUT_BD  = "#4338CA"
COL_BERT      = "#DBEAFE"
COL_BERT_BD   = "#1D4ED8"
COL_EMOTION   = "#FCE7F3"
COL_EMOTION_BD= "#BE185D"
COL_CAPTION   = "#FEF3C7"
COL_CAPTION_BD= "#B45309"
COL_TTS       = "#DCFCE7"
COL_TTS_BD    = "#15803D"
COL_GATE      = "#FFE4E6"
COL_GATE_BD   = "#BE123C"
COL_OUTPUT    = "#F3F4F6"
COL_OUTPUT_BD = "#374151"
COL_ANCHOR    = "#EDE9FE"
COL_ANCHOR_BD = "#6D28D9"
COL_REGISTRY  = "#FEE2E2"
COL_REGISTRY_BD = "#991B1B"

ARROW_KW   = dict(arrowstyle="-|>,head_width=4,head_length=6",
                  color="#1F2937", lw=1.2, mutation_scale=10)
RETRY_KW   = dict(arrowstyle="-|>,head_width=4,head_length=6",
                  color="#BE123C", lw=1.2, ls="--", mutation_scale=10)
DASH_KW    = dict(arrowstyle="-",
                  color="#6B7280", lw=1.0, ls=":")


def box(ax, xy, w, h, label, fc, ec, *,
        title=None, fontsize=8.5, title_fontsize=9, title_color=None):
    """Draw a rounded rectangle with optional bold title + body text."""
    x, y = xy
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, facecolor=fc, edgecolor=ec, zorder=2,
    )
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    if title:
        ax.text(cx, y + h - 0.18, title,
                ha="center", va="top",
                fontsize=title_fontsize, fontweight="bold",
                color=title_color or ec, zorder=3)
        ax.text(cx, y + h - 0.45, label,
                ha="center", va="top",
                fontsize=fontsize, color="#111827", zorder=3)
    else:
        ax.text(cx, cy, label,
                ha="center", va="center",
                fontsize=fontsize, color="#111827", zorder=3)
    return (x, y, w, h)


def arrow(ax, p_from, p_to, *, label=None, **kwargs):
    """Draw an arrow between two (x, y) points."""
    kw = {**ARROW_KW, **kwargs}
    a = FancyArrowPatch(p_from, p_to, **kw, zorder=4)
    ax.add_patch(a)
    if label:
        mx = (p_from[0] + p_to[0]) / 2
        my = (p_from[1] + p_to[1]) / 2
        ax.text(mx + 0.06, my, label, fontsize=7.5,
                color=kw.get("color", "#1F2937"),
                ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.9))


def draw_pipeline(out_path: str):
    plt.rcParams.update(PAPER_RCPARAMS)
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(6.0, 10.6,
            "BERT-Driven Sentiment-Guided IndicParlerTTS Narrator Pipeline",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color="#111827")
    ax.text(6.0, 10.3,
            "Per-chunk dynamic emotion conditioning with frozen voice anchor + "
            "F0 / ECAPA validation gates",
            ha="center", va="bottom", fontsize=9, color="#374151", style="italic")

    # ── ANCHOR STAGE  (top-left, one-time) ───────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.15, 7.55), 3.85, 2.0,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0, facecolor="#FAF5FF", edgecolor=COL_ANCHOR_BD,
        ls="--", alpha=0.5, zorder=1,
    ))
    ax.text(2.1, 9.42, "Anchor Stage  (one-time)",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=COL_ANCHOR_BD)

    a1 = box(ax, (0.35, 8.55), 1.55, 0.6, "anchor_text\n(seed clip)",
             COL_INPUT, COL_INPUT_BD, fontsize=7.5)
    a2 = box(ax, (2.15, 8.55), 1.7, 0.6, "IndicParlerTTS\n(seed synthesis)",
             COL_TTS, COL_TTS_BD, fontsize=7.5)
    a3 = box(ax, (0.35, 7.70), 1.55, 0.65,
             "ECAPA-TDNN\nspeaker embed", COL_GATE, COL_GATE_BD, fontsize=7.5)
    a4 = box(ax, (2.15, 7.70), 1.7, 0.65,
             "Praat F0 stats\n(mean, std, voiced)", COL_GATE, COL_GATE_BD, fontsize=7.5)

    arrow(ax, (1.9, 8.85), (2.15, 8.85))
    arrow(ax, (3.0, 8.55), (3.0, 8.35))
    arrow(ax, (3.0, 8.30), (1.13, 8.30), arrowstyle="-")
    arrow(ax, (1.13, 8.30), (1.13, 8.35))
    arrow(ax, (3.0, 8.30), (3.0, 8.35))

    # Registry box on the right side of anchor area
    reg = box(ax, (4.30, 8.05), 2.0, 0.95,
              "frozen CVC\n+ anchor F0\n+ anchor embed",
              COL_REGISTRY, COL_REGISTRY_BD, fontsize=7.5,
              title="CVC Registry", title_fontsize=9)

    arrow(ax, (3.85, 8.05), (4.30, 8.45),
          arrowstyle="-|>,head_width=4,head_length=6", color="#6D28D9", lw=1.0)
    arrow(ax, (3.85, 8.30), (4.30, 8.50),
          arrowstyle="-|>,head_width=4,head_length=6", color="#6D28D9", lw=1.0)

    # ── PER-CHUNK LOOP container ─────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.15, 1.05), 11.7, 6.20,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.0, facecolor="#F9FAFB", edgecolor="#374151",
        ls="--", alpha=0.6, zorder=1,
    ))
    ax.text(0.30, 7.10, "Per-Chunk Loop  (n chunks)",
            ha="left", va="top", fontsize=9.5, fontweight="bold",
            color="#374151")

    # ── 1. Story chunker ─────────────────────────────────────────────────────
    b_story = box(ax, (7.4, 8.55), 2.4, 0.6,
                  "Hindi story\n(sentence-segmented)",
                  COL_INPUT, COL_INPUT_BD, fontsize=7.5)
    b_chunker = box(ax, (10.1, 8.55), 1.7, 0.6,
                    "TextChunker\n(targets ~300 chars)",
                    COL_INPUT, COL_INPUT_BD, fontsize=7.5)
    arrow(ax, (9.80, 8.85), (10.10, 8.85))

    # Chunker → loop input
    arrow(ax, (10.95, 8.55), (10.95, 6.95), arrowstyle="-")
    arrow(ax, (10.95, 6.95), (5.85, 6.95), arrowstyle="-")
    arrow(ax, (5.85, 6.95), (5.85, 6.65))

    # ── 2. BERT sentiment classifier ─────────────────────────────────────────
    b_bert = box(ax, (4.20, 5.90), 3.30, 0.75,
                 "tabularisai/multilingual-sentiment-analysis\n"
                 "→ stars ∈ {1..5},  score ∈ [−1, 1],  confidence",
                 COL_BERT, COL_BERT_BD, fontsize=7.6,
                 title="BERT Sentiment Classifier", title_fontsize=8.5)
    arrow(ax, (5.85, 5.90), (5.85, 5.55))

    # ── 3. Emotion Assigner ──────────────────────────────────────────────────
    b_emo = box(ax, (4.20, 4.65), 3.30, 0.90,
                "score-bin → {VeryNeg, Neg, Neutral, Pos, VeryPos}\n"
                "+ Hindi keyword overrides (anger / fear / disgust /\n"
                "happy / surprise / sad / dialogue / imperative / news)",
                COL_EMOTION, COL_EMOTION_BD, fontsize=7.4,
                title="EmotionAssigner", title_fontsize=8.5)

    # Side annotation: emotion vocabulary
    b_vocab = box(ax, (8.10, 4.65), 3.50, 0.90,
                  "Anger | Command | Conversation | Disgust |\n"
                  "Fear | Happy | Narration | Neutral | News |\n"
                  "Proper Noun | Sad | Surprise",
                  "#FFFFFF", "#9CA3AF", fontsize=7.0,
                  title="IndicParler emotion tokens", title_fontsize=8.0,
                  title_color="#374151")
    arrow(ax, (7.50, 5.10), (8.10, 5.10), **DASH_KW)

    arrow(ax, (5.85, 4.65), (5.85, 4.30))
    ax.text(6.0, 4.45, "(emotion_token, expressivity_phrase)",
            fontsize=7, color="#374151", ha="left", va="center", style="italic")

    # ── 4. Caption composer ──────────────────────────────────────────────────
    b_cap = box(ax, (4.20, 3.45), 3.30, 0.85,
                "frozen narrator CVC  +  emotion token  +\n"
                "expressivity phrase\n"
                "→ IndicParler description string",
                COL_CAPTION, COL_CAPTION_BD, fontsize=7.4,
                title="Caption Composer", title_fontsize=8.5)

    # CVC arrives from registry on the left
    arrow(ax, (4.20, 3.85), (3.30, 3.85), arrowstyle="-",
          color="#991B1B", lw=1.0, ls=":")
    arrow(ax, (3.30, 3.85), (3.30, 8.40), arrowstyle="-",
          color="#991B1B", lw=1.0, ls=":")
    ax.text(3.20, 6.10, "frozen CVC\n+ F0 desc",
            fontsize=7, color="#991B1B", ha="right", va="center",
            style="italic", rotation=90)

    arrow(ax, (5.85, 3.45), (5.85, 3.10))

    # ── 5. IndicParlerTTS ────────────────────────────────────────────────────
    b_tts = box(ax, (4.20, 2.30), 3.30, 0.80,
                "ai4bharat/indic-parler-tts\n"
                "(description-conditioned 24 kHz synthesis)",
                COL_TTS, COL_TTS_BD, fontsize=7.6,
                title="IndicParlerTTS", title_fontsize=8.5)
    arrow(ax, (5.85, 2.30), (5.85, 2.05))

    # ── 6. Validation gates  (Praat F0 → ECAPA) ──────────────────────────────
    b_f0  = box(ax, (1.30, 1.20), 2.20, 0.85,
                "drift ≤ 0.15 (mean)\nstd ratio ≤ 0.35\n[~3 ms CPU]",
                COL_GATE, COL_GATE_BD, fontsize=7.4,
                title="Praat F0 Gate", title_fontsize=8.5)
    b_ec  = box(ax, (4.50, 1.20), 2.20, 0.85,
                "cos(chunk, anchor) ≥ τ\n[~45 ms GPU]",
                COL_GATE, COL_GATE_BD, fontsize=7.4,
                title="ECAPA Cosine Gate", title_fontsize=8.5)

    # TTS output → F0 gate
    arrow(ax, (5.85, 2.05), (3.50, 2.05), arrowstyle="-")
    arrow(ax, (3.50, 2.05), (2.40, 2.05))

    # F0 pass → ECAPA
    arrow(ax, (3.50, 1.62), (4.50, 1.62), label="pass")

    # Emit
    b_emit = box(ax, (7.80, 1.20), 2.10, 0.85,
                 "store chunk audio\nappend to story", COL_OUTPUT, COL_OUTPUT_BD,
                 fontsize=7.6, title="Emit", title_fontsize=8.5)
    arrow(ax, (6.70, 1.62), (7.80, 1.62), label="pass")

    # Retry path: F0 fail → patch CVC F0 descriptor → re-synthesise
    arrow(ax, (2.40, 1.20), (2.40, 0.55), **RETRY_KW)
    arrow(ax, (2.40, 0.55), (10.8, 0.55), arrowstyle="-",
          color="#BE123C", lw=1.2, ls="--")
    arrow(ax, (10.8, 0.55), (10.8, 8.35), arrowstyle="-",
          color="#BE123C", lw=1.2, ls="--")
    arrow(ax, (10.8, 4.05), (7.50, 3.85), **RETRY_KW)
    ax.text(6.55, 0.40,
            "F0 fail  →  patch CVC F0 descriptor  →  re-synthesise  (max 2 retries)",
            ha="center", va="center", fontsize=7.5, color="#BE123C", style="italic")

    # ECAPA fail → flag
    b_flag = box(ax, (4.50, 0.05), 2.20, 0.55,
                 "flag chunk + emit last audio\n(after retry exhaustion)",
                 "#FEF2F2", "#7F1D1D", fontsize=7.0)
    arrow(ax, (5.60, 1.20), (5.60, 0.60), **RETRY_KW)
    ax.text(5.78, 0.95, "fail", fontsize=7, color="#BE123C",
            ha="left", va="center", style="italic")

    # ── 7. Final stitch (visual chevron at the right of Emit) ────────────────
    arrow(ax, (9.90, 1.62), (10.80, 1.62))
    b_stitch = box(ax, (10.80, 1.20), 1.05, 0.85,
                   "stitch +\nexport WAV", COL_OUTPUT, COL_OUTPUT_BD,
                   fontsize=7.4)

    # ── Legend (lower-left, tight) ───────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COL_BERT,    edgecolor=COL_BERT_BD,
                       linewidth=1.4, label="Sentiment / language model"),
        mpatches.Patch(facecolor=COL_EMOTION, edgecolor=COL_EMOTION_BD,
                       linewidth=1.4, label="Emotion / token mapping"),
        mpatches.Patch(facecolor=COL_TTS,     edgecolor=COL_TTS_BD,
                       linewidth=1.4, label="TTS synthesis"),
        mpatches.Patch(facecolor=COL_GATE,    edgecolor=COL_GATE_BD,
                       linewidth=1.4, label="Validation gate"),
        mpatches.Patch(facecolor=COL_REGISTRY,edgecolor=COL_REGISTRY_BD,
                       linewidth=1.4, label="Speaker / voice registry"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", ncol=5,
              fontsize=8, framealpha=0.95, edgecolor="#9CA3AF",
              bbox_to_anchor=(0.0, -0.04))

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default="paper_figures")
    parser.add_argument("--format",  choices=["pdf", "png"], default="png")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_pipeline(str(out_dir / f"fig0_pipeline_diagram.{args.format}"))


if __name__ == "__main__":
    main()
