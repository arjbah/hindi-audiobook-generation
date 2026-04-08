"""
sentiment_eval.py
==================
Evaluates sentiment alignment between input text and synthesized audio
for audiobook segments using a matched-pair design.

Text Sentiment  : tabularisai/multilingual-sentiment-analysis (multilingual BERT)
                  → 5-class (1–5 stars = Very Negative … Very Positive)
Audio Sentiment : audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
                  → dimensional (valence, arousal, dominance ∈ [0,1])
                  → valence maps directly to sentiment polarity

Matched-pair design
-------------------
Both modalities are mapped to a 3-class polarity label:
  negative | neutral | positive

Agreement, Cohen's κ, Pearson r, and Spearman ρ are computed over
all segment pairs.

Inputs (choose one text source)
  --json       Captions JSON (story120_multi_captions.json format with 'text' field)
  --text       Plain text file — split on sentence-ending punctuation (. ! ? ।)

Audio
  --audio      Compiled/stitched audio file (WAV or MP3)
               Segments are aligned to text entries 1-to-1 in order, via
               silence-based VAD (Silero → librosa fallback).

Usage
-----
python sentiment_eval/sentiment_eval.py `
    --json  transcripts/story120_multi_captions.json `
    --audio Audios/story120_multi.wav `
    --out_dir sentiment_eval/results

python sentiment_eval/sentiment_eval.py `
    --text  transcripts/bad_blood_eng.txt `
    --audio Audios/Inference/IndicParler/ENG/elevenlabs_vampire_story.mp3 `
    --out_dir sentiment_eval/results

Optional flags
  --min_silence_ms  300    silence gap to split audio on (ms)
  --silence_thresh  40     top_db threshold for librosa silence split
  --valence_low     0.4    valence below this → negative
  --valence_high    0.6    valence above this → positive
  --save_segments          dump split WAV segments for inspection

Outputs  (all in --out_dir)
  <stem>_sentiment_eval.json       per-segment scores + aggregate stats
  <stem>_sentiment_timeline.png    colour strip: text vs audio sentiment over time
  <stem>_sentiment_scatter.png     scatter: text score vs audio valence
  <stem>_sentiment_confusion.png   confusion matrix (text polarity vs audio polarity)
  <stem>_sentiment_distribution.png polarity distribution comparison bar chart
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from transformers import (
    Wav2Vec2Processor,
    pipeline as hf_pipeline,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# ── Constants ──────────────────────────────────────────────────────────────────
TEXT_MODEL_ID  = "tabularisai/multilingual-sentiment-analysis"
AUDIO_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# Map 1–5 star labels to [-1, 1] numeric scores
STAR_TO_NUMERIC = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
POLARITY_LABELS = ["negative", "neutral", "positive"]

# tabularisai/multilingual-sentiment-analysis emits verbose labels like
# "Very Negative", "Negative", "Neutral", "Positive", "Very Positive".
# Normalise to an integer rank 1–5 regardless of format.
_LABEL_TO_STARS: dict[str, int] = {
    # verbose form (actual model output)
    "very negative": 1,
    "negative":      2,
    "neutral":       3,
    "positive":      4,
    "very positive": 5,
    # star form (older / fine-tuned variants)
    "1 star":  1,
    "2 stars": 2,
    "3 stars": 3,
    "4 stars": 4,
    "5 stars": 5,
    # LABEL_x form (some HF checkpoints)
    "label_0": 1,
    "label_1": 2,
    "label_2": 3,
    "label_3": 4,
    "label_4": 5,
}

# Colours for polarity classes (red / gray / green)
_POL_COLORS = {"negative": "#DC2626", "neutral": "#9CA3AF", "positive": "#059669"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Audio sentiment model  (audeering wav2vec2 dimensional emotion)
# ══════════════════════════════════════════════════════════════════════════════

class _RegressionHead(nn.Module):
    """Regression head as defined in the audeering model card."""

    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2PreTrainedModel):
    """wav2vec2 + regression head for valence/arousal/dominance prediction."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2   = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(
        self, input_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_hidden = self.wav2vec2(input_values).last_hidden_state   # (B, T, H)
        pooled      = last_hidden.mean(dim=1)                          # (B, H)
        logits      = self.classifier(pooled)                          # (B, 3)
        return pooled, logits


def load_audio_sentiment_model(device: str) -> tuple:
    """Download and return (processor, model) for the audeering emotion model."""
    print(f"  Loading audio sentiment model ({AUDIO_MODEL_ID}) on {device}…")
    processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_ID)
    model     = _EmotionModel.from_pretrained(AUDIO_MODEL_ID).to(device).eval()
    return processor, model


@torch.no_grad()
def predict_audio_emotion(
    wav: np.ndarray,
    sr: int,
    processor,
    model: _EmotionModel,
    device: str,
) -> dict:
    """
    Predict dimensional emotion from a raw waveform.

    Returns
    -------
    dict with keys: valence, arousal, dominance  (each in [0, 1])
        valence < 0.5  → negative sentiment
        valence > 0.5  → positive sentiment
    """
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    inputs       = processor(wav, sampling_rate=16000,
                              return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    _, logits    = model(input_values)
    vals         = logits.squeeze().cpu().numpy()
    # logits order: [valence, arousal, dominance]
    return {
        "valence":   float(np.clip(vals[0], 0.0, 1.0)),
        "arousal":   float(np.clip(vals[1], 0.0, 1.0)),
        "dominance": float(np.clip(vals[2], 0.0, 1.0)),
    }


def valence_to_polarity(
    valence: float,
    low: float = 0.4,
    high: float = 0.6,
) -> str:
    if valence < low:
        return "negative"
    if valence > high:
        return "positive"
    return "neutral"


def valence_to_numeric(valence: float) -> float:
    """Map [0, 1] valence to [-1, 1] so it is comparable with text scores."""
    return (valence - 0.5) * 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Text sentiment  (tabularisai multilingual BERT)
# ══════════════════════════════════════════════════════════════════════════════

def load_text_sentiment_pipeline(device_id: int):
    print(f"  Loading text sentiment model ({TEXT_MODEL_ID})…")
    return hf_pipeline(
        "text-classification",
        model=TEXT_MODEL_ID,
        device=device_id,
        truncation=True,
        max_length=512,
    )


def predict_text_sentiment(text: str, pipe) -> dict:
    """
    Returns
    -------
    dict with keys: label, confidence, stars, numeric_score, polarity
        label         : raw label string from the model
        stars         : int 1–5  (normalised from whatever label format the model uses)
        numeric_score : float in [-1, 1]  (1→-1, 2→-0.5, 3→0, 4→0.5, 5→1)
        polarity      : "negative" | "neutral" | "positive"
    """
    result = pipe(text)[0]
    label  = result["label"]
    conf   = float(result["score"])
    stars  = _label_to_stars(label)
    return {
        "label":         label,
        "confidence":    conf,
        "stars":         stars,
        "numeric_score": STAR_TO_NUMERIC[stars],
        "polarity":      _stars_to_polarity(stars),
    }


def _label_to_stars(label: str) -> int:
    """
    Map whatever label string the model returns to an integer rank 1–5.
    Handles "Very Negative"/"Negative"/… as well as "1 star"/"2 stars"/…
    and LABEL_0/LABEL_1/… forms.  Falls back to 3 (neutral) on unknown labels.
    """
    stars = _LABEL_TO_STARS.get(label.lower().strip())
    if stars is not None:
        return stars
    # Last-resort: try to parse a leading digit ("4 stars" → 4)
    try:
        return int(label.strip().split()[0])
    except (ValueError, IndexError):
        return 3  # treat completely unknown label as neutral


def _stars_to_polarity(stars: int) -> str:
    if stars <= 2:
        return "negative"
    if stars == 3:
        return "neutral"
    return "positive"


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Text segmentation
# ══════════════════════════════════════════════════════════════════════════════

# Split after sentence-ending punctuation followed by whitespace (or end-of-string)
# Handles English (. ! ?) and Hindi (।)
_SENT_RE = re.compile(r'(?<=[.!?।])\s+|(?<=[.!?।])$')


def split_text_to_segments(text: str) -> list[str]:
    """
    Segment a plain text into sentence-level units by splitting on
    sentence-ending punctuation (. ! ? । and their Unicode variants).
    Empty and whitespace-only tokens are dropped.
    """
    raw = _SENT_RE.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def load_segments_from_json(json_path: str) -> list[dict]:
    """
    Load text segments from story120_multi_captions.json format.
    Each entry must have a 'text' field.  Optional 'segment_id' is preserved.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, entry in enumerate(data):
        text = entry.get("text", "").strip()
        if text:
            out.append({
                "segment_id": entry.get("segment_id", str(i)),
                "text":       text,
            })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Audio loading + VAD splitting  (mirrors speaker_cluster_eval.py strategy)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_split_audio(
    audio_path: str,
    top_db: float = 40,
    min_silence_ms: int = 300,
) -> tuple[np.ndarray, int, list[tuple]]:
    """
    Load audio at 16 kHz mono and split into speech segments.
    Returns (full_waveform, sample_rate, list_of_(start_s, end_s, wav_array)).
    Tries Silero-VAD first, falls back to librosa energy-based split.
    """
    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"  Duration : {len(wav)/sr:.1f} s")

    segs = _split_with_silero(wav, sr, min_silence_ms=min_silence_ms)
    if segs is None:
        segs = _split_librosa(wav, sr, top_db=top_db, min_silence_ms=min_silence_ms)
    return wav, sr, segs


def _split_librosa(
    wav: np.ndarray,
    sr: int,
    top_db: float = 40,
    min_silence_ms: int = 300,
) -> list[tuple]:
    intervals = librosa.effects.split(
        wav, top_db=top_db, frame_length=512, hop_length=128
    )
    min_sil = int(min_silence_ms / 1000 * sr)
    merged  = []
    for s, e in intervals:
        if merged and (s - merged[-1][1]) < min_sil:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if (e - s) >= int(0.3 * sr):
            out.append((s / sr, e / sr, wav[s:e]))
    print(f"  [librosa split] {len(out)} segments")
    return out


def _split_with_silero(
    wav: np.ndarray,
    sr: int,
    min_silence_ms: int = 300,
) -> list[tuple] | None:
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
        )
        (get_speech_timestamps, *_) = utils
        ts = get_speech_timestamps(
            torch.from_numpy(wav).float(),
            model,
            sampling_rate=sr,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=300,
        )
        out = [
            (t["start"] / sr, t["end"] / sr, wav[t["start"]:t["end"]])
            for t in ts
        ]
        print(f"  [Silero-VAD] {len(out)} segments")
        return out
    except Exception as e:
        print(f"  [Silero-VAD] unavailable ({e}), falling back to librosa")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Matched-pair analysis
# ══════════════════════════════════════════════════════════════════════════════

def matched_pair_analysis(
    text_results: list[dict],
    audio_results: list[dict],
    valence_low: float = 0.4,
    valence_high: float = 0.6,
) -> dict:
    """
    Compute alignment statistics between text and audio sentiment results.

    Returns aggregate dict with:
      agreement, cohen_kappa, pearson_r/p, spearman_r/p,
      polarity distributions, confusion matrix.
    """
    n = len(text_results)
    assert n == len(audio_results), "Unequal number of text/audio results"

    text_scores  = np.array([r["numeric_score"] for r in text_results])
    audio_scores = np.array([valence_to_numeric(r["valence"]) for r in audio_results])

    text_pols  = [r["polarity"] for r in text_results]
    audio_pols = [
        valence_to_polarity(r["valence"], valence_low, valence_high)
        for r in audio_results
    ]

    agreement = sum(tp == ap for tp, ap in zip(text_pols, audio_pols)) / n
    kappa     = cohen_kappa_score(text_pols, audio_pols)

    pr, pp = pearsonr(text_scores, audio_scores)
    sr_, sp = spearmanr(text_scores, audio_scores)

    cm = confusion_matrix(text_pols, audio_pols, labels=POLARITY_LABELS)

    return {
        "n_segments":         n,
        "agreement":          float(agreement),
        "cohen_kappa":        float(kappa),
        "pearson_r":          float(pr),
        "pearson_p":          float(pp),
        "spearman_r":         float(sr_),
        "spearman_p":         float(sp),
        "text_polarity_dist": {p: text_pols.count(p) for p in POLARITY_LABELS},
        "audio_polarity_dist":{p: audio_pols.count(p) for p in POLARITY_LABELS},
        "confusion_matrix":   cm.tolist(),
        # carry forward for plots
        "_text_pols":         text_pols,
        "_audio_pols":        audio_pols,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeline(
    text_results: list[dict],
    audio_results: list[dict],
    audio_segs: list[tuple],
    valence_low: float,
    valence_high: float,
    stem: str,
    out_dir: str,
) -> None:
    """
    Two colour strips (text sentiment, audio sentiment) over time,
    coloured by polarity.
    """
    n = len(text_results)
    if audio_segs:
        starts = [s[0] for s in audio_segs[:n]]
        ends   = [s[1] for s in audio_segs[:n]]
    else:
        starts = list(range(n))
        ends   = [x + 1 for x in starts]

    text_pols  = [r["polarity"] for r in text_results]
    audio_pols = [
        valence_to_polarity(r["valence"], valence_low, valence_high)
        for r in audio_results
    ]

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 4),
        gridspec_kw={"height_ratios": [1, 1, 0.45]}
    )
    fig.suptitle(
        f"Sentiment Alignment Timeline: Text vs Audio  —  {stem}",
        fontsize=11, fontweight="bold"
    )

    for ax, pols, title in zip(
        axes[:2],
        [text_pols, audio_pols],
        ["Text Sentiment (BERT)", "Audio Sentiment (wav2vec2 valence)"],
    ):
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(s, e, color=_POL_COLORS[pols[i]], alpha=0.85)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=8)
        ax.set_xlim(starts[0], ends[-1])
        ax.set_xlabel("Time (s)" if title.startswith("Audio") else "")

    # Legend
    axes[2].axis("off")
    for pol, color in _POL_COLORS.items():
        axes[2].bar(0, 0, color=color, label=pol)
    axes[2].legend(loc="center", ncol=3, fontsize=10, framealpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_sentiment_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_scatter(
    text_results: list[dict],
    audio_results: list[dict],
    stats: dict,
    stem: str,
    out_dir: str,
) -> None:
    """Scatter plot of text numeric score vs audio valence, by polarity class."""
    ts = np.array([r["numeric_score"] for r in text_results])
    av = np.array([valence_to_numeric(r["valence"]) for r in audio_results])

    fig, ax = plt.subplots(figsize=(7, 6))
    text_pols = [r["polarity"] for r in text_results]

    for pol, color in _POL_COLORS.items():
        mask = np.array([p == pol for p in text_pols])
        ax.scatter(ts[mask], av[mask], c=color, label=pol, alpha=0.7, s=55, zorder=3)

    # Regression line — guarded against zero-variance / SVD failure
    if len(ts) >= 2 and np.std(ts) > 0 and np.std(av) > 0:
        try:
            m, b = np.polyfit(ts, av, 1)
            x_line = np.linspace(ts.min(), ts.max(), 200)
            ax.plot(x_line, m * x_line + b, "k--", linewidth=1.2, alpha=0.5,
                    label=f"fit  (slope={m:.2f})")
        except (np.linalg.LinAlgError, ValueError):
            pass  # skip fit line if data is degenerate

    ax.axhline(0, color="gray", linewidth=0.5, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, zorder=1)
    ax.set_xlabel("Text Sentiment Score  [-1 = very negative … +1 = very positive]")
    ax.set_ylabel("Audio Valence Score  [-1 … +1]")
    ax.set_title(
        f"Text vs Audio Sentiment  —  {stem}\n"
        f"Pearson r={stats['pearson_r']:.3f} (p={stats['pearson_p']:.3f})  "
        f"Spearman ρ={stats['spearman_r']:.3f}  "
        f"Agreement={stats['agreement']:.1%}  κ={stats['cohen_kappa']:.3f}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, zorder=0)

    path = os.path.join(out_dir, f"{stem}_sentiment_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_confusion(stats: dict, stem: str, out_dir: str) -> None:
    """Confusion matrix: rows = text polarity, cols = audio polarity."""
    cm = np.array(stats["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(POLARITY_LABELS, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(POLARITY_LABELS, fontsize=10)
    ax.set_xlabel("Audio Polarity")
    ax.set_ylabel("Text Polarity")
    ax.set_title(
        f"Sentiment Confusion Matrix  —  {stem}\n"
        f"Agreement={stats['agreement']:.1%}   κ={stats['cohen_kappa']:.3f}"
    )
    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > 0.6 * cm_max else "black")

    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_sentiment_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


def plot_distribution(stats: dict, stem: str, out_dir: str) -> None:
    """Grouped bar chart comparing text vs audio polarity distributions."""
    n   = stats["n_segments"]
    x   = np.arange(len(POLARITY_LABELS))
    w   = 0.35
    t_c = [stats["text_polarity_dist"][p]  for p in POLARITY_LABELS]
    a_c = [stats["audio_polarity_dist"][p] for p in POLARITY_LABELS]

    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w / 2, [c / n for c in t_c], w, label="Text (BERT)",
                color="#2563EB", alpha=0.85)
    b2 = ax.bar(x + w / 2, [c / n for c in a_c], w, label="Audio (wav2vec2)",
                color="#7C3AED", alpha=0.85)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(POLARITY_LABELS, fontsize=10)
    ax.set_ylabel("Proportion of segments")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Polarity Distribution: Text vs Audio  —  {stem}")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{stem}_sentiment_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Console report + JSON save
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    stats: dict,
    text_results: list[dict],
    audio_results: list[dict],
    valence_low: float,
    valence_high: float,
) -> None:
    n = stats["n_segments"]
    print("\n" + "═" * 65)
    print("  SENTIMENT ALIGNMENT REPORT")
    print("═" * 65)
    print(f"  Segments evaluated    : {n}")
    print(f"  Agreement (3-class)   : {stats['agreement']:.4f}  ({stats['agreement']:.1%})")
    print(f"  Cohen's kappa         : {stats['cohen_kappa']:.4f}")
    print(f"  Pearson r             : {stats['pearson_r']:.4f}  (p={stats['pearson_p']:.4f})")
    print(f"  Spearman ρ            : {stats['spearman_r']:.4f}  (p={stats['spearman_p']:.4f})")
    print()
    print(f"  {'Polarity':<12}  {'Text':>6}  {'Audio':>6}")
    print("  " + "-" * 30)
    for pol in POLARITY_LABELS:
        tc = stats["text_polarity_dist"][pol]
        ac = stats["audio_polarity_dist"][pol]
        print(f"  {pol:<12}  {tc:>4} ({tc/n:.0%})  {ac:>4} ({ac/n:.0%})")
    print()
    print("  Sample (first 8 segments):")
    print(f"  {'#':>3}  {'Text-score':>10}  {'Valence':>7}  {'T-Pol':<10}  {'A-Pol':<10}  Match")
    print("  " + "-" * 58)
    for i, (tr, ar) in enumerate(zip(text_results[:8], audio_results[:8])):
        ap    = valence_to_polarity(ar["valence"], valence_low, valence_high)
        match = "✓" if tr["polarity"] == ap else "✗"
        print(
            f"  {i:>3}  {tr['numeric_score']:>10.3f}  {ar['valence']:>7.3f}"
            f"  {tr['polarity']:<10}  {ap:<10}  {match}"
        )
    print("═" * 65)


def save_results(
    text_results: list[dict],
    audio_results: list[dict],
    stats: dict,
    segments: list[dict],
    valence_low: float,
    valence_high: float,
    stem: str,
    out_dir: str,
) -> None:
    per_segment = []
    for i, (seg, tr, ar) in enumerate(zip(segments, text_results, audio_results)):
        audio_pol = valence_to_polarity(ar["valence"], valence_low, valence_high)
        per_segment.append({
            "idx":        i,
            "segment_id": seg.get("segment_id", str(i)),
            "text":       seg["text"],
            "text_sentiment": {
                "label":         tr["label"],
                "stars":         tr["stars"],
                "confidence":    tr["confidence"],
                "numeric_score": tr["numeric_score"],
                "polarity":      tr["polarity"],
            },
            "audio_sentiment": {
                "valence":       ar["valence"],
                "arousal":       ar["arousal"],
                "dominance":     ar["dominance"],
                "numeric_score": valence_to_numeric(ar["valence"]),
                "polarity":      audio_pol,
            },
            "sentiment_match": tr["polarity"] == audio_pol,
        })

    # Strip internal keys not meant for JSON output
    stats_out = {k: v for k, v in stats.items() if not k.startswith("_")}

    out = {
        "stem":        stem,
        "text_model":  TEXT_MODEL_ID,
        "audio_model": AUDIO_MODEL_ID,
        "valence_thresholds": {"low": valence_low, "high": valence_high},
        "aggregate":   stats_out,
        "segments":    per_segment,
    }
    path = os.path.join(out_dir, f"{stem}_sentiment_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  [saved] {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sentiment alignment evaluation for audio-text pairs"
    )
    # Text source (exactly one required)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json",  help="Captions JSON with 'text' field per entry")
    src.add_argument("--text",  help="Plain text file — split by sentence punctuation")

    parser.add_argument("--audio",          required=True,
                        help="Compiled audio file (WAV or MP3)")
    parser.add_argument("--out_dir",        default="./sentiment_eval_results")
    parser.add_argument("--min_silence_ms", type=int,   default=300,
                        help="Minimum silence gap for audio splitting (ms)")
    parser.add_argument("--silence_thresh", type=float, default=40,
                        help="top_db threshold for librosa silence split")
    parser.add_argument("--valence_low",    type=float, default=0.4,
                        help="Valence below this threshold → negative")
    parser.add_argument("--valence_high",   type=float, default=0.6,
                        help="Valence above this threshold → positive")
    parser.add_argument("--save_segments",  action="store_true",
                        help="Dump split WAV segments to <out_dir>/segments/")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    stem      = Path(args.audio).stem

    print(f"\n{'═'*65}")
    print(f"  Sentiment Eval  :  {stem}")
    print(f"  Audio           :  {args.audio}")
    print(f"  Device          :  {device}")
    print(f"{'═'*65}")

    # ── Load text segments ─────────────────────────────────────────────────
    if args.json:
        print(f"\n  Loading segments from JSON : {args.json}")
        segments = load_segments_from_json(args.json)
    else:
        print(f"\n  Loading text from          : {args.text}")
        raw_text = Path(args.text).read_text(encoding="utf-8")
        sents    = split_text_to_segments(raw_text)
        segments = [{"segment_id": str(i), "text": s} for i, s in enumerate(sents)]
    print(f"  Text segments  : {len(segments)}")

    # ── Load + split audio ─────────────────────────────────────────────────
    print(f"\n  Loading audio  : {args.audio}")
    wav, sr, audio_segs = load_and_split_audio(
        args.audio,
        top_db=args.silence_thresh,
        min_silence_ms=args.min_silence_ms,
    )
    print(f"  Audio segments : {len(audio_segs)}")

    if args.save_segments:
        seg_dir = os.path.join(args.out_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for i, (s, e, w) in enumerate(audio_segs):
            sf.write(os.path.join(seg_dir, f"seg_{i:04d}.wav"), w, sr)
        print(f"  [saved] {len(audio_segs)} WAVs → {seg_dir}/")

    # ── Align (ordered 1-to-1) ─────────────────────────────────────────────
    n = min(len(segments), len(audio_segs))
    if len(segments) != len(audio_segs):
        warnings.warn(
            f"Text segments ({len(segments)}) ≠ audio segments ({len(audio_segs)}). "
            f"Truncating to {n} for matched-pair analysis. "
            f"Adjust --min_silence_ms or --silence_thresh for a better split."
        )
    segments        = segments[:n]
    audio_segs_used = audio_segs[:n]

    # ── Load models ────────────────────────────────────────────────────────
    print()
    text_pipe = load_text_sentiment_pipeline(device_id)
    audio_processor, audio_model = load_audio_sentiment_model(device)

    # ── Text sentiment ─────────────────────────────────────────────────────
    print(f"\n  Running text sentiment on {n} segments…")
    text_results: list[dict] = []
    for i, seg in enumerate(segments):
        try:
            text_results.append(predict_text_sentiment(seg["text"], text_pipe))
        except Exception as e:
            warnings.warn(f"  Segment {i} text failed: {e}")
            text_results.append({
                "label": "3 stars", "stars": 3, "confidence": 0.0,
                "numeric_score": 0.0, "polarity": "neutral",
            })
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n}")
    print(f"  Done ({n} segments).")

    # ── Audio sentiment ────────────────────────────────────────────────────
    print(f"\n  Running audio sentiment on {n} segments…")
    audio_results: list[dict] = []
    for i, (s, e, w) in enumerate(audio_segs_used):
        try:
            audio_results.append(
                predict_audio_emotion(w, sr, audio_processor, audio_model, device)
            )
        except Exception as e:
            warnings.warn(f"  Segment {i} audio failed: {e}")
            audio_results.append({"valence": 0.5, "arousal": 0.5, "dominance": 0.5})
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n}")
    print(f"  Done ({n} segments).")

    # ── Matched-pair analysis ──────────────────────────────────────────────
    print("\n  Running matched-pair analysis…")
    stats = matched_pair_analysis(
        text_results, audio_results,
        valence_low=args.valence_low,
        valence_high=args.valence_high,
    )

    print_report(stats, text_results, audio_results, args.valence_low, args.valence_high)

    # ── Visualisation ──────────────────────────────────────────────────────
    print("\n  Generating plots…")
    plot_timeline(text_results, audio_results, audio_segs_used,
                  args.valence_low, args.valence_high, stem, args.out_dir)
    plot_scatter(text_results, audio_results, stats, stem, args.out_dir)
    plot_confusion(stats, stem, args.out_dir)
    plot_distribution(stats, stem, args.out_dir)

    # ── Save JSON ──────────────────────────────────────────────────────────
    save_results(text_results, audio_results, stats, segments,
                 args.valence_low, args.valence_high, stem, args.out_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
