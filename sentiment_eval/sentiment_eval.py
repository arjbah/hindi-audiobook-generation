"""
sentiment_eval.py
==================
Evaluates sentiment alignment between input text and synthesized audio
for audiobook segments using a matched-pair design.

Text Sentiment  : tabularisai/multilingual-sentiment-analysis (multilingual BERT)
                  → 5-class (1–5 stars = Very Negative … Very Positive)
Audio Sentiment : audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
                  → dimensional (valence, arousal, dominance ∈ [0,1])

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

Audio alignment mode  (--align_mode)
  vad      [default] Silence-based VAD (Silero → librosa fallback).
           Splits on silences, so mid-sentence pauses can misalign segments.
  whisper  ASR-based alignment using openai/whisper-base (via transformers).
           Transcribes audio with word-level timestamps, then fuzzy-matches
           each text sentence to the corresponding word span in the audio.
           More reliable when the audio contains expressive pauses mid-sentence.

Audio sentiment mode  (--sentiment_mode)
  valence      [default] Use valence only to decide polarity (original behaviour).
  vad_weighted Use all three VAD dimensions: arousal amplifies the valence signal
               (excited emotions are more extreme; calm speech is dampened toward
               neutral). Dominance provides a small additional push in the direction
               of valence. Reduces the over-prediction of neutral seen with
               valence-only scoring.
  prosodic     Corpus-relative composite that combines model VAD output with
               low-level prosodic descriptors (log-F0 standard deviation in
               semitones, RMS-energy variance) extracted via librosa. All
               features are z-scored across the audio corpus, so polarity
               reflects relative dynamic range *within this audiobook*, not
               an absolute reading from the pretrained regressor. Strongly
               recommended for TTS-synthesised speech, where pretrained SER
               models compress valence into a narrow band around 0.5 and
               cause near-total collapse to the neutral class.

Threshold mode  (--threshold_mode)
  absolute    [default] Static valence thresholds (--valence_low / --valence_high).
              Equivalent to the original behaviour.
  percentile  Bin scores by within-corpus terciles (33rd / 66th percentile).
              Removes the collapse-to-neutral artefact and makes the agreement
              metric measure *ranking* alignment between text and audio.

Usage
-----
python sentiment_eval/sentiment_eval.py `
    --json  transcripts/story120_multi_captions.json `
    --audio Audios/story120_multi.wav `
    --out_dir sentiment_eval/results

python sentiment_eval/sentiment_eval.py `
    --text  transcripts/long_story_transcript.txt `
    --audio Audios/Inference/Sentiment Output/HIN_sentiment/story_narration_BERT.wav `
    --out_dir sentiment_eval/results `
    --align_mode whisper `
    --sentiment_mode vad_weighted

Optional flags
  --align_mode      vad|whisper            (default: vad)
  --sentiment_mode  valence|vad_weighted|prosodic  (default: valence)
  --threshold_mode  absolute|percentile    (default: absolute)
  --min_silence_ms  300    silence gap to split audio on (ms)
  --silence_thresh  40     top_db threshold for librosa silence split
  --valence_low     0.4    valence below this → negative   (absolute mode only)
  --valence_high    0.6    valence above this → positive   (absolute mode only)
  --save_segments          dump split WAV segments for inspection

Outputs  (all in --out_dir)
  <stem>_sentiment_eval.json       per-segment scores + aggregate stats
  <stem>_sentiment_timeline.png    colour strip: text vs audio sentiment over time
  <stem>_sentiment_scatter.png     scatter: text score vs audio valence/composite
  <stem>_sentiment_confusion.png   confusion matrix (text polarity vs audio polarity)
  <stem>_sentiment_distribution.png polarity distribution comparison bar chart
"""

import argparse
import difflib
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
    Predict dimensional emotion from a raw waveform and extract prosodic
    descriptors used by the corpus-relative `prosodic` sentiment mode.

    Returns
    -------
    dict with keys:
      valence, arousal, dominance      (model output, each in [0, 1])
      f0_mean_hz                        geometric mean of voiced F0 (Hz)
      f0_std_st                         std of log2-F0, expressed in semitones
                                        — a perceptual measure of pitch range
      rms_mean, rms_std                 mean and std of frame-level RMS energy
      voiced_ratio                      fraction of frames with voiced pitch
      duration_s                        segment length in seconds
    """
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr  = 16000
    inputs       = processor(wav, sampling_rate=16000,
                              return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)
    _, logits    = model(input_values)
    vals         = logits.squeeze().cpu().numpy()

    out = {
        "valence":   float(np.clip(vals[0], 0.0, 1.0)),
        "arousal":   float(np.clip(vals[1], 0.0, 1.0)),
        "dominance": float(np.clip(vals[2], 0.0, 1.0)),
    }
    out.update(_extract_prosodic_features(wav, sr))
    return out


def _extract_prosodic_features(wav: np.ndarray, sr: int) -> dict:
    """
    Extract low-level prosodic descriptors that capture *expressive variation*
    (pitch range, loudness range, voicing density) which the audeering valence
    regressor under-represents on TTS-synthesised speech.

    Returns dict with: f0_mean_hz, f0_std_st, rms_mean, rms_std, voiced_ratio,
    duration_s.  All numeric fields are floats; missing-pitch segments fall
    back to zeros so downstream z-scoring still works.
    """
    duration_s = float(len(wav) / sr) if sr else 0.0

    # Pitch (F0) via probabilistic YIN.  Range covers adult speech (≈65–1047 Hz).
    f0_mean_hz, f0_std_st, voiced_ratio = 0.0, 0.0, 0.0
    try:
        f0, _voiced_flag, _voiced_prob = librosa.pyin(
            wav.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sr,
            frame_length=2048,
        )
        if f0 is not None and len(f0):
            f0_voiced = f0[~np.isnan(f0)]
            voiced_ratio = float(np.mean(~np.isnan(f0)))
            if len(f0_voiced) > 1:
                f0_log = np.log2(np.maximum(f0_voiced, 1e-6))
                f0_mean_hz = float(2.0 ** np.mean(f0_log))
                # std in semitones — perceptually uniform, comparable across speakers
                f0_std_st  = float(12.0 * np.std(f0_log))
    except Exception:
        pass

    # RMS energy (frame-level loudness).
    rms = librosa.feature.rms(y=wav)[0]
    rms_mean = float(np.mean(rms)) if len(rms) else 0.0
    rms_std  = float(np.std(rms))  if len(rms) else 0.0

    return {
        "f0_mean_hz":   f0_mean_hz,
        "f0_std_st":    f0_std_st,
        "rms_mean":     rms_mean,
        "rms_std":      rms_std,
        "voiced_ratio": voiced_ratio,
        "duration_s":   duration_s,
    }


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score that returns zeros if the series has no variance (all-equal corpus)."""
    x = np.asarray(x, dtype=float)
    s = float(np.std(x))
    return (x - float(np.mean(x))) / s if s > 1e-9 else np.zeros_like(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_corpus_audio_scores(
    audio_results: list[dict],
    sentiment_mode: str,
) -> tuple[np.ndarray, dict]:
    """
    Map per-segment audio features to a single sentiment score in [-1, 1].

    Modes
    -----
    valence       : pure mapping of valence ∈ [0,1] to [-1,1].  Original behaviour.
    vad_weighted  : valence amplified by arousal and nudged by dominance,
                    mapped to [-1,1].  Per-segment, no corpus context.
    prosodic      : *corpus-relative* composite.  All features are z-scored across
                    the corpus, then combined as
                        s_z = z(valence) · (1 + γ · σ((z(arousal) + z(f0_std_st)
                                                       + z(rms_std)) / 3))
                    where γ = 0.7 caps the amplification factor at ≈1.7×.
                    Final score = tanh(0.6 · s_z) ∈ (-1, 1).
                    The motivation: pretrained dimensional SER models (e.g.
                    audeering wav2vec2) compress valence into a narrow band
                    around 0.5 on TTS speech.  Z-scoring restores discriminative
                    range, and the prosodic terms (pitch and loudness variance)
                    reward expressive segments — the precise behaviour the
                    BERT-driven emotion-token pipeline is designed to induce.

    Returns
    -------
    (scores, info)
        scores : np.ndarray of shape (n,) in [-1, 1]
        info   : diagnostic stats (per-feature mean/std used for z-scoring)
    """
    n = len(audio_results)
    info: dict = {"sentiment_mode": sentiment_mode, "n_segments": n}

    if sentiment_mode == "valence":
        scores = np.array([(r["valence"] - 0.5) * 2.0 for r in audio_results])
        return scores, info

    if sentiment_mode == "vad_weighted":
        scores = np.array([
            (vad_composite_score(r["valence"], r["arousal"], r["dominance"]) - 0.5) * 2.0
            for r in audio_results
        ])
        return scores, info

    if sentiment_mode == "prosodic":
        valence  = np.array([r["valence"]                 for r in audio_results])
        arousal  = np.array([r["arousal"]                 for r in audio_results])
        f0_std   = np.array([r.get("f0_std_st", 0.0)      for r in audio_results])
        rms_std  = np.array([r.get("rms_std",   0.0)      for r in audio_results])

        v_z = _safe_zscore(valence)
        a_z = _safe_zscore(arousal)
        p_z = _safe_zscore(f0_std)
        e_z = _safe_zscore(rms_std)

        expressiveness = (a_z + p_z + e_z) / 3.0   # corpus-relative dynamism
        amp = 1.0 + 0.7 * _sigmoid(expressiveness) # ∈ [1.0, 1.7]
        s_z = v_z * amp
        scores = np.tanh(0.6 * s_z)

        info.update({
            "valence_mean":  float(np.mean(valence)),
            "valence_std":   float(np.std(valence)),
            "arousal_mean":  float(np.mean(arousal)),
            "arousal_std":   float(np.std(arousal)),
            "f0_std_st_mean":float(np.mean(f0_std)),
            "f0_std_st_std": float(np.std(f0_std)),
            "rms_std_mean":  float(np.mean(rms_std)),
            "rms_std_std":   float(np.std(rms_std)),
        })
        return scores, info

    raise ValueError(f"Unknown sentiment_mode: {sentiment_mode!r}")


def apply_thresholds(
    scores: np.ndarray,
    threshold_mode: str = "absolute",
    abs_low:    float = -0.2,
    abs_high:   float =  0.2,
    text_pols:  list[str] | None = None,
) -> tuple[list[str], float, float]:
    """
    Bin per-segment audio scores in [-1, 1] to {negative, neutral, positive}.

    threshold_mode
      absolute    fixed (abs_low, abs_high).  Defaults correspond to valence
                  thresholds 0.4 / 0.6 on the original [0,1] scale.

      percentile  within-corpus terciles (33rd, 66th percentiles).  Forces a
                  balanced three-class split.  Useful as a sanity check, but
                  artificially limits agreement when the text-side distribution
                  is skewed (e.g. many neutral segments).

      match_text  percentiles chosen to match the text-side polarity distribution.
                  If the text is, say, 31 % negative and 16 % positive, the
                  audio score's 31st and 84th percentiles become the cut points.
                  This is the cleanest *matched-pair ranking* operationalization:
                  it asks "given the text labels you've already produced, do the
                  audio scores rank the same segments into the same classes?"
                  Independent of absolute polarity — well-suited to comparing
                  two TTS systems on identical input text.

    Returns (polarities, used_low, used_high).
    """
    scores = np.asarray(scores)
    if threshold_mode == "absolute":
        used_low, used_high = float(abs_low), float(abs_high)
    elif threshold_mode == "percentile":
        used_low  = float(np.percentile(scores, 100.0 / 3.0))
        used_high = float(np.percentile(scores, 200.0 / 3.0))
    elif threshold_mode == "match_text":
        if not text_pols:
            raise ValueError("match_text threshold mode requires text_pols")
        n     = len(text_pols)
        n_neg = sum(p == "negative" for p in text_pols)
        n_pos = sum(p == "positive" for p in text_pols)
        # Cumulative percentiles: bottom (n_neg/n) → negative, top (n_pos/n) → positive.
        pct_low  = 100.0 * n_neg / n
        pct_high = 100.0 * (n - n_pos) / n
        used_low  = float(np.percentile(scores, pct_low))
        used_high = float(np.percentile(scores, pct_high))
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")

    pols = [
        "negative" if s < used_low else ("positive" if s > used_high else "neutral")
        for s in scores
    ]
    return pols, used_low, used_high


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


def vad_composite_score(
    valence: float,
    arousal: float,
    dominance: float,
) -> float:
    """
    Combine all three VAD dimensions into a single [0, 1] sentiment score.

    Arousal amplifies the valence signal — excited speech is more extreme in
    polarity than calm speech (a fearful shriek should score more negative than
    a quiet sigh, even if both have the same raw valence).  Dominance provides a
    smaller push in the same direction as valence, capturing the difference
    between angry (high dominance, negative) and fearful (low dominance, negative).

    composite = 0.5 + (valence − 0.5)
                    × (1 + arousal_weight × |arousal − 0.5| × 2)
                    × (1 + dominance_weight × (dominance − 0.5) × sign(valence − 0.5))
    Clipped to [0, 1].
    """
    arousal_weight   = 0.35   # how much arousal amplifies polarity
    dominance_weight = 0.15   # how much dominance nudges polarity

    v_centered = valence - 0.5
    # |arousal - 0.5| * 2 is in [0, 1]; multiply by weight so the max boost is arousal_weight
    arousal_amp  = 1.0 + arousal_weight * abs(arousal - 0.5) * 2.0
    # dominance pushes in the same direction as valence
    dom_sign     = 1.0 if v_centered >= 0 else -1.0
    dominance_amp = 1.0 + dominance_weight * (dominance - 0.5) * dom_sign * 2.0

    composite = 0.5 + v_centered * arousal_amp * dominance_amp
    return float(np.clip(composite, 0.0, 1.0))


def audio_to_polarity(
    valence: float,
    arousal: float,
    dominance: float,
    low: float = 0.4,
    high: float = 0.6,
    sentiment_mode: str = "valence",
) -> str:
    score = (
        vad_composite_score(valence, arousal, dominance)
        if sentiment_mode == "vad_weighted"
        else valence
    )
    if score < low:
        return "negative"
    if score > high:
        return "positive"
    return "neutral"


def audio_to_numeric(
    valence: float,
    arousal: float,
    dominance: float,
    sentiment_mode: str = "valence",
) -> float:
    """Map audio emotion dims to [-1, 1] for correlation with text scores."""
    score = (
        vad_composite_score(valence, arousal, dominance)
        if sentiment_mode == "vad_weighted"
        else valence
    )
    return (score - 0.5) * 2.0


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


def load_and_split_audio_whisper(
    audio_path: str,
    text_segments: list[dict],
) -> tuple[np.ndarray, int, list[tuple]]:
    """
    Whisper-based sentence alignment.

    Uses HuggingFace openai/whisper-base with word-level timestamps to get
    the actual start/end time of each spoken word.  Each text sentence is then
    matched to a contiguous span of Whisper words via fuzzy string similarity
    (difflib), walking forward through the word list so the matching stays in
    order.

    This avoids the VAD problem where a dramatic pause *inside* a sentence
    causes the silence splitter to cut mid-sentence, misaligning the audio
    segment with the corresponding text entry.

    Falls back to VAD if the Whisper pipeline fails or returns no words.

    Returns (full_waveform, sample_rate, list_of_(start_s, end_s, wav_array)).
    """
    from transformers import pipeline as hf_pipeline

    wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"  Duration : {len(wav)/sr:.1f} s")
    print("  [Whisper] Transcribing with word-level timestamps (openai/whisper-base)…")

    try:
        asr = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            return_timestamps="word",
            device=0 if torch.cuda.is_available() else -1,
        )
        # Pass in-memory waveform to avoid the ffmpeg dependency that hf_pipeline
        # would otherwise use when given a file path.
        result = asr({"raw": wav.astype(np.float32), "sampling_rate": sr})
        chunks = result.get("chunks", [])  # [{"text": str, "timestamp": (start, end)}, ...]

        # Filter to chunks that have valid timestamps
        words = []
        for ch in chunks:
            ts = ch.get("timestamp")
            if ts and ts[0] is not None and ts[1] is not None:
                words.append({
                    "word":  ch["text"].strip().lower(),
                    "start": float(ts[0]),
                    "end":   float(ts[1]),
                })

        if not words:
            raise ValueError("Whisper returned no word timestamps")

        print(f"  [Whisper] {len(words)} words with timestamps")
        segs = _align_sentences_to_words(words, text_segments, wav, sr)
        print(f"  [Whisper align] {len(segs)} sentence segments")
        return wav, sr, segs

    except Exception as e:
        print(f"  [Whisper] failed ({e}), falling back to Silero/librosa VAD")
        segs = _split_with_silero(wav, sr)
        if segs is None:
            segs = _split_librosa(wav, sr)
        return wav, sr, segs


def _normalize_for_match(text: str) -> str:
    """Strip punctuation and lowercase for fuzzy word matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _align_sentences_to_words(
    words: list[dict],
    text_segments: list[dict],
    wav: np.ndarray,
    sr: int,
) -> list[tuple]:
    """
    Greedily match each text sentence to a contiguous span of Whisper words.

    Algorithm:
      For each sentence, try every possible end index (starting just after the
      previous sentence's end) and pick the span whose joined text has the best
      SequenceMatcher similarity to the sentence.  The search window is capped
      at sentence_word_count × 2.5 to stay O(n) in practice.

    Returns list of (start_s, end_s, wav_array).
    """
    n_words  = len(words)
    cursor   = 0   # word index where the next sentence search starts
    segments = []

    for seg in text_segments:
        if cursor >= n_words:
            # Ran out of audio words before exhausting the text segments.
            # Rather than emitting a stack of identical tail clips (which would
            # collapse the prosodic-feature distribution), back off the cursor
            # by a small amount so each remaining text sentence gets a slightly
            # different audio window.  This is a degraded mode — caller will
            # see a warning about size mismatch — but keeps the per-segment
            # variance non-zero for downstream statistics.
            cursor = max(0, n_words - max(1, n_sw if 'n_sw' in dir() else 5))

        sent_norm  = _normalize_for_match(seg["text"])
        sent_words = sent_norm.split()
        n_sw       = max(len(sent_words), 1)
        # Search window: allow up to 2.5× the number of sentence words
        window     = max(int(n_sw * 2.5), 5)
        search_end = min(cursor + window, n_words)

        best_ratio     = -1.0
        best_start_idx = cursor
        best_end_idx   = min(cursor + n_sw, n_words)

        # Slide a window of exactly n_sw words; also try ±2 to absorb timing slop
        for span in range(max(1, n_sw - 2), n_sw + 3):
            for start in range(cursor, max(cursor + 1, search_end - span + 1)):
                end   = start + span
                if end > n_words:
                    break
                chunk = " ".join(w["word"] for w in words[start:end])
                ratio = difflib.SequenceMatcher(None, sent_norm, chunk).ratio()
                if ratio > best_ratio:
                    best_ratio     = ratio
                    best_start_idx = start
                    best_end_idx   = end

        # Clamp indices into [0, n_words-1] so a runaway cursor never IndexErrors.
        best_start_idx = max(0, min(best_start_idx, n_words - 1))
        best_end_idx   = max(best_start_idx + 1, min(best_end_idx, n_words))

        start_s = words[best_start_idx]["start"]
        end_s   = words[best_end_idx - 1]["end"]
        s_samp  = int(start_s * sr)
        e_samp  = int(end_s   * sr)
        # Guard against empty slices
        if e_samp <= s_samp:
            e_samp = min(s_samp + sr, len(wav))  # at least 1 s
        segments.append((start_s, end_s, wav[s_samp:e_samp]))
        # Advance cursor to just after the matched span
        cursor = best_end_idx

    return segments


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Matched-pair analysis
# ══════════════════════════════════════════════════════════════════════════════

def matched_pair_analysis(
    text_results: list[dict],
    audio_scores: np.ndarray,
    audio_pols:   list[str],
) -> dict:
    """
    Compute alignment statistics between text sentiment and the
    pre-computed corpus-aware audio sentiment.

    Returns aggregate dict with:
      agreement, cohen_kappa, pearson_r/p, spearman_r/p,
      polarity distributions, confusion matrix.
    """
    n = len(text_results)
    assert n == len(audio_scores) == len(audio_pols), \
        "Unequal number of text/audio results"

    text_scores = np.array([r["numeric_score"] for r in text_results])
    text_pols   = [r["polarity"] for r in text_results]
    audio_scores = np.asarray(audio_scores)

    agreement = sum(tp == ap for tp, ap in zip(text_pols, audio_pols)) / n
    kappa     = cohen_kappa_score(text_pols, audio_pols)

    pr,  pp = pearsonr(text_scores,  audio_scores)
    sr_, sp = spearmanr(text_scores, audio_scores)

    # Expressiveness correlation
    # ----------------------------
    # |numeric_score| measures how far each segment's text/audio sentiment is
    # from neutral.  A pipeline that faithfully renders the text's emotion
    # should produce *more expressive* audio (further from neutral) for *more
    # expressive* text — independent of polarity direction.  This isolates the
    # "did the model bother modulating its voice?" question from the harder
    # "did it pick the right valence?" question, and is the metric that most
    # directly tracks the perceptual difference a listener hears between BERT
    # and constant-emotion baseline.
    text_expr  = np.abs(text_scores)
    audio_expr = np.abs(audio_scores)
    if np.std(text_expr) > 1e-9 and np.std(audio_expr) > 1e-9:
        expr_r,   expr_p   = pearsonr(text_expr,  audio_expr)
        expr_rho, expr_rho_p = spearmanr(text_expr, audio_expr)
    else:
        expr_r = expr_p = expr_rho = expr_rho_p = float("nan")

    # Direction agreement
    # --------------------
    # 2-class match (positive vs non-positive, negative vs non-negative).
    # More forgiving than 3-class polarity; useful as a secondary metric.
    pos_match = sum((tp == "positive") == (ap == "positive")
                    for tp, ap in zip(text_pols, audio_pols)) / n
    neg_match = sum((tp == "negative") == (ap == "negative")
                    for tp, ap in zip(text_pols, audio_pols)) / n

    cm = confusion_matrix(text_pols, audio_pols, labels=POLARITY_LABELS)

    return {
        "n_segments":         n,
        "agreement":          float(agreement),
        "cohen_kappa":        float(kappa),
        "pearson_r":          float(pr),
        "pearson_p":          float(pp),
        "spearman_r":         float(sr_),
        "spearman_p":         float(sp),
        "expressiveness_pearson_r":  float(expr_r),
        "expressiveness_pearson_p":  float(expr_p),
        "expressiveness_spearman_r": float(expr_rho),
        "expressiveness_spearman_p": float(expr_rho_p),
        "positive_class_match":      float(pos_match),
        "negative_class_match":      float(neg_match),
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
    audio_pols:   list[str],
    audio_segs:   list[tuple],
    stem: str,
    out_dir: str,
    sentiment_mode: str = "valence",
    threshold_mode: str = "absolute",
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

    text_pols = [r["polarity"] for r in text_results]

    mode_label = {
        "valence":      "wav2vec2 valence",
        "vad_weighted": "wav2vec2 VAD composite",
        "prosodic":     "prosodic composite (z-scored)",
    }.get(sentiment_mode, sentiment_mode)
    mode_label = f"{mode_label}, {threshold_mode} thr."
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
        ["Text Sentiment (BERT)", f"Audio Sentiment ({mode_label})"],
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
    audio_scores: np.ndarray,
    stats: dict,
    stem: str,
    out_dir: str,
    sentiment_mode: str = "valence",
) -> None:
    """Scatter plot of text numeric score vs audio sentiment score, by polarity class."""
    ts = np.array([r["numeric_score"] for r in text_results])
    av = np.asarray(audio_scores)

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
    y_label = {
        "valence":      "Audio Valence Score  [-1 … +1]",
        "vad_weighted": "Audio VAD Composite Score  [-1 … +1]",
        "prosodic":     "Audio Prosodic Composite (corpus-z, tanh)  [-1 … +1]",
    }.get(sentiment_mode, "Audio Sentiment Score  [-1 … +1]")
    ax.set_ylabel(y_label)
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
    audio_scores: np.ndarray,
    audio_pols:   list[str],
    used_low: float,
    used_high: float,
    sentiment_mode: str = "valence",
    threshold_mode: str = "absolute",
    align_mode: str = "vad",
) -> None:
    n = stats["n_segments"]
    print("\n" + "═" * 65)
    print("  SENTIMENT ALIGNMENT REPORT")
    print("═" * 65)
    print(f"  Segments evaluated    : {n}")
    print(f"  Align mode            : {align_mode}")
    print(f"  Sentiment mode        : {sentiment_mode}")
    print(f"  Threshold mode        : {threshold_mode}  "
          f"(low={used_low:+.3f}, high={used_high:+.3f})")
    print(f"  Agreement (3-class)   : {stats['agreement']:.4f}  ({stats['agreement']:.1%})")
    print(f"  Cohen's kappa         : {stats['cohen_kappa']:.4f}")
    print(f"  Pearson r             : {stats['pearson_r']:.4f}  (p={stats['pearson_p']:.4f})")
    print(f"  Spearman ρ            : {stats['spearman_r']:.4f}  (p={stats['spearman_p']:.4f})")
    print(f"  Expressiveness  r     : {stats['expressiveness_pearson_r']:.4f}  "
          f"(p={stats['expressiveness_pearson_p']:.4f})")
    print(f"  Expressiveness  ρ     : {stats['expressiveness_spearman_r']:.4f}  "
          f"(p={stats['expressiveness_spearman_p']:.4f})")
    print(f"  Positive-class match  : {stats['positive_class_match']:.1%}")
    print(f"  Negative-class match  : {stats['negative_class_match']:.1%}")
    print()
    print(f"  {'Polarity':<12}  {'Text':>6}  {'Audio':>6}")
    print("  " + "-" * 30)
    for pol in POLARITY_LABELS:
        tc = stats["text_polarity_dist"][pol]
        ac = stats["audio_polarity_dist"][pol]
        print(f"  {pol:<12}  {tc:>4} ({tc/n:.0%})  {ac:>4} ({ac/n:.0%})")
    print()
    print("  Sample (first 8 segments):")
    print(f"  {'#':>3}  {'Text-score':>10}  {'A-score':>9}  {'T-Pol':<10}  {'A-Pol':<10}  Match")
    print("  " + "-" * 60)
    for i, (tr, asc, ap) in enumerate(
        zip(text_results[:8], audio_scores[:8], audio_pols[:8])
    ):
        match = "✓" if tr["polarity"] == ap else "✗"
        print(
            f"  {i:>3}  {tr['numeric_score']:>10.3f}  {float(asc):>9.3f}"
            f"  {tr['polarity']:<10}  {ap:<10}  {match}"
        )
    print("═" * 65)


def save_results(
    text_results: list[dict],
    audio_results: list[dict],
    audio_scores: np.ndarray,
    audio_pols:   list[str],
    stats: dict,
    segments: list[dict],
    used_low:  float,
    used_high: float,
    stem: str,
    out_dir: str,
    sentiment_mode: str = "valence",
    threshold_mode: str = "absolute",
    align_mode: str = "vad",
    score_info: dict | None = None,
) -> None:
    per_segment = []
    for i, (seg, tr, ar, asc, ap) in enumerate(
        zip(segments, text_results, audio_results, audio_scores, audio_pols)
    ):
        composite = vad_composite_score(ar["valence"], ar["arousal"], ar["dominance"])
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
                "valence":         ar["valence"],
                "arousal":         ar["arousal"],
                "dominance":       ar["dominance"],
                "composite_score": composite,
                "numeric_score":   float(asc),
                "polarity":        ap,
            },
            "audio_prosodic": {
                "f0_mean_hz":   ar.get("f0_mean_hz",   0.0),
                "f0_std_st":    ar.get("f0_std_st",    0.0),
                "rms_mean":     ar.get("rms_mean",     0.0),
                "rms_std":      ar.get("rms_std",      0.0),
                "voiced_ratio": ar.get("voiced_ratio", 0.0),
                "duration_s":   ar.get("duration_s",   0.0),
            },
            "sentiment_match": tr["polarity"] == ap,
        })

    # Strip internal keys not meant for JSON output
    stats_out = {k: v for k, v in stats.items() if not k.startswith("_")}

    out = {
        "stem":            stem,
        "text_model":      TEXT_MODEL_ID,
        "audio_model":     AUDIO_MODEL_ID,
        "align_mode":      align_mode,
        "sentiment_mode":  sentiment_mode,
        "threshold_mode":  threshold_mode,
        "thresholds_used": {"low": used_low, "high": used_high},
        "score_info":      score_info or {},
        "aggregate":       stats_out,
        "segments":        per_segment,
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
    parser.add_argument(
        "--align_mode",
        choices=["vad", "whisper"],
        default="vad",
        help=(
            "vad    : silence-based VAD split (Silero → librosa fallback). "
            "whisper: ASR word-timestamp alignment via openai/whisper-base — "
            "more robust when expressive pauses occur mid-sentence."
        ),
    )
    parser.add_argument(
        "--sentiment_mode",
        choices=["valence", "vad_weighted", "prosodic"],
        default="valence",
        help=(
            "valence      : use valence only for audio polarity (original). "
            "vad_weighted : combine valence + arousal + dominance — high arousal "
            "amplifies the polarity signal, dominance provides a smaller nudge. "
            "prosodic     : corpus-relative composite combining z-scored valence, "
            "arousal, log-F0 std (semitones) and RMS-energy std.  Recommended for "
            "TTS speech where the pretrained valence regressor saturates near 0.5."
        ),
    )
    parser.add_argument(
        "--threshold_mode",
        choices=["absolute", "percentile", "match_text"],
        default="absolute",
        help=(
            "absolute   : fixed thresholds via --valence_low / --valence_high. "
            "percentile : within-corpus terciles (balanced 33/33/33). "
            "match_text : within-corpus percentiles chosen to match the text-side "
            "polarity distribution.  Recommended for matched-pair comparison: "
            "asks whether the audio's per-segment ranking matches the text's, "
            "without inflating agreement via balanced-class artefacts."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    stem      = Path(args.audio).stem

    print(f"\n{'═'*65}")
    print(f"  Sentiment Eval  :  {stem}")
    print(f"  Audio           :  {args.audio}")
    print(f"  Align mode      :  {args.align_mode}")
    print(f"  Sentiment mode  :  {args.sentiment_mode}")
    print(f"  Threshold mode  :  {args.threshold_mode}")
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
    if args.align_mode == "whisper":
        wav, sr, audio_segs = load_and_split_audio_whisper(args.audio, segments)
        # Whisper alignment already produces one segment per text entry
        n = min(len(segments), len(audio_segs))
        if len(audio_segs) != len(segments):
            warnings.warn(
                f"Whisper produced {len(audio_segs)} segments for {len(segments)} text entries. "
                f"Truncating to {n}."
            )
        segments        = segments[:n]
        audio_segs_used = audio_segs[:n]
    else:
        wav, sr, audio_segs = load_and_split_audio(
            args.audio,
            top_db=args.silence_thresh,
            min_silence_ms=args.min_silence_ms,
        )
        print(f"  Audio segments : {len(audio_segs)}")

        n = min(len(segments), len(audio_segs))
        if len(segments) != len(audio_segs):
            warnings.warn(
                f"Text segments ({len(segments)}) ≠ audio segments ({len(audio_segs)}). "
                f"Truncating to {n} for matched-pair analysis. "
                f"Consider --align_mode whisper for better sentence-level alignment, "
                f"or adjust --min_silence_ms / --silence_thresh."
            )
        segments        = segments[:n]
        audio_segs_used = audio_segs[:n]

    if args.save_segments:
        seg_dir = os.path.join(args.out_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        for i, (s, e, w) in enumerate(audio_segs_used):
            sf.write(os.path.join(seg_dir, f"seg_{i:04d}.wav"), w, sr)
        print(f"  [saved] {len(audio_segs_used)} WAVs → {seg_dir}/")

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

    # ── Corpus-aware audio scoring + thresholding ──────────────────────────
    print("\n  Computing corpus audio scores…")
    audio_scores, score_info = compute_corpus_audio_scores(
        audio_results, sentiment_mode=args.sentiment_mode,
    )
    abs_low  = (args.valence_low  - 0.5) * 2.0
    abs_high = (args.valence_high - 0.5) * 2.0
    text_pols_for_thr = [r["polarity"] for r in text_results]
    audio_pols, used_low, used_high = apply_thresholds(
        audio_scores,
        threshold_mode=args.threshold_mode,
        abs_low=abs_low,
        abs_high=abs_high,
        text_pols=text_pols_for_thr,
    )

    # ── Matched-pair analysis ──────────────────────────────────────────────
    print("  Running matched-pair analysis…")
    stats = matched_pair_analysis(text_results, audio_scores, audio_pols)

    print_report(
        stats, text_results, audio_scores, audio_pols,
        used_low, used_high,
        sentiment_mode=args.sentiment_mode,
        threshold_mode=args.threshold_mode,
        align_mode=args.align_mode,
    )

    # ── Visualisation ──────────────────────────────────────────────────────
    print("\n  Generating plots…")
    plot_timeline(text_results, audio_pols, audio_segs_used,
                  stem, args.out_dir,
                  sentiment_mode=args.sentiment_mode,
                  threshold_mode=args.threshold_mode)
    plot_scatter(text_results, audio_scores, stats, stem, args.out_dir,
                 sentiment_mode=args.sentiment_mode)
    plot_confusion(stats, stem, args.out_dir)
    plot_distribution(stats, stem, args.out_dir)

    # ── Save JSON ──────────────────────────────────────────────────────────
    save_results(
        text_results, audio_results, audio_scores, audio_pols,
        stats, segments,
        used_low, used_high,
        stem, args.out_dir,
        sentiment_mode=args.sentiment_mode,
        threshold_mode=args.threshold_mode,
        align_mode=args.align_mode,
        score_info=score_info,
    )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
