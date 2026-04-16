#!/usr/bin/env python3
"""
Sentiment-Guided IndicParlerTTS Narrator Pipeline — English
============================================================
Per-chunk BERT sentiment analysis drives dynamic IndicParler emotion token
and expressivity descriptor selection, aligning TTS delivery to text intent.

New vs. english_indic_narrator_pipeline.py
------------------------------------------
  · F0 drift threshold : 0.12 → 0.15  (relaxed; named speakers are stable)
  · F0 std ratio       : 0.30 → 0.35
  · max_retries fixed at 2  (3 TTS calls worst-case)
  · SentimentClassifier   — tabularisai/multilingual-sentiment-analysis BERT
  · EmotionAssigner       — score + text heuristics → IndicParler emotion token
  · Dynamic caption       — frozen CVC + emotion token + expressivity phrase
  · Log augmented         — sentiment stars, score, emotion, expressivity phrase

IndicParler emotion tokens
--------------------------
  Anger | Command | Conversation | Disgust | Fear | Happy |
  Narration | Neutral | News | Proper Noun | Sad | Surprise

Architecture
------------
  Anchor (one-time)
    synthesize seed clip → ECAPA embed → Praat F0 stats → registry

  Per-chunk loop
    → BERT classifier on chunk text → numeric score + confidence
    → EmotionAssigner: score + keyword heuristics → (emotion, expressivity_phrase)
    → compose caption: frozen CVC + dynamic emotion + expressivity phrase
    → IndicParlerTTS inference
    → Praat F0 validation  [primary gate, ~3 ms CPU]
    → if F0 passed: ECAPA cosine check  [secondary, ~45 ms]
    → if failed: patch CVC F0 descriptor → retry (max 2)
    → flag + emit last audio on exhaustion

Usage
-----
  python sentiment_narrator_eng.py
  python sentiment_narrator_eng.py --story transcripts/bad_blood_eng.txt \\
      --output output/ENG_sentiment/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

# Ensure Unicode output works on Windows consoles (cp1252 → utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Narrator voice ────────────────────────────────────────────────────────
    narrator_cvc: str = (
        "Thoma speaks at a slightly slow pace with a deep, low-pitched voice "
        "and a narrow pitch range in a very close-sounding studio environment. "
        "The audio is of excellent quality with no background noise."
    )
    # Fallback emotion if sentiment is ambiguous (very low confidence)
    default_emotion: str = "Narration"

    # ── Chunking ──────────────────────────────────────────────────────────────
    max_chunk_chars: int = 300       # English chars; 300 ≈ 280 Hindi equivalents
    min_chunk_chars: int = 40

    # ── F0 consistency (primary gate) ─────────────────────────────────────────
    # Relaxed slightly from 0.12/0.30 — named speakers are consistently stable.
    f0_drift_threshold: float = 0.15
    f0_std_ratio_threshold: float = 0.35

    # ── Speaker embedding (secondary gate) ────────────────────────────────────
    speaker_sim_threshold: float = 0.82

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 2             # worst-case = 3 TTS calls per chunk

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Anchor ────────────────────────────────────────────────────────────────
    anchor_text: str = "In a calm and steady voice, this story begins to unfold."


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER  (English — no Devanagari dandas)
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits English text at sentence and paragraph boundaries,
    then greedy-groups into chunks under max_chunk_chars.
    """

    _STANDARD = re.compile(r'(?<=[.!?])\s+(?=[^\d])')
    _PARA      = re.compile(r'\n{2,}')

    def __init__(self, cfg: PipelineConfig):
        self.max_chars = cfg.max_chunk_chars
        self.min_chars = cfg.min_chunk_chars

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        paragraphs = self._PARA.split(text)
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            parts = self._STANDARD.split(para)
            sentences.extend(s.strip() for s in parts if s.strip())

        chunks: list[str] = []
        buf = ""
        for sent in sentences:
            candidate = (buf + " " + sent).strip() if buf else sent
            if buf and len(candidate) > self.max_chars:
                if len(buf) >= self.min_chars:
                    chunks.append(buf)
                buf = sent
            else:
                buf = candidate
        if buf and len(buf) >= self.min_chars:
            chunks.append(buf)
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# F0 EXTRACTOR  — Praat via parselmouth, CPU ~3 ms per clip
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class F0Stats:
    mean: float
    std: float
    voiced_fraction: float

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "voiced_fraction": round(self.voiced_fraction, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "F0Stats":
        return cls(**d)


class F0Extractor:
    F0_FLOOR   = 75.0   # Hz
    F0_CEILING = 300.0  # Hz
    TIME_STEP  = 0.01   # s

    def extract(self, audio: np.ndarray, sr: int) -> Optional[F0Stats]:
        try:
            import parselmouth
        except ImportError:
            raise ImportError("Run: pip install praat-parselmouth")
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))
        pitch = snd.to_pitch(
            time_step=self.TIME_STEP,
            pitch_floor=self.F0_FLOOR,
            pitch_ceiling=self.F0_CEILING,
        )
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]
        if len(voiced) < 5:
            return None
        return F0Stats(
            mean=float(np.mean(voiced)),
            std=float(np.std(voiced)),
            voiced_fraction=float(len(voiced) / max(len(f0_values), 1)),
        )

    def drift(self, anchor: F0Stats, chunk: F0Stats) -> tuple[float, float]:
        mean_drift = abs(chunk.mean - anchor.mean) / max(anchor.mean, 1e-6)
        std_drift  = abs(chunk.std  - anchor.std)  / max(anchor.std,  1e-6)
        return mean_drift, std_drift

    def direction(self, anchor: F0Stats, chunk: F0Stats) -> str:
        return "high" if chunk.mean > anchor.mean else "low"


# ─────────────────────────────────────────────────────────────────────────────
# CVC REGISTRY + PATCHER
# ─────────────────────────────────────────────────────────────────────────────

F0_DESCRIPTORS: list[str] = [
    "very low-pitched",
    "low-pitched",
    "moderately low-pitched",
    "moderate pitch",
    "moderately high-pitched",
    "slightly high-pitched",
    "high-pitched",
]

_DEFAULT_F0_IDX = 1

_PITCH_RE = re.compile(
    r"very low-pitched|low-pitched|moderately low-pitched|moderate pitch"
    r"|moderately high-pitched|slightly high-pitched|high-pitched",
    re.IGNORECASE,
)


@dataclass
class CharacterRecord:
    cvc: str
    anchor_f0: Optional[F0Stats]
    anchor_embed: Optional[np.ndarray]
    f0_descriptor_idx: int = _DEFAULT_F0_IDX
    retry_counts: dict = field(default_factory=dict)

    def bump_f0(self, direction: str) -> bool:
        if direction == "high" and self.f0_descriptor_idx > 0:
            self.f0_descriptor_idx -= 1
            return True
        if direction == "low" and self.f0_descriptor_idx < len(F0_DESCRIPTORS) - 1:
            self.f0_descriptor_idx += 1
            return True
        return False


class CVCRegistry:
    def __init__(self):
        self._records: dict[str, CharacterRecord] = {}

    def has(self, char_id: str) -> bool:
        return char_id in self._records

    def get(self, char_id: str) -> CharacterRecord:
        return self._records[char_id]

    def register(self, char_id: str, record: CharacterRecord) -> None:
        self._records[char_id] = record

    def patch_f0(self, char_id: str, direction: str) -> str:
        rec = self._records[char_id]
        if not rec.bump_f0(direction):
            return rec.cvc
        new_descriptor = F0_DESCRIPTORS[rec.f0_descriptor_idx]
        new_cvc = _PITCH_RE.sub(new_descriptor, rec.cvc, count=1)
        if new_cvc == rec.cvc:
            new_cvc = rec.cvc.rstrip(".") + f", with a {new_descriptor} voice."
        rec.cvc = new_cvc
        return new_cvc

    def compose_caption(
        self,
        char_id: str,
        emotion: str,
        expressivity_phrase: str = "",
    ) -> str:
        """
        Composes the full IndicParlerTTS caption.
        CVC (frozen identity) + dynamic emotion token + expressivity phrase.
        """
        rec = self._records[char_id]
        base   = rec.cvc.rstrip(".")
        suffix = f"The intended style is {emotion}."
        if expressivity_phrase:
            suffix += f" {expressivity_phrase}"
        return f"{base}. {suffix}"


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER EMBEDDER — ECAPA-TDNN via SpeechBrain
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEmbedder:
    MODEL_HUB = "speechbrain/spkrec-ecapa-voxceleb"
    TARGET_SR  = 16000

    def __init__(self, device: str):
        self.device = device
        print("[SpeakerEmbedder] Loading ECAPA-TDNN ...")
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=self.MODEL_HUB,
                run_opts={"device": device},
                savedir="pretrained_models/ecapa-tdnn",
            )
        except ImportError:
            raise ImportError("Run: pip install speechbrain")
        print("[SpeakerEmbedder] Ready.")

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.TARGET_SR:
            audio = self._resample(audio, sr, self.TARGET_SR)
        wav = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        emb = self.model.encode_batch(wav)
        vec = emb.squeeze().cpu().numpy()
        return vec / (np.linalg.norm(vec) + 1e-8)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            ratio    = orig_sr / target_sr
            new_len  = int(len(audio) / ratio)
            indices  = np.clip((np.arange(new_len) * ratio).astype(int), 0, len(audio) - 1)
            return audio[indices]


# ─────────────────────────────────────────────────────────────────────────────
# INDICPARLER TTS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class IndicParlerTTS:
    MODEL_ID = "ai4bharat/indic-parler-tts"

    def __init__(self, device: str):
        self.device = device
        print(f"[IndicParlerTTS] Loading on {device} ...")
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from parler_tts.configuration_parler_tts import ParlerTTSConfig
            from transformers import AutoTokenizer, GenerationConfig, GenerationMixin
        except ImportError:
            raise ImportError(
                "Run: pip install git+https://github.com/huggingface/parler-tts.git"
            )
        # ── Compatibility shims for transformers ≥ 4.50 ──────────────────────
        # parler_tts was written against an older transformers where PreTrainedModel
        # inherited GenerationMixin.  Three things break with newer transformers:
        #
        # 1. to_diff_dict() tries ParlerTTSConfig() with no args (repr crash).
        # 2. generate() calls self._validate_model_kwargs() which only exists on
        #    GenerationMixin — no longer inherited via PreTrainedModel.
        # 3. generation_config is not auto-loaded for non-GenerationMixin models.
        #
        # All three patches are one-time class-level operations (guarded by flags).

        # Fix 1: safe to_diff_dict (repr/logging only — no functional impact)
        if not getattr(ParlerTTSConfig, "_to_diff_dict_patched", False):
            def _to_diff_dict_safe(self):
                return self.to_dict()
            ParlerTTSConfig.to_diff_dict = _to_diff_dict_safe
            ParlerTTSConfig._to_diff_dict_patched = True

        # Fix 2: inject GenerationMixin so generate() can call _validate_model_kwargs
        # and other mixin helpers.  Then delete parler_tts's own
        # _get_initial_cache_position — it uses the old 2-arg API while new
        # GenerationMixin._sample calls it with 3 args (cur_len, device, model_kwargs).
        # Removing it lets the correct GenerationMixin version be inherited instead.
        try:
            from parler_tts.modeling_parler_tts import ParlerTTSForCausalLM
            _gen_classes = (ParlerTTSForConditionalGeneration, ParlerTTSForCausalLM)
        except ImportError:
            _gen_classes = (ParlerTTSForConditionalGeneration,)
        for _cls in _gen_classes:
            if not issubclass(_cls, GenerationMixin):
                try:
                    _bases = list(_cls.__bases__)
                    _bases.insert(1, GenerationMixin)
                    _cls.__bases__ = tuple(_bases)
                except TypeError:
                    pass  # immutable or MRO conflict — skip
            # Drop parler_tts's old-signature _get_initial_cache_position
            if "_get_initial_cache_position" in _cls.__dict__:
                try:
                    delattr(_cls, "_get_initial_cache_position")
                except AttributeError:
                    pass

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.MODEL_ID
        ).to(device)
        if dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

        # Fix 3: manually load generation_config (not auto-loaded for non-mixin models)
        if self.model.generation_config is None:
            try:
                self.model.generation_config = GenerationConfig.from_pretrained(
                    self.MODEL_ID
                )
            except Exception:
                self.model.generation_config = GenerationConfig()

        # Fix 4: cache_position[0] under-counts past length because it tracks
        # only decoder positions, not the prompt tokens prepended as embeddings
        # on step 1.  On step 2+, prepare_inputs_for_generation uses
        # cache_position[0] (e.g. 1) instead of past_key_values.get_seq_length()
        # (e.g. 5 = 4 prompt + 1 decoder), so generated_length goes negative.
        # Clearing cache_position when the cache is already populated forces the
        # fallback to get_seq_length(), which is always correct.
        if not getattr(ParlerTTSForConditionalGeneration,
                       "_prepare_inputs_patched", False):
            _orig_prepare = \
                ParlerTTSForConditionalGeneration.prepare_inputs_for_generation
            def _patched_prepare(
                self, decoder_input_ids,
                past_key_values=None, cache_position=None, **kwargs
            ):
                if (past_key_values is not None
                        and cache_position is not None):
                    try:
                        if past_key_values.get_seq_length() > 0:
                            cache_position = None
                    except Exception:
                        pass
                return _orig_prepare(
                    self, decoder_input_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **kwargs,
                )
            ParlerTTSForConditionalGeneration.prepare_inputs_for_generation = \
                _patched_prepare
            ParlerTTSForConditionalGeneration._prepare_inputs_patched = True

        # Fix 5: transformers ≥ 4.50 replaced DynamicCache's flat key_cache /
        # value_cache lists with a layered DynamicLayer structure.  parler_tts's
        # attention code still indexes into past_key_value.key_cache[layer_idx]
        # (OLD API), which now raises AttributeError.  Add read-only properties
        # that expose the same interface via the new layer objects.
        from transformers.cache_utils import DynamicCache
        if not getattr(DynamicCache, "_legacy_api_patched", False):
            @property
            def _key_cache_compat(self):
                return [layer.keys for layer in self.layers]
            @property
            def _value_cache_compat(self):
                return [layer.values for layer in self.layers]
            DynamicCache.key_cache = _key_cache_compat
            DynamicCache.value_cache = _value_cache_compat
            DynamicCache._legacy_api_patched = True

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.desc_tokenizer   = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sr: int = self.model.config.sampling_rate
        print(f"[IndicParlerTTS] Ready. SR={self.sr} Hz")

    @torch.no_grad()
    def synthesize(self, text: str, caption: str) -> np.ndarray:
        desc_enc   = self.desc_tokenizer(caption, return_tensors="pt").to(self.device)
        prompt_enc = self.prompt_tokenizer(text,    return_tensors="pt").to(self.device)
        gen = self.model.generate(
            input_ids=desc_enc.input_ids,
            attention_mask=desc_enc.attention_mask,
            prompt_input_ids=prompt_enc.input_ids,
            prompt_attention_mask=prompt_enc.attention_mask,
        )
        return gen.cpu().numpy().squeeze().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT CLASSIFIER  — BERT text classification
# ─────────────────────────────────────────────────────────────────────────────

class SentimentClassifier:
    """
    Wraps tabularisai/multilingual-sentiment-analysis.
    Returns (stars 1-5, numeric score -1→1, confidence 0→1).
    """

    MODEL_ID = "tabularisai/multilingual-sentiment-analysis"

    _LABEL_TO_STARS: dict[str, int] = {
        "very negative": 1, "negative": 2, "neutral": 3,
        "positive": 4, "very positive": 5,
        "1 star": 1, "2 stars": 2, "3 stars": 3, "4 stars": 4, "5 stars": 5,
        "label_0": 1, "label_1": 2, "label_2": 3, "label_3": 4, "label_4": 5,
    }
    _STAR_TO_NUMERIC = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}

    def __init__(self, device: str):
        print(f"[SentimentClassifier] Loading {self.MODEL_ID} ...")
        from transformers import pipeline as hf_pipeline
        hf_device = 0 if device == "cuda" else -1
        self._pipe = hf_pipeline(
            "text-classification",
            model=self.MODEL_ID,
            device=hf_device,
            truncation=True,
            max_length=512,
        )
        print("[SentimentClassifier] Ready.")

    def predict(self, text: str) -> tuple[int, float, float]:
        """Returns (stars, numeric_score, confidence)."""
        result     = self._pipe(text)[0]
        label_key  = result["label"].lower().strip()
        stars      = self._LABEL_TO_STARS.get(label_key, 3)
        numeric    = self._STAR_TO_NUMERIC[stars]
        confidence = float(result["score"])
        return stars, numeric, confidence


# ─────────────────────────────────────────────────────────────────────────────
# EMOTION ASSIGNER  — maps BERT output + text heuristics → IndicParler style
# ─────────────────────────────────────────────────────────────────────────────

# ── English keyword patterns ──────────────────────────────────────────────────
_EN_DIALOGUE = re.compile(
    '[\u201c\u201d\u2018\u2019\'"]'   # curly and straight quote characters
    r'|(?:\b(?:said|asked|replied|whispered|shouted|murmured|answered|'
    r'explained|cried|laughed|sighed|hissed|snapped|exclaimed|called|'
    r'responded|muttered|roared|declared|announced)\b)',
    re.I,
)
_EN_IMPERATIVE = re.compile(
    r"^\s*(?:come|go|stop|wait|look|listen|run|get|take|put|make|do|try|help|"
    r"let|sit|stand|turn|open|close|tell|show|give|bring|move|stay|hold|keep|"
    r"speak|be|remember|forget|find|watch|hear|feel|follow|leave|return|"
    r"read|write|say|call|ask|answer|start|end|begin|finish)\b",
    re.I,
)
_EN_ANGER = re.compile(
    r"\b(?:ang(?:er|ry|rily)|rage|fury|furious|shout(?:ed|ing)?|yell(?:ed|ing)?|"
    r"scream(?:ed|ing)?|infuriat|wrath|snarl(?:ed)?|growl(?:ed)?|slam(?:med)?|"
    r"slam(?:ming)?|bitter(?:ly)?|fierce(?:ly)?|hatred|hate[ds]?|hostile|"
    r"glare[ds]?|glaring|outrage(?:d|ous)?|livid|seething)\b",
    re.I,
)
_EN_FEAR = re.compile(
    r"\b(?:fear(?:ed|ful|fully)?|scar(?:e[ds]?|ing)|terrif(?:ied|ying|y)?|"
    r"panic(?:ked|king)?|dread(?:ed|ful|fully)?|trembl(?:e[ds]?|ing)|"
    r"fright(?:en(?:ed|ing)?)?|horror|horrif(?:ied|ying)?|afraid|"
    r"shudder(?:ed|ing)?|quak(?:e[ds]?|ing)|cower(?:ed|ing)?|cringe[ds]?|"
    r"dread|ominous|menacing|threat(?:en(?:ing)?)?)\b",
    re.I,
)
_EN_DISGUST = re.compile(
    r"\b(?:disgust(?:ed|ing)?|revolting|nauseating|repuls(?:ive|ed)?|"
    r"gross|filthy|vile|loath(?:e[ds]?|ing|some)?|revolted|sicken(?:ed|ing)?|"
    r"abhor(?:rent|red)?|repugn(?:ant)?|detest(?:ed|ing)?)\b",
    re.I,
)
_EN_HAPPY = re.compile(
    r"\b(?:happi(?:ly|ness)?|joy(?:ful|ous|fully)?|laugh(?:ed|ing|ter)?|"
    r"smil(?:e[ds]?|ing)|delight(?:ed|ful|fully)?|cheer(?:ful|ed|ing|fully)?|"
    r"celebrat(?:e[ds]?|ing|ion)?|wonderful|warm(?:th|ly)?|"
    r"excit(?:ed|ing|edly)?|glee|bliss(?:ful)?|radiant|beam(?:ed|ing)?|"
    r"love[ds]?|loving|tender(?:ly)?|sweet(?:ly)?|elat(?:ed|ing)?)\b",
    re.I,
)
_EN_SURPRISE = re.compile(
    r"\b(?:surpris(?:e[ds]?|ing|ingly)?|shock(?:ed|ing)?|amaz(?:ed|ing|ingly)?|"
    r"astonish(?:ed|ing|ingly)?|startl(?:e[ds]?|ing)|unexpected(?:ly)?|"
    r"sudden(?:ly)?|gasp(?:ed|ing)?|awe(?:struck|d)?|stunned|jaw.drop|"
    r"disbelief|incredulous|unbelievable|stagger(?:ed|ing)?)\b",
    re.I,
)
_EN_NEWS = re.compile(
    r"\b(?:according(?:\s+to)?|report(?:ed|ing|s)?|announced|stated|"
    r"official(?:ly)?|confirm(?:ed|ing)?|sources?|spokesman|spokesperson|"
    r"statement|press\s+release|revealed|disclosed)\b",
    re.I,
)
_EN_PROPER_NOUN = re.compile(
    r"(?<![.!?]\s)(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})",  # 3+ title-case words
)

# ── Expressivity phrase table ─────────────────────────────────────────────────
_EXPRESSIVITY: dict[tuple[str, str], str] = {
    ("Anger",        "high"): "Delivered with intense anger and a sharp, raised voice.",
    ("Anger",        "mid"):  "Delivered with underlying frustration in an otherwise controlled voice.",
    ("Anger",        "low"):  "Delivered with a tinge of irritation in an otherwise even voice.",
    ("Fear",         "high"): "Delivered with open terror and a trembling, urgent quality.",
    ("Fear",         "mid"):  "Delivered with a nervous, uneasy quality.",
    ("Fear",         "low"):  "Delivered with a subtle undercurrent of tension.",
    ("Disgust",      "high"): "Delivered with cold disdain and a repulsed, strained quality.",
    ("Disgust",      "mid"):  "Delivered with an understated distaste.",
    ("Disgust",      "low"):  "Delivered with mild discomfort in an otherwise even voice.",
    ("Happy",        "high"): "Delivered with bright warmth and upbeat, animated energy.",
    ("Happy",        "mid"):  "Delivered with warmth and gentle lightness.",
    ("Happy",        "low"):  "Delivered with a quietly pleasant and content tone.",
    ("Surprise",     "high"): "Delivered with sharp surprise and heightened animation.",
    ("Surprise",     "mid"):  "Delivered with a sense of wonder and mild astonishment.",
    ("Surprise",     "low"):  "Delivered with a slight note of the unexpected.",
    ("Sad",          "high"): "Delivered with deep melancholy and a slow, heavy pace.",
    ("Sad",          "mid"):  "Delivered with quiet sadness and a subdued tone.",
    ("Sad",          "low"):  "Delivered with a slightly somber undertone.",
    ("Command",      "high"): "Delivered with firm authority and a direct, strong voice.",
    ("Command",      "mid"):  "Delivered with clear direction and steady conviction.",
    ("Command",      "low"):  "Delivered with gentle but clear instruction.",
    ("Conversation", "high"): "Delivered with natural, animated conversational energy.",
    ("Conversation", "mid"):  "Delivered with relaxed, natural conversational pacing.",
    ("Conversation", "low"):  "Delivered in a casual, easy conversational tone.",
    ("Narration",    "high"): "Delivered with rich, expressive narrative storytelling.",
    ("Narration",    "mid"):  "Delivered at a steady, measured narrative pace.",
    ("Narration",    "low"):  "Delivered in a calm, clear narrative voice.",
    ("Neutral",      "high"): "Delivered with a calm and even tone.",
    ("Neutral",      "mid"):  "Delivered with a calm and even tone.",
    ("Neutral",      "low"):  "Delivered with a calm and even tone.",
    ("News",         "high"): "Delivered with crisp, journalistic clarity and composure.",
    ("News",         "mid"):  "Delivered in a clear, composed, journalistic style.",
    ("News",         "low"):  "Delivered with a measured, informational quality.",
    ("Proper Noun",  "high"): "Delivered clearly and distinctly, giving weight to names and places.",
    ("Proper Noun",  "mid"):  "Delivered clearly with attention to named references.",
    ("Proper Noun",  "low"):  "Delivered in a clear, informational tone.",
}


class EmotionAssigner:
    """
    Maps BERT sentiment score + text heuristics to an IndicParler emotion token
    and a natural-language expressivity phrase.

    Decision logic
    --------------
    1. Map numeric score [-1, 1] to a sentiment bin
    2. Within the bin, check keyword/pattern overrides
    3. Cross-bin upgrades: dialogue → Conversation, imperative → Command
    4. Look up (emotion, intensity) → expressivity phrase
    """

    def __init__(self, default_emotion: str = "Narration"):
        self.default_emotion = default_emotion

    def assign(
        self,
        text: str,
        numeric_score: float,
        confidence: float,
    ) -> tuple[str, str]:
        """
        Returns (emotion_token, expressivity_phrase).
        Falls back to default_emotion when confidence < 0.45.
        """
        if confidence < 0.45:
            emotion = self.default_emotion
        else:
            emotion = self._pick_emotion(text, numeric_score)

        intensity = _score_intensity(numeric_score)
        phrase    = _EXPRESSIVITY.get((emotion, intensity), "")
        return emotion, phrase

    def _pick_emotion(self, text: str, score: float) -> str:
        has_anger    = bool(_EN_ANGER.search(text))
        has_fear     = bool(_EN_FEAR.search(text))
        has_disgust  = bool(_EN_DISGUST.search(text))
        has_happy    = bool(_EN_HAPPY.search(text))
        has_surprise = bool(_EN_SURPRISE.search(text))
        has_dialogue = bool(_EN_DIALOGUE.search(text))
        has_command  = bool(_EN_IMPERATIVE.match(text))
        has_news     = bool(_EN_NEWS.search(text))
        has_proper   = bool(_EN_PROPER_NOUN.search(text))

        # Command and Conversation are cross-bin — check first
        if has_command and score >= -0.2:
            return "Command"

        # Very Negative
        if score < -0.65:
            if has_disgust and not has_anger:
                return "Disgust"
            if has_fear:
                return "Fear"
            return "Anger"

        # Negative
        if score < -0.2:
            if has_anger:
                return "Anger"
            if has_disgust:
                return "Disgust"
            if has_fear:
                return "Fear"
            return "Sad"

        # Neutral
        if score <= 0.2:
            if has_dialogue:
                return "Conversation"
            if has_news:
                return "News"
            if has_proper and not has_dialogue:
                return "Proper Noun"
            return "Narration"

        # Positive
        if score <= 0.65:
            if has_surprise:
                return "Surprise"
            if has_dialogue:
                return "Conversation"
            return "Happy"

        # Very Positive
        if has_surprise:
            return "Surprise"
        return "Happy"


def _score_intensity(score: float) -> str:
    abs_s = abs(score)
    if abs_s > 0.65:
        return "high"
    elif abs_s > 0.3:
        return "mid"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class SentimentNarratorPipeline:
    """
    End-to-end sentiment-guided TTS for an English story.

    Per-chunk order:
      1. BERT sentiment → numeric score + confidence
      2. EmotionAssigner → (emotion_token, expressivity_phrase)
      3. Compose caption: frozen CVC + emotion + expressivity phrase
      4. IndicParlerTTS synthesis
      5. Praat F0 gate
      6. ECAPA cosine gate (only if F0 passed)
      7. Patch CVC F0 descriptor + retry on failure (max 2)
      8. Flag + emit last audio on exhaustion
    """

    NARRATOR_ID = "narrator"

    def __init__(self, cfg: PipelineConfig):
        self.cfg            = cfg
        self.chunker        = TextChunker(cfg)
        self.tts            = IndicParlerTTS(cfg.device)
        self.embedder       = SpeakerEmbedder(cfg.device)
        self.f0_extractor   = F0Extractor()
        self.registry       = CVCRegistry()
        self.sentiment      = SentimentClassifier(cfg.device)
        self.emotion_assign = EmotionAssigner(cfg.default_emotion)

    # ── Anchor ────────────────────────────────────────────────────────────────

    def _build_anchor(self) -> None:
        print("\n[Anchor] Synthesizing narrator anchor clip...")
        # Use default_emotion for anchor to establish the base identity
        anchor_caption = (
            f"{self.cfg.narrator_cvc.rstrip('.')}. "
            f"The intended style is {self.cfg.default_emotion}."
        )
        audio   = self.tts.synthesize(self.cfg.anchor_text, anchor_caption)
        f0_stats = self.f0_extractor.extract(audio, self.tts.sr)
        if f0_stats is None:
            raise RuntimeError("F0 extraction failed on anchor. Check narrator_cvc.")
        embed = self.embedder.embed(audio, self.tts.sr)

        rec = CharacterRecord(
            cvc=self.cfg.narrator_cvc,
            anchor_f0=f0_stats,
            anchor_embed=embed,
            f0_descriptor_idx=_DEFAULT_F0_IDX,
        )
        self.registry.register(self.NARRATOR_ID, rec)
        print(
            f"[Anchor] F0 mean={f0_stats.mean:.1f} Hz  "
            f"std={f0_stats.std:.1f} Hz  "
            f"voiced={f0_stats.voiced_fraction:.2%}"
        )

    # ── Chunk validation ──────────────────────────────────────────────────────

    def _validate(self, audio: np.ndarray) -> tuple[bool, str, dict]:
        rec  = self.registry.get(self.NARRATOR_ID)
        diag: dict = {}

        chunk_f0 = self.f0_extractor.extract(audio, self.tts.sr)
        if chunk_f0 is None:
            diag["f0"] = "extraction_failed"
            return False, "low", diag

        mean_drift, std_drift = self.f0_extractor.drift(rec.anchor_f0, chunk_f0)
        direction             = self.f0_extractor.direction(rec.anchor_f0, chunk_f0)
        f0_ok = (
            mean_drift <= self.cfg.f0_drift_threshold
            and std_drift <= self.cfg.f0_std_ratio_threshold
        )
        diag.update({
            "f0_anchor_mean":  rec.anchor_f0.mean,
            "f0_anchor_std":   rec.anchor_f0.std,
            "f0_chunk_mean":   round(chunk_f0.mean, 1),
            "f0_chunk_std":    round(chunk_f0.std,  1),
            "f0_mean_drift":   round(mean_drift,    4),
            "f0_std_drift":    round(std_drift,     4),
            "f0_direction":    direction,
            "f0_passed":       f0_ok,
        })

        if not f0_ok:
            diag["ecapa_sim"] = "skipped (F0 failed)"
            return False, direction, diag

        chunk_embed = self.embedder.embed(audio, self.tts.sr)
        sim         = self.embedder.cosine(chunk_embed, rec.anchor_embed)
        spk_ok      = sim >= self.cfg.speaker_sim_threshold
        diag.update({"ecapa_sim": round(sim, 4), "ecapa_passed": spk_ok})
        return spk_ok, direction, diag

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, story_path: str, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        text   = Path(story_path).read_text(encoding="utf-8")
        chunks = self.chunker.chunk(text)
        n      = len(chunks)
        print(f"\n[Pipeline] Story: {len(text)} chars → {n} chunks")

        self._build_anchor()

        run_log:      list[dict]      = []
        segments:     list[np.ndarray] = []
        flagged:      list[int]       = []
        total_retries = 0

        for idx, chunk in enumerate(chunks):
            preview = chunk[:70] + ("..." if len(chunk) > 70 else "")
            print(f"\n[{idx+1:>3}/{n}] {preview}")

            # ── Step 1: Sentiment → Emotion ──────────────────────────────────
            stars, score, conf = self.sentiment.predict(chunk)
            emotion, expr_phrase = self.emotion_assign.assign(chunk, score, conf)
            intensity = _score_intensity(score)
            print(
                f"  [Sentiment] stars={stars}  score={score:+.2f}  conf={conf:.2f}"
                f"  → emotion={emotion!r} ({intensity})"
            )

            # ── Step 2: TTS + validation loop ────────────────────────────────
            last_audio: Optional[np.ndarray] = None
            passed = False

            for attempt in range(self.cfg.max_retries + 1):
                caption = self.registry.compose_caption(
                    self.NARRATOR_ID,
                    emotion=emotion,
                    expressivity_phrase=expr_phrase,
                )
                audio      = self.tts.synthesize(chunk, caption)
                last_audio = audio

                ok, direction, diag = self._validate(audio)
                log_entry = {
                    "chunk_idx":         idx,
                    "attempt":           attempt,
                    "sentiment_stars":   stars,
                    "sentiment_score":   round(score, 4),
                    "sentiment_conf":    round(conf, 4),
                    "emotion":           emotion,
                    "expressivity":      expr_phrase,
                    "caption":           caption,
                    "passed":            ok,
                    **diag,
                }
                run_log.append(log_entry)

                f0_str  = f"F0={diag.get('f0_chunk_mean','?')} Hz drift={diag.get('f0_mean_drift','?'):.3f}"
                sim_str = f"ECAPA={diag.get('ecapa_sim','skipped')}"
                marker  = "✓" if ok else "✗"
                print(f"  [{marker}] attempt={attempt}  {f0_str}  {sim_str}")

                if ok:
                    passed = True
                    break

                if attempt < self.cfg.max_retries:
                    new_cvc = self.registry.patch_f0(self.NARRATOR_ID, direction)
                    total_retries += 1
                    new_desc = F0_DESCRIPTORS[self.registry.get(self.NARRATOR_ID).f0_descriptor_idx]
                    print(f"  [Patch] dir='{direction}' → descriptor='{new_desc}'")

            if not passed:
                flagged.append(idx)
                print(f"  [Flag] Chunk {idx} exhausted retries — emitting last audio.")

            segments.append(last_audio)

        # ── Write outputs ─────────────────────────────────────────────────────
        final_audio = np.concatenate(segments)
        audio_out   = output_path / "story_narration.wav"
        sf.write(str(audio_out), final_audio, self.tts.sr)
        print(f"\n[Output] Audio → {audio_out}  ({len(final_audio)/self.tts.sr:.1f}s)")

        summary = {
            "story":              str(story_path),
            "language":           "english",
            "model":              IndicParlerTTS.MODEL_ID,
            "sentiment_model":    SentimentClassifier.MODEL_ID,
            "narrator_cvc_final": self.registry.get(self.NARRATOR_ID).cvc,
            "anchor_f0":          self.registry.get(self.NARRATOR_ID).anchor_f0.to_dict(),
            "total_chunks":       n,
            "total_retries":      total_retries,
            "flagged_chunks":     flagged,
            "f0_drift_threshold": self.cfg.f0_drift_threshold,
            "f0_std_threshold":   self.cfg.f0_std_ratio_threshold,
            "chunks":             run_log,
        }
        log_out = output_path / "pipeline_log.json"
        with open(log_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Output] Log   → {log_out}")

        # ── Emotion distribution summary ──────────────────────────────────────
        from collections import Counter
        emotion_dist = Counter(e["emotion"] for e in run_log if e["attempt"] == 0)
        print("\n" + "─" * 60)
        print(f"  Language:      English")
        print(f"  Chunks:        {n}")
        print(f"  Total retries: {total_retries}")
        print(f"  Flagged:       {len(flagged)}" + (f"  {flagged}" if flagged else "  (none)"))
        print(f"  Emotion dist:  {dict(emotion_dist.most_common())}")
        print(f"  Final CVC:     {self.registry.get(self.NARRATOR_ID).cvc[:90]}...")
        print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sentiment-guided English IndicParlerTTS pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--story",   default="transcripts/bad_blood_eng.txt",
                   help="Path to story text file (UTF-8, English)")
    p.add_argument("--output",  default="output/ENG_sentiment/",
                   help="Directory for audio and log outputs")
    p.add_argument("--narrator-cvc", default=None,
                   help="Override narrator CVC string")
    p.add_argument("--default-emotion", default="Narration",
                   help="Fallback emotion when sentiment confidence is low")
    p.add_argument("--anchor-text", default=None,
                   help="English seed sentence for anchor clip")
    p.add_argument("--f0-threshold",     type=float, default=0.15)
    p.add_argument("--f0-std-threshold", type=float, default=0.35)
    p.add_argument("--sim-threshold",    type=float, default=0.82)
    p.add_argument("--max-retries",      type=int,   default=2)
    p.add_argument("--max-chunk-chars",  type=int,   default=300)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = PipelineConfig()

    if args.narrator_cvc:   cfg.narrator_cvc          = args.narrator_cvc
    if args.anchor_text:    cfg.anchor_text            = args.anchor_text
    if args.device:         cfg.device                 = args.device
    cfg.default_emotion          = args.default_emotion
    cfg.f0_drift_threshold       = args.f0_threshold
    cfg.f0_std_ratio_threshold   = args.f0_std_threshold
    cfg.speaker_sim_threshold    = args.sim_threshold
    cfg.max_retries              = args.max_retries
    cfg.max_chunk_chars          = args.max_chunk_chars

    pipeline = SentimentNarratorPipeline(cfg)
    pipeline.run(args.story, args.output)


if __name__ == "__main__":
    main()
