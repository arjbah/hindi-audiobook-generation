#!/usr/bin/env python3
"""
IndicParlerTTS Long-Form Narrator Pipeline
==========================================
F0-anchored speaker identity consistency for single-narrator Hindi stories.

Architecture
------------
  Anchor (one-time)
    synthesize short seed clip → ECAPA-TDNN embed → Praat F0 stats → registry

  Per-chunk loop (streaming)
    compose caption (frozen CVC + style suffix)
    → IndicParlerTTS inference
    → Praat F0 validation  [primary gate, ~3 ms, CPU]
    → if F0 passed: ECAPA cosine check  [secondary, ~45 ms]
    → if failed: patch CVC F0 descriptor (string op, 0 GPU) → retry (max 2)
    → if still failing: flag for review, emit last audio anyway

Low-compute choices
-------------------
  - Praat F0 (parselmouth): CPU, ~3 ms per clip — the primary consistency gate
  - ECAPA runs on every chunk but only after F0 passes (skipped on obvious drift)
  - CVC patching is pure string replacement — no GPU, no model calls
  - Both models loaded once at startup, stay resident

Usage
-----
  python indic_narrator_pipeline.py --story path/to/story.txt --output out/
  python indic_narrator_pipeline.py --story story.txt --narrator-cvc "Rohit..." --f0-threshold 0.10
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Narrator voice ────────────────────────────────────────────────────────
    # Rohit is a recommended Hindi male speaker in IndicParlerTTS.
    # This CVC is the "frozen" identity layer — only the F0 descriptor changes
    # dynamically when drift is detected.
    narrator_cvc: str = (
        "Rohit speaks at a slightly slow pace with a deep, low-pitched voice "
        "and a narrow pitch range in a very close-sounding studio environment. "
        "The audio is of excellent quality with no background noise."
    )
    narrator_style: str = "Narration"       # IndicParlerTTS emotion token
    narrator_rate_modifier: str = ""        # optional: "slightly slow", "moderate", etc.

    # ── Chunking ──────────────────────────────────────────────────────────────
    max_chunk_chars: int = 280              # keeps TTS inference latency bounded
    min_chunk_chars: int = 40              # avoids degenerate short clips

    # ── F0 consistency (primary gate) ─────────────────────────────────────────
    # 12% relative deviation on F0 mean is the primary trip wire.
    # F0 std drift catches voices that wander in expressivity range.
    f0_drift_threshold: float = 0.12       # relative deviation from anchor mean
    f0_std_ratio_threshold: float = 0.30   # relative deviation from anchor std

    # ── Speaker embedding (secondary gate) ────────────────────────────────────
    # ECAPA cosine checked only after F0 passes — avoids cost on obvious drift.
    speaker_sim_threshold: float = 0.82

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 2                   # worst case = 3× TTS per chunk

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Anchor ────────────────────────────────────────────────────────────────
    # Short Hindi sentence used to build the character anchor clip.
    # Should be neutral, mid-length, same language as story.
    anchor_text: str = "एक शांत और स्थिर आवाज़ में, यह कहानी शुरू होती है।"


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits Devanagari/Hindi text at sentence and paragraph boundaries,
    then groups into chunks under max_chunk_chars.

    Handles:
      ।  ॥  (Devanagari dandas)
      .  !  ?  (standard punctuation)
      Quoted dialogue lines separated by blank lines
    """

    # Sentence-ending punctuation — split after these
    _DANDA = re.compile(r'(?<=[।॥])\s+')
    _STANDARD = re.compile(r'(?<=[.!?])\s+(?=[^\d])')
    # Paragraph breaks (two or more newlines)
    _PARA = re.compile(r'\n{2,}')

    def __init__(self, cfg: PipelineConfig):
        self.max_chars = cfg.max_chunk_chars
        self.min_chars = cfg.min_chunk_chars

    def chunk(self, text: str) -> list[str]:
        text = text.strip()

        # 1. Split at paragraphs first, then at sentence boundaries within each paragraph
        paragraphs = self._PARA.split(text)
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Split the paragraph into sentences
            parts = self._DANDA.split(para)
            for part in parts:
                subparts = self._STANDARD.split(part)
                sentences.extend(s.strip() for s in subparts if s.strip())

        # 2. Greedy grouping: pack sentences into chunks under max_chars
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
    mean: float           # Hz, voiced frames only
    std: float            # Hz, voiced frames only
    voiced_fraction: float  # proportion of frames that are voiced

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
    """
    Praat autocorrelation F0 via parselmouth.
    Configured for a male Hindi narrator; covers female range too (75–300 Hz).

    Why Praat over pyin/CREPE:
      - parselmouth is a thin C++ binding — no Python overhead in the hot path
      - autocorrelation is accurate for clean studio speech (which IndicParlerTTS produces)
      - ~3 ms per 5-second clip; pyin is ~25 ms, CREPE is GPU-dependent
    """
    F0_FLOOR = 75.0      # Hz — bottom of male narrator range
    F0_CEILING = 300.0   # Hz — top of female range (headroom for style transfer)
    TIME_STEP = 0.01     # s — 10 ms hop, matches RASMALAI attribute extraction

    def extract(self, audio: np.ndarray, sr: int) -> Optional[F0Stats]:
        """Extract voiced F0 statistics. Returns None if clip has < 5 voiced frames."""
        try:
            import parselmouth
        except ImportError:
            raise ImportError(
                "parselmouth not installed. Run: pip install praat-parselmouth"
            )
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
        """
        Returns (mean_drift_ratio, std_drift_ratio).
        Both are relative (dimensionless) — threshold comparison is scale-free.
        """
        mean_drift = abs(chunk.mean - anchor.mean) / max(anchor.mean, 1e-6)
        std_drift = abs(chunk.std - anchor.std) / max(anchor.std, 1e-6)
        return mean_drift, std_drift

    def direction(self, anchor: F0Stats, chunk: F0Stats) -> str:
        """'high' if chunk F0 crept up, 'low' if it dropped."""
        return "high" if chunk.mean > anchor.mean else "low"


# ─────────────────────────────────────────────────────────────────────────────
# CVC REGISTRY + PATCHER
# ─────────────────────────────────────────────────────────────────────────────

# F0 descriptor vocabulary.
# These exact phrases appear in RASMALAI/IndicParlerTTS training captions,
# making them the most reliable control tokens for the model.
# Ordered low → high (approximate Hz midpoints for male narrator range).
F0_DESCRIPTORS: list[str] = [
    "very low-pitched",         # ~90 Hz
    "low-pitched",              # ~110 Hz
    "moderately low-pitched",   # ~130 Hz
    "moderate pitch",           # ~155 Hz
    "moderately high-pitched",  # ~175 Hz
    "slightly high-pitched",    # ~200 Hz
    "high-pitched",             # ~230 Hz
]

# Starting index for "low-pitched" (matches default narrator CVC)
_DEFAULT_F0_IDX = 1

# Regex to find the current F0 descriptor token inside a CVC string
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
        """
        Shifts F0 descriptor one step toward the corrective direction.
          direction='high' → chunk was too high → push descriptor lower (idx--)
          direction='low'  → chunk was too low  → push descriptor higher (idx++)
        Returns True if the descriptor actually changed.
        """
        if direction == "high" and self.f0_descriptor_idx > 0:
            self.f0_descriptor_idx -= 1
            return True
        if direction == "low" and self.f0_descriptor_idx < len(F0_DESCRIPTORS) - 1:
            self.f0_descriptor_idx += 1
            return True
        return False


class CVCRegistry:
    """In-memory store: character_id → CharacterRecord."""

    def __init__(self):
        self._records: dict[str, CharacterRecord] = {}

    def has(self, char_id: str) -> bool:
        return char_id in self._records

    def get(self, char_id: str) -> CharacterRecord:
        return self._records[char_id]

    def register(self, char_id: str, record: CharacterRecord) -> None:
        self._records[char_id] = record

    # ── CVC operations ────────────────────────────────────────────────────────

    def patch_f0(self, char_id: str, direction: str) -> str:
        """
        Patches the F0 descriptor in the stored CVC string.
        Pure string op — zero GPU, zero model calls.
        Updates the registry in-place so future chunks use the corrected CVC.
        Returns the new CVC string.
        """
        rec = self._records[char_id]
        changed = rec.bump_f0(direction)
        if not changed:
            return rec.cvc  # already at limit — nothing to patch

        new_descriptor = F0_DESCRIPTORS[rec.f0_descriptor_idx]
        new_cvc = _PITCH_RE.sub(new_descriptor, rec.cvc, count=1)

        if new_cvc == rec.cvc:
            # No existing pitch descriptor found — append the constraint
            new_cvc = rec.cvc.rstrip(".") + f", with a {new_descriptor} voice."

        rec.cvc = new_cvc
        return new_cvc

    def compose_caption(self, char_id: str, style: str, rate_modifier: str = "") -> str:
        """
        Composes the full IndicParlerTTS caption for one inference call.
        CVC (frozen identity) + expressivity suffix (per-chunk).
        """
        rec = self._records[char_id]
        base = rec.cvc.rstrip(".")
        suffix = f"The intended style is {style}."
        if rate_modifier:
            suffix += f" Delivered at a {rate_modifier} pace."
        return f"{base}. {suffix}"

    def increment_retry(self, char_id: str, chunk_idx: int) -> None:
        self._records[char_id].retry_counts[chunk_idx] = (
            self._records[char_id].retry_counts.get(chunk_idx, 0) + 1
        )

    def retry_count(self, char_id: str, chunk_idx: int) -> int:
        return self._records[char_id].retry_counts.get(chunk_idx, 0)


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER EMBEDDER — ECAPA-TDNN via SpeechBrain
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEmbedder:
    """
    ECAPA-TDNN loaded once at startup.

    Role in the pipeline:
      - Anchor extraction: one call per character, stores reference embedding
      - Per-chunk validation: cosine distance to anchor (secondary gate,
        only runs after F0 passes — skips the ECAPA call on obvious F0 drift)

    ECAPA expects 16 kHz mono input; resampling applied automatically.
    """
    MODEL_HUB = "speechbrain/spkrec-ecapa-voxceleb"
    TARGET_SR = 16000

    def __init__(self, device: str):
        self.device = device
        print("[SpeakerEmbedder] Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
        try:
            from speechbrain.pretrained import EncoderClassifier
            self.model = EncoderClassifier.from_hparams(
                source=self.MODEL_HUB,
                run_opts={"device": device},
                savedir="pretrained_models/ecapa-tdnn",
            )
        except ImportError:
            raise ImportError(
                "speechbrain not installed. Run: pip install speechbrain"
            )
        print("[SpeakerEmbedder] Ready.")

    @torch.no_grad()
    def embed(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Returns L2-normalized 192-dim ECAPA embedding."""
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
            # Fallback: naive decimation (acceptable for 44100 → 16000)
            ratio = orig_sr / target_sr
            new_len = int(len(audio) / ratio)
            indices = (np.arange(new_len) * ratio).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            return audio[indices]


# ─────────────────────────────────────────────────────────────────────────────
# INDICPARLER TTS WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class IndicParlerTTS:
    """
    Thin wrapper around ai4bharat/indic-parler-tts.

    Both tokenizers are loaded once:
      - description_tokenizer: Flan-T5 family (text encoder)
      - prompt_tokenizer: expanded Llama2 tokenizer (for Indic scripts)

    Float16 on CUDA, float32 on CPU.
    """
    MODEL_ID = "ai4bharat/indic-parler-tts"

    def __init__(self, device: str):
        self.device = device
        print(f"[IndicParlerTTS] Loading model on {device} ...")
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "parler_tts not installed.\n"
                "Run: pip install git+https://github.com/huggingface/parler-tts.git"
            )

        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()

        self.prompt_tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sr: int = self.model.config.sampling_rate
        print(f"[IndicParlerTTS] Ready. Sample rate: {self.sr} Hz")

    @torch.no_grad()
    def synthesize(self, text: str, caption: str) -> np.ndarray:
        """
        Synthesizes audio for one chunk.
        Returns float32 mono waveform at self.sr.
        """
        desc_enc = self.desc_tokenizer(caption, return_tensors="pt").to(self.device)
        prompt_enc = self.prompt_tokenizer(text, return_tensors="pt").to(self.device)

        gen = self.model.generate(
            input_ids=desc_enc.input_ids,
            attention_mask=desc_enc.attention_mask,
            prompt_input_ids=prompt_enc.input_ids,
            prompt_attention_mask=prompt_enc.attention_mask,
        )
        return gen.cpu().numpy().squeeze().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class NarratorPipeline:
    """
    End-to-end long-form TTS for a single-narrator story.

    Processing order per chunk:
      1. Compose caption from registry (frozen CVC + style suffix)
      2. IndicParlerTTS → audio
      3. Praat F0 extraction → compare mean + std drift against anchor
      4. If F0 passed → ECAPA cosine check against anchor embedding
      5. If either gate fails → patch CVC F0 descriptor (string op) → retry
      6. After max_retries → flag chunk, emit last audio, continue
    """

    NARRATOR_ID = "narrator"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.chunker = TextChunker(cfg)
        self.tts = IndicParlerTTS(cfg.device)
        self.embedder = SpeakerEmbedder(cfg.device)
        self.f0_extractor = F0Extractor()
        self.registry = CVCRegistry()

    # ── Anchor ────────────────────────────────────────────────────────────────

    def _build_anchor(self) -> None:
        """
        Synthesizes the anchor clip from the seed CVC, extracts F0 stats
        and ECAPA embedding, and registers them as the narrator's identity.
        Called exactly once per pipeline run.
        """
        print("\n[Anchor] Synthesizing narrator anchor clip...")
        audio = self.tts.synthesize(self.cfg.anchor_text, self.cfg.narrator_cvc)

        f0_stats = self.f0_extractor.extract(audio, self.tts.sr)
        if f0_stats is None:
            raise RuntimeError(
                "F0 extraction failed on anchor clip. "
                "Check that narrator_cvc produces voiced speech."
            )
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

    def _validate(
        self,
        audio: np.ndarray,
    ) -> tuple[bool, str, dict]:
        """
        Validates a synthesized chunk against the narrator anchor.

        Returns:
          passed (bool)
          drift_direction (str)  — 'high' or 'low', used for CVC patching
          diagnostics (dict)     — written to the run log
        """
        rec = self.registry.get(self.NARRATOR_ID)
        diag: dict = {}

        # ── Primary: F0 ─────────────────────────────────────────────────────
        chunk_f0 = self.f0_extractor.extract(audio, self.tts.sr)
        if chunk_f0 is None:
            # Degenerate clip — treat as low-pitch failure to trigger a retry
            diag["f0"] = "extraction_failed"
            return False, "low", diag

        mean_drift, std_drift = self.f0_extractor.drift(rec.anchor_f0, chunk_f0)
        direction = self.f0_extractor.direction(rec.anchor_f0, chunk_f0)

        f0_ok = (
            mean_drift <= self.cfg.f0_drift_threshold
            and std_drift <= self.cfg.f0_std_ratio_threshold
        )
        diag.update({
            "f0_anchor_mean": rec.anchor_f0.mean,
            "f0_anchor_std": rec.anchor_f0.std,
            "f0_chunk_mean": round(chunk_f0.mean, 1),
            "f0_chunk_std": round(chunk_f0.std, 1),
            "f0_mean_drift": round(mean_drift, 4),
            "f0_std_drift": round(std_drift, 4),
            "f0_direction": direction,
            "f0_passed": f0_ok,
        })

        if not f0_ok:
            diag["ecapa_sim"] = "skipped (F0 failed)"
            return False, direction, diag

        # ── Secondary: ECAPA speaker similarity ──────────────────────────────
        chunk_embed = self.embedder.embed(audio, self.tts.sr)
        sim = self.embedder.cosine(chunk_embed, rec.anchor_embed)
        spk_ok = sim >= self.cfg.speaker_sim_threshold
        diag.update({
            "ecapa_sim": round(sim, 4),
            "ecapa_passed": spk_ok,
        })

        return spk_ok, direction, diag

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, story_path: str, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load and chunk story
        text = Path(story_path).read_text(encoding="utf-8")
        chunks = self.chunker.chunk(text)
        n = len(chunks)
        print(f"\n[Pipeline] Story loaded: {len(text)} chars → {n} chunks")

        # One-time anchor
        self._build_anchor()

        # Per-chunk processing
        run_log: list[dict] = []
        segments: list[np.ndarray] = []
        flagged: list[int] = []
        total_retries = 0

        for idx, chunk in enumerate(chunks):
            print(f"\n[{idx+1:>3}/{n}] {chunk[:70]}{'...' if len(chunk)>70 else ''}")
            last_audio: Optional[np.ndarray] = None
            passed = False

            for attempt in range(self.cfg.max_retries + 1):
                caption = self.registry.compose_caption(
                    self.NARRATOR_ID,
                    style=self.cfg.narrator_style,
                    rate_modifier=self.cfg.narrator_rate_modifier,
                )
                audio = self.tts.synthesize(chunk, caption)
                last_audio = audio

                ok, direction, diag = self._validate(audio)
                log_entry = {
                    "chunk_idx": idx,
                    "attempt": attempt,
                    "caption": caption,
                    "passed": ok,
                    **diag,
                }
                run_log.append(log_entry)

                f0_str = f"F0={diag.get('f0_chunk_mean','?')} Hz drift={diag.get('f0_mean_drift','?'):.3f}"
                sim_str = f"ECAPA={diag.get('ecapa_sim','skipped')}"
                marker = "✓" if ok else "✗"
                print(f"  [{marker}] attempt={attempt}  {f0_str}  {sim_str}")

                if ok:
                    passed = True
                    break

                if attempt < self.cfg.max_retries:
                    new_cvc = self.registry.patch_f0(self.NARRATOR_ID, direction)
                    total_retries += 1
                    print(
                        f"  [Patch] F0 drift direction='{direction}' → "
                        f"new descriptor: '{F0_DESCRIPTORS[self.registry.get(self.NARRATOR_ID).f0_descriptor_idx]}'"
                    )
                    print(f"          CVC → {new_cvc[:90]}...")

            if not passed:
                flagged.append(idx)
                print(f"  [Flag] Chunk {idx} failed after all retries — emitting last audio.")

            segments.append(last_audio)

        # ── Write outputs ─────────────────────────────────────────────────────
        final_audio = np.concatenate(segments)
        audio_out = output_path / "story_narration.wav"
        sf.write(str(audio_out), final_audio, self.tts.sr)
        print(f"\n[Output] Audio → {audio_out}  ({len(final_audio)/self.tts.sr:.1f}s)")

        summary = {
            "story": str(story_path),
            "narrator_cvc_final": self.registry.get(self.NARRATOR_ID).cvc,
            "anchor_f0": self.registry.get(self.NARRATOR_ID).anchor_f0.to_dict(),
            "total_chunks": n,
            "total_retries": total_retries,
            "flagged_chunks": flagged,
            "chunks": run_log,
        }
        log_out = output_path / "pipeline_log.json"
        with open(log_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Output] Log   → {log_out}")

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "─" * 60)
        print(f"  Chunks:        {n}")
        print(f"  Total retries: {total_retries}")
        print(f"  Flagged:       {len(flagged)}" +
              (f"  {flagged}" if flagged else "  (none)"))
        print(f"  Final CVC:     {self.registry.get(self.NARRATOR_ID).cvc[:90]}...")
        print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IndicParlerTTS F0-anchored narrator pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--story", required=True,
        help="Path to story text file (UTF-8, Hindi/Devanagari)",
    )
    p.add_argument(
        "--output", default="output/",
        help="Directory for audio and log outputs",
    )
    p.add_argument(
        "--narrator-cvc", default=None,
        help="Override narrator CVC string (enclose in quotes)",
    )
    p.add_argument(
        "--narrator-style", default="Narration",
        help="IndicParlerTTS emotion/style token",
    )
    p.add_argument(
        "--narrator-rate", default="",
        help="Optional rate modifier e.g. 'slightly slow'",
    )
    p.add_argument(
        "--anchor-text", default=None,
        help="Hindi sentence used to generate the anchor clip",
    )
    p.add_argument(
        "--f0-threshold", type=float, default=0.12,
        help="F0 mean relative drift threshold (0.12 = 12%%)",
    )
    p.add_argument(
        "--f0-std-threshold", type=float, default=0.30,
        help="F0 std relative drift threshold",
    )
    p.add_argument(
        "--sim-threshold", type=float, default=0.82,
        help="ECAPA cosine similarity floor",
    )
    p.add_argument(
        "--max-retries", type=int, default=2,
        help="Max retry attempts per chunk",
    )
    p.add_argument(
        "--max-chunk-chars", type=int, default=280,
        help="Max characters per TTS chunk",
    )
    p.add_argument(
        "--device", default=None,
        help="Force device: 'cuda' or 'cpu'",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PipelineConfig()
    if args.narrator_cvc:
        cfg.narrator_cvc = args.narrator_cvc
    if args.anchor_text:
        cfg.anchor_text = args.anchor_text
    if args.device:
        cfg.device = args.device

    cfg.narrator_style = args.narrator_style
    cfg.narrator_rate_modifier = args.narrator_rate
    cfg.f0_drift_threshold = args.f0_threshold
    cfg.f0_std_ratio_threshold = args.f0_std_threshold
    cfg.speaker_sim_threshold = args.sim_threshold
    cfg.max_retries = args.max_retries
    cfg.max_chunk_chars = args.max_chunk_chars

    pipeline = NarratorPipeline(cfg)
    pipeline.run(args.story, args.output)


if __name__ == "__main__":
    main()
