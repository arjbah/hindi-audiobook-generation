#!/usr/bin/env python3
"""
English IndicParlerTTS Narrator Pipeline
=========================================
Same F0-anchored architecture as indic_narrator_pipeline.py, but wired for
English input text.  The goal is an A/B diagnostic:

  Hindi pipeline (indic_narrator_pipeline.py)  ← same model, same gating
  English pipeline (this file)                 ← same model, English text

If the pause / tone issues disappear on English, they are likely caused by
Hindi data paucity in IndicParlerTTS training (the model is strong on English
because it inherits a well-trained Parler-TTS base).  If they persist, the
problem is architectural.

Changes vs. the Hindi pipeline
-------------------------------
  1. TextChunker  — drops Devanagari danda rules; uses English punctuation only
  2. PipelineConfig.narrator_cvc — English speaker description ("Gary")
  3. PipelineConfig.anchor_text  — English seed sentence
  4. CLI help strings updated
  (Everything else — F0, ECAPA, CVC patching, retry loop — is identical.)

Usage
-----
  python english_indic_narrator_pipeline.py --story path/to/story.txt --output out/
  python english_indic_narrator_pipeline.py --story story.txt --narrator-cvc "Gary..." --f0-threshold 0.10
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
    # "Gary" is a well-represented English male speaker in Parler-TTS / IndicParler
    # training data.  Uses the same RASMALAI-style CVC format as the Hindi pipeline
    # so the F0 descriptor patching machinery works unchanged.
    narrator_cvc: str = (
        "Gary speaks at a slightly slow pace with a deep, low-pitched voice "
        "and a narrow pitch range in a very close-sounding studio environment. "
        "The audio is of excellent quality with no background noise."
    )
    narrator_style: str = "Narration"       # IndicParlerTTS emotion token
    narrator_rate_modifier: str = ""        # optional: "slightly slow", "moderate", etc.

    # ── Chunking ──────────────────────────────────────────────────────────────
    max_chunk_chars: int = 300              # English chars are wider; 300 ≈ 280 Hindi
    min_chunk_chars: int = 40

    # ── F0 consistency (primary gate) ─────────────────────────────────────────
    f0_drift_threshold: float = 0.12
    f0_std_ratio_threshold: float = 0.30

    # ── Speaker embedding (secondary gate) ────────────────────────────────────
    speaker_sim_threshold: float = 0.82

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 2

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Anchor ────────────────────────────────────────────────────────────────
    # Neutral English sentence at mid-length — matches the role of the Hindi seed.
    anchor_text: str = (
        "In a calm and steady voice, this story begins to unfold."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER  (English only — no Devanagari dandas)
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Splits English text at sentence and paragraph boundaries,
    then greedy-groups into chunks under max_chunk_chars.
    """

    _STANDARD = re.compile(r'(?<=[.!?])\s+(?=[^\d])')
    _PARA = re.compile(r'\n{2,}')

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
# F0 EXTRACTOR  (identical to Hindi pipeline)
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
    F0_FLOOR = 75.0
    F0_CEILING = 300.0
    TIME_STEP = 0.01

    def extract(self, audio: np.ndarray, sr: int) -> Optional[F0Stats]:
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
        mean_drift = abs(chunk.mean - anchor.mean) / max(anchor.mean, 1e-6)
        std_drift = abs(chunk.std - anchor.std) / max(anchor.std, 1e-6)
        return mean_drift, std_drift

    def direction(self, anchor: F0Stats, chunk: F0Stats) -> str:
        return "high" if chunk.mean > anchor.mean else "low"


# ─────────────────────────────────────────────────────────────────────────────
# CVC REGISTRY + PATCHER  (identical to Hindi pipeline)
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
        changed = rec.bump_f0(direction)
        if not changed:
            return rec.cvc

        new_descriptor = F0_DESCRIPTORS[rec.f0_descriptor_idx]
        new_cvc = _PITCH_RE.sub(new_descriptor, rec.cvc, count=1)

        if new_cvc == rec.cvc:
            new_cvc = rec.cvc.rstrip(".") + f", with a {new_descriptor} voice."

        rec.cvc = new_cvc
        return new_cvc

    def compose_caption(self, char_id: str, style: str, rate_modifier: str = "") -> str:
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
# SPEAKER EMBEDDER  (identical to Hindi pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class SpeakerEmbedder:
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
            ratio = orig_sr / target_sr
            new_len = int(len(audio) / ratio)
            indices = (np.arange(new_len) * ratio).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            return audio[indices]


# ─────────────────────────────────────────────────────────────────────────────
# INDICPARLER TTS WRAPPER  (same model ID as Hindi pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class IndicParlerTTS:
    """
    Same ai4bharat/indic-parler-tts model as the Hindi pipeline.
    English text is fed directly — the model handles it via its Parler-TTS base.
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
# MAIN PIPELINE  (identical logic to Hindi pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class NarratorPipeline:
    NARRATOR_ID = "narrator"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.chunker = TextChunker(cfg)
        self.tts = IndicParlerTTS(cfg.device)
        self.embedder = SpeakerEmbedder(cfg.device)
        self.f0_extractor = F0Extractor()
        self.registry = CVCRegistry()

    def _build_anchor(self) -> None:
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

    def _validate(self, audio: np.ndarray) -> tuple[bool, str, dict]:
        rec = self.registry.get(self.NARRATOR_ID)
        diag: dict = {}

        chunk_f0 = self.f0_extractor.extract(audio, self.tts.sr)
        if chunk_f0 is None:
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

        chunk_embed = self.embedder.embed(audio, self.tts.sr)
        sim = self.embedder.cosine(chunk_embed, rec.anchor_embed)
        spk_ok = sim >= self.cfg.speaker_sim_threshold
        diag.update({
            "ecapa_sim": round(sim, 4),
            "ecapa_passed": spk_ok,
        })

        return spk_ok, direction, diag

    def run(self, story_path: str, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        text = Path(story_path).read_text(encoding="utf-8")
        chunks = self.chunker.chunk(text)
        n = len(chunks)
        print(f"\n[Pipeline] Story loaded: {len(text)} chars → {n} chunks")

        self._build_anchor()

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

        final_audio = np.concatenate(segments)
        audio_out = output_path / "story_narration.wav"
        sf.write(str(audio_out), final_audio, self.tts.sr)
        print(f"\n[Output] Audio → {audio_out}  ({len(final_audio)/self.tts.sr:.1f}s)")

        summary = {
            "story": str(story_path),
            "language": "english",
            "model": IndicParlerTTS.MODEL_ID,
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

        print("\n" + "─" * 60)
        print(f"  Language:      English (IndicParlerTTS)")
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
        description="English IndicParlerTTS F0-anchored narrator pipeline "
                    "(diagnostic counterpart to indic_narrator_pipeline.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--story", required=True,
        help="Path to story text file (UTF-8, English)",
    )
    p.add_argument(
        "--output", default="output_english/",
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
        help="English sentence used to generate the anchor clip",
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
        "--max-chunk-chars", type=int, default=300,
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
