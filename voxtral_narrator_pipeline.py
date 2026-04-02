#!/usr/bin/env python3
"""
Voxtral-4B-TTS Narrator Pipeline
==================================
F0-anchored single-narrator pipeline using mistralai/Voxtral-4B-TTS-2603
instead of IndicParlerTTS.  Supports both Hindi and English so you can run
the same diagnostic as the IndicParler A/B:

  English: python voxtral_narrator_pipeline.py --story story_en.txt --language english
  Hindi:   python voxtral_narrator_pipeline.py --story story_hi.txt --language hindi

Architecture
------------
  Voxtral-4B-TTS is served via a vLLM Omni server (HTTP).  This script is a
  thin client that POSTs to the /v1/audio/speech endpoint and pipes audio
  through the same F0 + ECAPA consistency gating as indic_narrator_pipeline.py.

  The CVC patching mechanism is intentionally kept even though Voxtral uses
  preset voice names rather than free-text CVCs.  Here the "patch" step simply
  cycles through a ranked list of alternative voices when F0 drift is detected,
  which is the closest analogue for a preset-voice system.

Prerequisites
-------------
  1. Start the vLLM Omni server (requires ~16 GB GPU):
       vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
     The server listens on http://localhost:8000 by default.

  2. Install client deps (already in your env if you have the IndicParler env):
       pip install httpx soundfile praat-parselmouth speechbrain torch

Usage
-----
  python voxtral_narrator_pipeline.py --story story.txt --language english
  python voxtral_narrator_pipeline.py --story C:/Users/prana/source/repos/hindi-audiobook-generation/transcripts/long_story_transcript.txt --language hindi --voice casual_male
  python voxtral_narrator_pipeline.py --story story.txt --server http://my-gpu-box:8000
"""

from __future__ import annotations

import argparse
import io
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# VOICE FALLBACK LADDER
# ─────────────────────────────────────────────────────────────────────────────
# Voxtral preset voices, roughly ordered low-pitch → high-pitch.
# When F0 drifts high (chunk too high), step down the ladder; when low, step up.
# "casual_male" is the default starting point for narration.
VOICE_LADDER: list[str] = [
    "deep_male",
    "casual_male",
    "narration_male",
    "calm_male",
    "casual_female",
    "narration_female",
    "calm_female",
]

_DEFAULT_VOICE_IDX = 1  # "casual_male"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Language ──────────────────────────────────────────────────────────────
    # Voxtral supports: english, french, spanish, german, italian,
    #                   portuguese, dutch, arabic, hindi
    language: str = "english"

    # ── Voice ─────────────────────────────────────────────────────────────────
    voice: str = "casual_male"           # starting preset voice
    voice_ladder_idx: int = _DEFAULT_VOICE_IDX

    # ── Server ────────────────────────────────────────────────────────────────
    server_url: str = "http://localhost:8000"
    model_id: str = "mistralai/Voxtral-4B-TTS-2603"
    request_timeout: float = 120.0       # seconds per chunk request

    # ── Chunking ──────────────────────────────────────────────────────────────
    max_chunk_chars: int = 300
    min_chunk_chars: int = 40

    # ── F0 consistency (primary gate) ─────────────────────────────────────────
    f0_drift_threshold: float = 0.12
    f0_std_ratio_threshold: float = 0.30

    # ── Speaker embedding (secondary gate) ────────────────────────────────────
    speaker_sim_threshold: float = 0.82

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 2

    # ── Device (for ECAPA-TDNN only — Voxtral runs on the vLLM server) ────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Anchor ────────────────────────────────────────────────────────────────
    # Override per language with --anchor-text, or use the defaults below.
    anchor_text_english: str = (
        "In a calm and steady voice, this story begins to unfold."
    )
    anchor_text_hindi: str = "एक शांत और स्थिर आवाज़ में, यह कहानी शुरू होती है।"

    @property
    def anchor_text(self) -> str:
        return (
            self.anchor_text_hindi
            if self.language.lower() == "hindi"
            else self.anchor_text_english
        )

    def bump_voice(self, direction: str) -> bool:
        """Shift voice ladder one step toward corrective direction."""
        if direction == "high" and self.voice_ladder_idx > 0:
            self.voice_ladder_idx -= 1
            self.voice = VOICE_LADDER[self.voice_ladder_idx]
            return True
        if direction == "low" and self.voice_ladder_idx < len(VOICE_LADDER) - 1:
            self.voice_ladder_idx += 1
            self.voice = VOICE_LADDER[self.voice_ladder_idx]
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHUNKER
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Language-aware sentence splitter.
    Hindi mode adds Devanagari danda rules; English mode uses standard punct.
    """

    _DANDA = re.compile(r'(?<=[।॥])\s+')
    _STANDARD = re.compile(r'(?<=[.!?])\s+(?=[^\d])')
    _PARA = re.compile(r'\n{2,}')

    def __init__(self, cfg: PipelineConfig):
        self.max_chars = cfg.max_chunk_chars
        self.min_chars = cfg.min_chunk_chars
        self.hindi = cfg.language.lower() == "hindi"

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        paragraphs = self._PARA.split(text)
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if self.hindi:
                parts = self._DANDA.split(para)
                for part in parts:
                    subparts = self._STANDARD.split(part)
                    sentences.extend(s.strip() for s in subparts if s.strip())
            else:
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
# F0 EXTRACTOR  (unchanged from indic_narrator_pipeline.py)
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
# SPEAKER EMBEDDER  (unchanged from indic_narrator_pipeline.py)
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
# VOXTRAL TTS CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class VoxtralTTS:
    """
    HTTP client for the vLLM Omni /v1/audio/speech endpoint.

    The server must be running before calling synthesize():
      vllm serve mistralai/Voxtral-4B-TTS-2603 --omni

    Output sample rate is 24 000 Hz (Voxtral default).
    """
    SAMPLE_RATE = 24000

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.endpoint = f"{cfg.server_url.rstrip('/')}/v1/audio/speech"
        self.sr = self.SAMPLE_RATE

        try:
            import httpx
            self._client = httpx.Client(timeout=cfg.request_timeout)
        except ImportError:
            raise ImportError(
                "httpx not installed. Run: pip install httpx"
            )

        # Verify the server is reachable
        print(f"[VoxtralTTS] Connecting to {self.endpoint} ...")
        try:
            resp = self._client.get(f"{cfg.server_url.rstrip('/')}/health")
            if resp.status_code == 200:
                print(f"[VoxtralTTS] Server healthy.  Model: {cfg.model_id}")
            else:
                print(f"[VoxtralTTS] Warning: /health returned {resp.status_code}")
        except Exception as e:
            print(
                f"[VoxtralTTS] Warning: could not reach server at {cfg.server_url}\n"
                f"  ({e})\n"
                f"  Make sure to run: vllm serve {cfg.model_id} --omni"
            )

    def synthesize(self, text: str, voice: str) -> np.ndarray:
        """
        POST text to vLLM Omni and return float32 mono waveform at self.sr.
        """
        import httpx
        payload = {
            "input": text,
            "model": self.cfg.model_id,
            "response_format": "wav",
            "voice": voice,
        }
        try:
            resp = self._client.post(self.endpoint, json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Voxtral server returned {e.response.status_code}: {e.response.text}"
            ) from e

        audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono if needed

        # Resample if the server returns something other than 24 kHz
        if sr != self.SAMPLE_RATE:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
            except ImportError:
                pass  # keep as-is; F0 extractor will adapt

        return audio.astype(np.float32)

    def close(self):
        self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class NarratorPipeline:
    """
    Same F0 + ECAPA gating loop as indic_narrator_pipeline.py.
    Voice drift correction patches the voice ladder index instead of the CVC
    F0 descriptor string — structurally analogous, different knob.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.chunker = TextChunker(cfg)
        self.tts = VoxtralTTS(cfg)
        self.embedder = SpeakerEmbedder(cfg.device)
        self.f0_extractor = F0Extractor()
        self.anchor_f0: Optional[F0Stats] = None
        self.anchor_embed: Optional[np.ndarray] = None

    # ── Anchor ────────────────────────────────────────────────────────────────

    def _build_anchor(self) -> None:
        print(f"\n[Anchor] Synthesizing anchor clip (voice='{self.cfg.voice}')...")
        audio = self.tts.synthesize(self.cfg.anchor_text, self.cfg.voice)

        f0_stats = self.f0_extractor.extract(audio, self.tts.sr)
        if f0_stats is None:
            raise RuntimeError(
                "F0 extraction failed on anchor clip. "
                "Try a different voice or a longer anchor_text."
            )
        self.anchor_f0 = f0_stats
        self.anchor_embed = self.embedder.embed(audio, self.tts.sr)

        print(
            f"[Anchor] F0 mean={f0_stats.mean:.1f} Hz  "
            f"std={f0_stats.std:.1f} Hz  "
            f"voiced={f0_stats.voiced_fraction:.2%}"
        )

    # ── Chunk validation ──────────────────────────────────────────────────────

    def _validate(self, audio: np.ndarray) -> tuple[bool, str, dict]:
        diag: dict = {}

        chunk_f0 = self.f0_extractor.extract(audio, self.tts.sr)
        if chunk_f0 is None:
            diag["f0"] = "extraction_failed"
            return False, "low", diag

        mean_drift, std_drift = self.f0_extractor.drift(self.anchor_f0, chunk_f0)
        direction = self.f0_extractor.direction(self.anchor_f0, chunk_f0)

        f0_ok = (
            mean_drift <= self.cfg.f0_drift_threshold
            and std_drift <= self.cfg.f0_std_ratio_threshold
        )
        diag.update({
            "f0_anchor_mean": self.anchor_f0.mean,
            "f0_anchor_std": self.anchor_f0.std,
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
        sim = self.embedder.cosine(chunk_embed, self.anchor_embed)
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

        text = Path(story_path).read_text(encoding="utf-8")
        chunks = self.chunker.chunk(text)
        n = len(chunks)
        print(
            f"\n[Pipeline] Story loaded: {len(text)} chars → {n} chunks  "
            f"(language={self.cfg.language})"
        )

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
                current_voice = self.cfg.voice
                audio = self.tts.synthesize(chunk, current_voice)
                last_audio = audio

                ok, direction, diag = self._validate(audio)
                log_entry = {
                    "chunk_idx": idx,
                    "attempt": attempt,
                    "voice": current_voice,
                    "passed": ok,
                    **diag,
                }
                run_log.append(log_entry)

                f0_str = f"F0={diag.get('f0_chunk_mean','?')} Hz drift={diag.get('f0_mean_drift','?'):.3f}"
                sim_str = f"ECAPA={diag.get('ecapa_sim','skipped')}"
                marker = "✓" if ok else "✗"
                print(f"  [{marker}] attempt={attempt}  voice='{current_voice}'  {f0_str}  {sim_str}")

                if ok:
                    passed = True
                    break

                if attempt < self.cfg.max_retries:
                    changed = self.cfg.bump_voice(direction)
                    total_retries += 1
                    if changed:
                        print(
                            f"  [Patch] F0 drift direction='{direction}' → "
                            f"new voice: '{self.cfg.voice}'"
                        )
                    else:
                        print(
                            f"  [Patch] F0 drift direction='{direction}' but "
                            f"already at voice ladder limit ('{self.cfg.voice}')"
                        )

            if not passed:
                flagged.append(idx)
                print(f"  [Flag] Chunk {idx} failed after all retries — emitting last audio.")

            segments.append(last_audio)

        self.tts.close()

        final_audio = np.concatenate(segments)
        audio_out = output_path / "story_narration.wav"
        sf.write(str(audio_out), final_audio, self.tts.sr)
        print(f"\n[Output] Audio → {audio_out}  ({len(final_audio)/self.tts.sr:.1f}s)")

        summary = {
            "story": str(story_path),
            "language": self.cfg.language,
            "model": self.cfg.model_id,
            "voice_final": self.cfg.voice,
            "anchor_f0": self.anchor_f0.to_dict(),
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
        print(f"  Model:         {self.cfg.model_id}")
        print(f"  Language:      {self.cfg.language}")
        print(f"  Chunks:        {n}")
        print(f"  Total retries: {total_retries}")
        print(f"  Flagged:       {len(flagged)}" +
              (f"  {flagged}" if flagged else "  (none)"))
        print(f"  Final voice:   {self.cfg.voice}")
        print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Voxtral-4B-TTS F0-anchored narrator pipeline (English & Hindi)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--story", required=True,
        help="Path to story text file (UTF-8)",
    )
    p.add_argument(
        "--output", default="output_voxtral/",
        help="Directory for audio and log outputs",
    )
    p.add_argument(
        "--language", default="english",
        choices=["english", "hindi", "french", "spanish", "german",
                 "italian", "portuguese", "dutch", "arabic"],
        help="Story language (passed to Voxtral for pronunciation)",
    )
    p.add_argument(
        "--voice", default="casual_male",
        help=f"Starting Voxtral preset voice. Available ladder: {VOICE_LADDER}",
    )
    p.add_argument(
        "--server", default="http://localhost:8000",
        help="Base URL of the running vLLM Omni server",
    )
    p.add_argument(
        "--anchor-text", default=None,
        help="Sentence used to generate the anchor clip (overrides language default)",
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
        help="Force device for ECAPA: 'cuda' or 'cpu'",
    )
    p.add_argument(
        "--timeout", type=float, default=120.0,
        help="Per-chunk HTTP request timeout in seconds",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PipelineConfig()
    cfg.language = args.language
    cfg.voice = args.voice
    # Set ladder index to match requested voice (fall back to default if not found)
    if cfg.voice in VOICE_LADDER:
        cfg.voice_ladder_idx = VOICE_LADDER.index(cfg.voice)
    cfg.server_url = args.server
    cfg.request_timeout = args.timeout
    cfg.f0_drift_threshold = args.f0_threshold
    cfg.f0_std_ratio_threshold = args.f0_std_threshold
    cfg.speaker_sim_threshold = args.sim_threshold
    cfg.max_retries = args.max_retries
    cfg.max_chunk_chars = args.max_chunk_chars
    if args.device:
        cfg.device = args.device
    if args.anchor_text:
        if cfg.language == "hindi":
            cfg.anchor_text_hindi = args.anchor_text
        else:
            cfg.anchor_text_english = args.anchor_text

    pipeline = NarratorPipeline(cfg)
    pipeline.run(args.story, args.output)


if __name__ == "__main__":
    main()
