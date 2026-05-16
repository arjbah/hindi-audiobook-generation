#!/usr/bin/env python3
"""
IndicF5 REA-TTS Inference Pipeline
==================================
Hindi clone of the REA-TTS framework
(https://happylittlecat2333.github.io/interspeech2024-RAG/) using AI4Bharat IndicF5 as the speech synthesizer.

Reference manifest
------------------
A JSON file pointing at reference clips and their transcripts.  Tags are
optional and only used by the stub encoders for end-to-end testing before
CLAP:

  {
    "clips": [
      {"id": "happy_01",   "path": "refs/happy_01.wav",   "text": "...", "tags": ["happy"]},
      {"id": "sad_01",     "path": "refs/sad_01.wav",     "text": "...", "tags": ["sad"]},
      {"id": "neutral_01", "path": "refs/neutral_01.wav", "text": "...", "tags": ["neutral"]}
    ]
  }

Usage
-----
  python indicf5_rea_pipeline.py `
    --story transcripts/long_story_transcript.txt `
    --refs reference_manifest.json `
    --output output/IndicF5_REA/
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import re
import sys
import traceback
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

# ── pyarrow-on-Windows access-violation workaround ────────────────────────────
# IndicF5's model.py drags in f5_tts.model.__init__ → trainer → dataset →
# `datasets` → `pyarrow.dataset`, and pyarrow segfaults on import under
# Windows here.  Inference doesn't actually use any of it, so we shim
# `datasets` with a dummy module *before* HF resolves the IndicF5 code.
def _install_datasets_stub() -> None:
    if "datasets" in sys.modules:
        return
    class _Stub:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return self
        def __getattr__(self, name): return _Stub()
    fake = types.ModuleType("datasets")
    for name in (
        "Dataset", "IterableDataset", "DatasetDict", "IterableDatasetDict",
        "load_dataset", "load_from_disk", "concatenate_datasets",
        "Features", "Value", "Audio", "Sequence", "ClassLabel", "Array2D",
    ):
        setattr(fake, name, _Stub)
    sys.modules["datasets"] = fake
_install_datasets_stub()

# ── meta-device init workaround ──────────────────────────────────────────────
# transformers/accelerate install a `torch.device("meta")` context during
# from_pretrained so weights can be loaded lazily.  IndicF5's __init__ eagerly
# builds a Vocos vocoder + torchaudio MelSpectrogram inside that context, and
# torchaudio.melscale_fbanks crashes mixing meta + cpu tensors.  Force
# `init_empty_weights` to a no-op context manager.
def _disable_init_empty_weights() -> None:
    import contextlib
    fake = lambda *a, **k: contextlib.nullcontext()
    for mod_name in ("accelerate.big_modeling", "accelerate",
                     "transformers.modeling_utils"):
        try:
            mod = __import__(mod_name, fromlist=["init_empty_weights"])
            if hasattr(mod, "init_empty_weights"):
                mod.init_empty_weights = fake
        except ImportError:
            pass
_disable_init_empty_weights()

# Belt-and-suspenders: rewrite any `torch.device("meta")` context to "cpu" at
# the DeviceContext level.  Catches meta init paths the function patch above
# misses (transformers/accelerate sometimes install meta contexts directly).
def _redirect_meta_device_to_cpu() -> None:
    import torch.utils._device as _dev_mod
    _orig_init = _dev_mod.DeviceContext.__init__
    def _patched_init(self, device):
        dev = torch.device(device)
        if dev.type == "meta":
            dev = torch.device("cpu")
        _orig_init(self, dev)
    _dev_mod.DeviceContext.__init__ = _patched_init
_redirect_meta_device_to_cpu()

# ── f5_tts load_model API drift ──────────────────────────────────────────────
# IndicF5's model.py was written against an older f5_tts where load_model
# accepted no ckpt_path (caller loaded weights afterward).  Current f5_tts
# requires ckpt_path positionally.  Patch load_model so a missing/empty
# ckpt_path just builds the architecture with random weights;
# transformers.from_pretrained loads the real IndicF5 weights right after
# __init__ returns.
def _patch_f5tts_load_model() -> None:
    try:
        import f5_tts.infer.utils_infer as ui
    except ImportError:
        return
    _orig = ui.load_model

    def _patched(model_cls, model_cfg, ckpt_path=None,
                 mel_spec_type=None, vocab_file="", ode_method=None,
                 use_ema=True, device=None):
        if mel_spec_type is None:
            mel_spec_type = ui.mel_spec_type
        if ode_method is None:
            ode_method = ui.ode_method
        if device is None:
            device = ui.device
        if ckpt_path:
            return _orig(model_cls, model_cfg, ckpt_path, mel_spec_type,
                         vocab_file, ode_method, use_ema, device)
        # No checkpoint path: build the architecture and return uninitialized.
        from importlib.resources import files
        from f5_tts.model import CFM
        from f5_tts.model.utils import get_tokenizer
        if not vocab_file:
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
        vocab_char_map, vocab_size = get_tokenizer(vocab_file, "custom")
        model = CFM(
            transformer=model_cls(
                **model_cfg,
                text_num_embeds=vocab_size,
                mel_dim=ui.n_mel_channels,
            ),
            mel_spec_kwargs=dict(
                n_fft=ui.n_fft,
                hop_length=ui.hop_length,
                win_length=ui.win_length,
                n_mel_channels=ui.n_mel_channels,
                target_sample_rate=ui.target_sample_rate,
                mel_spec_type=mel_spec_type,
            ),
            odeint_kwargs=dict(method=ode_method),
            vocab_char_map=vocab_char_map,
        ).to(device)
        return model

    ui.load_model = _patched
_patch_f5tts_load_model()

# ── post-init cleanup that assumes meta-device init ──────────────────────────
# transformers calls _move_missing_keys_from_meta_to_device after loading
# weights to copy meta placeholders to the real device.  Since we disabled
# meta init, there are no meta tensors to move.  The function also accesses
# self.all_tied_weights_keys, a property missing on IndicF5's custom class.
def _patch_finalize_meta_move() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        return
    PreTrainedModel._move_missing_keys_from_meta_to_device = (
        lambda self, *a, **k: None
    )
    # IndicF5's INF5Model doesn't define _tied_weights_keys / all_tied_weights_keys.
    # No-op tie_weights and provide an empty mapping for any other internal access.
    PreTrainedModel.tie_weights = lambda self, *a, **k: None
    PreTrainedModel.all_tied_weights_keys = {}
_patch_finalize_meta_move()

import numpy as np
import soundfile as sf
import torch

faulthandler.enable()
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Chunking ──────────────────────────────────────────────────────────────
    max_chunk_chars: int = 280
    min_chunk_chars: int = 40

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # top_k > 1 enables the REA-TTS "concat reference" trick — picks the best k
    # clips and concatenates them as one long reference for IndicF5.
    top_k: int = 1
    max_concat_seconds: float = 12.0   # safety cap on concatenated ref length

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Output ────────────────────────────────────────────────────────────────
    output_sr: int = 24000             # IndicF5 native sample rate



class TextChunker:
    _DANDA = re.compile(r'(?<=[।॥])\s+')
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
            for part in self._DANDA.split(para):
                for sub in self._STANDARD.split(part):
                    if sub.strip():
                        sentences.append(sub.strip())

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




class TextEmotionEncoder(Protocol):
    """Encodes a text chunk into the shared emotion embedding space."""
    dim: int
    def encode(self, text: str) -> np.ndarray: ...


class SpeechEmotionEncoder(Protocol):
    """Encodes a reference audio clip into the shared emotion embedding space."""
    dim: int
    def encode(self, audio: np.ndarray, sr: int) -> np.ndarray: ...




EMOTION_LABELS: list[str] = [
    "happy", "sad", "angry", "fearful",
    "surprised", "disgusted", "neutral",
]


def _label_to_onehot(label: str) -> np.ndarray:
    vec = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    if label in EMOTION_LABELS:
        vec[EMOTION_LABELS.index(label)] = 1.0
    else:
        vec[EMOTION_LABELS.index("neutral")] = 1.0
    return vec


class StubTextEmotionEncoder:
    """
    Runs a small multilingual sentiment classifier and maps its output to one
    of `EMOTION_LABELS`.  Good enough to exercise retrieval during development.
    """
    MODEL_ID = "tabularisai/multilingual-sentiment-analysis"
    dim = len(EMOTION_LABELS)

    _SENT_TO_EMOTION = {
        "Very Positive": "happy",
        "Positive": "happy",
        "Neutral": "neutral",
        "Negative": "sad",
        "Very Negative": "angry",
    }

    def __init__(self, device: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        label_id = int(torch.argmax(logits).item())
        sentiment = self.model.config.id2label[label_id]
        return _label_to_onehot(self._SENT_TO_EMOTION.get(sentiment, "neutral"))


class StubSpeechEmotionEncoder:
    """
    Looks up the clip's manifest tags and emits the corresponding one-hot
    vector.  No audio is read here — this is purely a placeholder so retrieval
    is sensible before a real audio encoder lands.
    """
    dim = len(EMOTION_LABELS)

    def encode(self, audio: np.ndarray, sr: int, tags: Sequence[str] = ()) -> np.ndarray:
        for t in tags:
            if t in EMOTION_LABELS:
                return _label_to_onehot(t)
        return _label_to_onehot("neutral")


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReferenceClip:
    clip_id: str
    path: Path
    text: str                  # transcript of the ref clip — IndicF5 requires this
    tags: list[str] = field(default_factory=list)
    speech_emb: Optional[np.ndarray] = None   # populated by encode_all()
    _audio: Optional[np.ndarray] = None       # lazy-loaded cache
    _sr: Optional[int] = None

    def load_audio(self) -> tuple[np.ndarray, int]:
        if self._audio is None:
            audio, sr = sf.read(str(self.path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            self._audio, self._sr = audio, sr
        return self._audio, self._sr


class ReferenceDatabase:
    """
    In-memory store of reference clips + their precomputed speech embeddings.

    Retrieval is plain cosine top-k.  Swap in FAISS later if the corpus grows
    past a few thousand clips; the public surface (`add`, `retrieve`) stays the
    same.
    """

    def __init__(self):
        self._clips: list[ReferenceClip] = []
        self._matrix: Optional[np.ndarray] = None   # (N, dim) L2-normalized

    def __len__(self) -> int:
        return len(self._clips)

    def add(self, clip: ReferenceClip) -> None:
        self._clips.append(clip)
        self._matrix = None  # invalidate

    @classmethod
    def from_manifest(cls, manifest_path: str) -> "ReferenceDatabase":
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        root = Path(manifest_path).parent
        db = cls()
        for entry in manifest["clips"]:
            db.add(ReferenceClip(
                clip_id=entry["id"],
                path=(root / entry["path"]).resolve(),
                text=entry["text"],
                tags=list(entry.get("tags", [])),
            ))
        return db

    def encode_all(self, encoder: SpeechEmotionEncoder) -> None:
        """Populate `speech_emb` for every clip and build the lookup matrix."""
        embs = []
        for clip in self._clips:
            audio, sr = clip.load_audio()
            # Stub encoder accepts tags as an extra arg; real CLAP encoders won't,
            # so this is the only place we special-case the stub.
            if isinstance(encoder, StubSpeechEmotionEncoder):
                emb = encoder.encode(audio, sr, tags=clip.tags)
            else:
                emb = encoder.encode(audio, sr)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            clip.speech_emb = emb
            embs.append(emb)
        self._matrix = np.stack(embs, axis=0)

    def retrieve(self, query_emb: np.ndarray, top_k: int = 1) -> list[ReferenceClip]:
        if self._matrix is None:
            raise RuntimeError("Call encode_all(...) before retrieve(...)")
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        sims = self._matrix @ q
        idx = np.argsort(-sims)[:top_k]
        return [self._clips[int(i)] for i in idx]


# ─────────────────────────────────────────────────────────────────────────────
# INDICF5 SYNTHESIZER
# ─────────────────────────────────────────────────────────────────────────────

class IndicF5Synthesizer:
    """
    Thin wrapper around ai4bharat/IndicF5.

      audio = model(text, ref_audio_path=..., ref_text=...)

    Returns int16-scaled float at 24 kHz.  We rescale to [-1, 1] float32 here
    so downstream code never has to think about it.
    """
    MODEL_ID = "ai4bharat/IndicF5"
    SR = 24000

    def __init__(self, device: str):
        from transformers import AutoModel
        print(f"[IndicF5] Loading {self.MODEL_ID} on {device} ...", flush=True)
        try:
            # low_cpu_mem_usage=False disables transformers' "init on meta
            # device" trick. IndicF5's __init__ eagerly constructs a Vocos
            # vocoder + torchaudio MelSpectrogram, which fails on meta because
            # melscale_fbanks mixes meta + cpu tensors.
            model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
            print("[IndicF5] from_pretrained OK; moving to device ...", flush=True)
            self.model = model.to(device).eval()
        except Exception:
            print("[IndicF5] ERROR during model load:", flush=True)
            traceback.print_exc()
            sys.exit(1)
        self.device = device
        print("[IndicF5] Ready.", flush=True)

    @torch.no_grad()
    def synthesize(self, text: str, ref_audio_path: str, ref_text: str) -> np.ndarray:
        out = self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)
        out = np.asarray(out, dtype=np.float32)
        # IndicF5 returns int16-range floats; normalize.
        peak = float(np.max(np.abs(out)))
        if peak > 1.5:
            out = out / 32768.0
        return out


# ─────────────────────────────────────────────────────────────────────────────
# CONCAT-REFERENCE HELPER  (REA-TTS top-k trick)
# ─────────────────────────────────────────────────────────────────────────────

def build_concat_reference(
    clips: Sequence[ReferenceClip],
    out_path: Path,
    max_seconds: float,
) -> tuple[str, str]:
    """
    Concatenates top-k retrieved clips (with a brief silence between them) and
    writes them to `out_path`.  Returns (audio_path_str, joined_transcript).

    IndicF5 takes a single reference path + transcript per call, so when we
    want to give it multiple references we splice them into one file here.
    """
    silence_sec = 0.2
    sr_target = IndicF5Synthesizer.SR
    audio_parts: list[np.ndarray] = []
    text_parts: list[str] = []
    total = 0.0

    for clip in clips:
        audio, sr = clip.load_audio()
        if sr != sr_target:
            # Lightweight resample — avoids a librosa dep if possible.
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_target)
            except ImportError:
                ratio = sr / sr_target
                idx = (np.arange(int(len(audio) / ratio)) * ratio).astype(int)
                audio = audio[np.clip(idx, 0, len(audio) - 1)]
        dur = len(audio) / sr_target
        if total + dur > max_seconds and audio_parts:
            break
        if audio_parts:
            audio_parts.append(np.zeros(int(silence_sec * sr_target), dtype=np.float32))
        audio_parts.append(audio.astype(np.float32))
        text_parts.append(clip.text.strip())
        total += dur + silence_sec

    concat = np.concatenate(audio_parts)
    sf.write(str(out_path), concat, sr_target)
    return str(out_path), " ".join(text_parts)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class REATTSPipeline:
    def __init__(
        self,
        cfg: PipelineConfig,
        text_encoder: TextEmotionEncoder,
        speech_encoder: SpeechEmotionEncoder,
    ):
        self.cfg = cfg
        self.chunker = TextChunker(cfg)
        self.tts = IndicF5Synthesizer(cfg.device)
        self.text_encoder = text_encoder
        self.speech_encoder = speech_encoder
        self.db: Optional[ReferenceDatabase] = None

    def load_references(self, manifest_path: str) -> None:
        print(f"\n[DB] Loading reference manifest: {manifest_path}")
        self.db = ReferenceDatabase.from_manifest(manifest_path)
        print(f"[DB] {len(self.db)} clips loaded; encoding with speech encoder ...")
        self.db.encode_all(self.speech_encoder)
        print("[DB] Indexed.")

    def _retrieve_for_chunk(self, chunk_text: str) -> list[ReferenceClip]:
        if self.db is None or len(self.db) == 0:
            raise RuntimeError("No reference database loaded.")
        q = self.text_encoder.encode(chunk_text)
        return self.db.retrieve(q, top_k=self.cfg.top_k)

    def run(self, story_path: str, output_dir: str) -> None:
        if self.db is None:
            raise RuntimeError("Call load_references(...) before run(...)")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scratch = out / "_scratch"
        scratch.mkdir(exist_ok=True)

        text = Path(story_path).read_text(encoding="utf-8")
        chunks = self.chunker.chunk(text)
        n = len(chunks)
        print(f"\n[Pipeline] Story loaded: {len(text)} chars → {n} chunks")

        segments: list[np.ndarray] = []
        run_log: list[dict] = []

        for idx, chunk in enumerate(chunks):
            print(f"\n[{idx+1:>3}/{n}] {chunk[:70]}{'...' if len(chunk) > 70 else ''}")
            retrieved = self._retrieve_for_chunk(chunk)
            ref_ids = [c.clip_id for c in retrieved]
            print(f"  [RAG] top-{self.cfg.top_k}: {ref_ids}")

            if len(retrieved) == 1:
                ref_path = str(retrieved[0].path)
                ref_text = retrieved[0].text
            else:
                ref_path, ref_text = build_concat_reference(
                    retrieved,
                    out_path=scratch / f"ref_chunk_{idx:04d}.wav",
                    max_seconds=self.cfg.max_concat_seconds,
                )

            audio = self.tts.synthesize(chunk, ref_path, ref_text)
            segments.append(audio)
            run_log.append({
                "chunk_idx": idx,
                "chunk_preview": chunk[:80],
                "retrieved_ids": ref_ids,
                "ref_audio": ref_path,
                "ref_text": ref_text,
                "audio_samples": int(len(audio)),
            })

        final = np.concatenate(segments)
        audio_out = out / "story_narration.wav"
        sf.write(str(audio_out), final, self.cfg.output_sr)
        print(f"\n[Output] Audio → {audio_out}  ({len(final)/self.cfg.output_sr:.1f}s)")

        log_out = out / "pipeline_log.json"
        log_out.write_text(
            json.dumps({
                "story": str(story_path),
                "total_chunks": n,
                "top_k": self.cfg.top_k,
                "chunks": run_log,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Output] Log   → {log_out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="IndicF5 REA-TTS inference pipeline (Hindi)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--story", required=True, help="UTF-8 Hindi story text file")
    p.add_argument("--refs", required=True, help="Reference manifest JSON")
    p.add_argument("--output", default="output/IndicF5_REA/")
    p.add_argument("--top-k", type=int, default=1,
                   help="Retrieve & concat top-k reference clips per chunk")
    p.add_argument("--max-chunk-chars", type=int, default=280)
    p.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PipelineConfig()
    cfg.top_k = args.top_k
    cfg.max_chunk_chars = args.max_chunk_chars
    if args.device:
        cfg.device = args.device

    # ── Plug-in points for the CLAP team ──────────────────────────────────────
    # Replace these two lines with the real CLAP text / audio encoders once
    # they're ready.  Anything matching the TextEmotionEncoder /
    # SpeechEmotionEncoder protocols will work.
    text_encoder: TextEmotionEncoder = StubTextEmotionEncoder(device=cfg.device)
    speech_encoder: SpeechEmotionEncoder = StubSpeechEmotionEncoder()

    pipeline = REATTSPipeline(cfg, text_encoder, speech_encoder)
    pipeline.load_references(args.refs)
    pipeline.run(args.story, args.output)


if __name__ == "__main__":
    main()
