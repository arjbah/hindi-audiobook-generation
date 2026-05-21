import argparse
import os
import re
import sys
from collections import deque

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add relevant paths
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "clap_encoders"))
sys.path.append(os.path.join(os.getcwd(), "IndicF5"))

from clap_model import CLAPModel
from train_clap import HTSATConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel


def load_indicf5_model(repo_id, device):
    """Load IndicF5 and restore safetensors without Transformers gamma/beta key rewriting."""
    # Add local model path to sys.path to allow manual import
    local_model_path = os.path.join(os.getcwd(), "indicf5_local")
    if local_model_path not in sys.path:
        sys.path.insert(0, local_model_path)
    
    from model import INF5Model, INF5Config
    
    print(f"Loading modified IndicF5 from local path: {local_model_path}")
    config = INF5Config.from_pretrained(local_model_path)
    model = INF5Model(config).to(device)

    weights_path = hf_hub_download("ai4bharat/IndicF5", filename="model.safetensors")
    state_dict = load_file(weights_path)
    
    # Strip _orig_mod. from keys (happens if model was saved while compiled)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
        
    load_result = model.load_state_dict(new_state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"Warning: IndicF5 structural mismatches after stripping _orig_mod: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
        # If there are still many missing keys, we might need to be more aggressive or fail
        if len(load_result.missing_keys) > 100:
             raise RuntimeError(f"Too many missing keys: {len(load_result.missing_keys)}")

    print("IndicF5 safetensors reloaded cleanly (stripped _orig_mod).")
    return model


class VariableReferenceIndicAudiobookGenerator:
    """
    Audiobook generator that keeps a stable global reference pool but varies the
    selected reference clip by sentence.

    The current reference library has no real speaker IDs, so this approximates
    narrator consistency by staying inside the top-K clips for the full text and
    adding continuity scoring between neighboring selections.
    """

    def __init__(self, device="cpu", pool_size=80):
        self.device = device
        self.pool_size = pool_size

        print("Loading CLAP Model...")
        config = HTSATConfig()
        self.clap = CLAPModel(config).to(device)
        checkpoint = torch.load("clap_model_epoch_120.pt", map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.clap.load_state_dict(state_dict, strict=False)
        self.clap.eval()

        print("Loading Reference Library...")
        lib_path = "reference_library/voice_library.pt"
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found at {lib_path}. Run build_reference_library.py first.")
        self.library = torch.load(lib_path)

        self.lib_embs = torch.tensor([item["embedding"] for item in self.library], device=device)
        self.lib_embs = F.normalize(self.lib_embs, p=2, dim=-1)

        self.reference_stats = self._build_reference_stats()
        self.speaker_proxy_cache = {}

        print("Loading IndicF5 (Hugging Face)...")
        self.f5 = load_indicf5_model("ai4bharat/IndicF5", device)

    def _build_reference_stats(self):
        stats = []
        for item in self.library:
            try:
                info = sf.info(item["local_path"])
                duration = float(info.duration)
            except Exception:
                duration = 0.0

            transcript = item.get("transcript", "")
            transcript_bytes = max(1, len(transcript.encode("utf-8")))
            bytes_per_second = transcript_bytes / max(duration, 1e-6)

            stats.append(
                {
                    "duration": duration,
                    "bytes_per_second": bytes_per_second,
                    "valid_audio": duration > 0,
                }
            )

        return stats

    def _load_reference_audio(self, idx, max_seconds=8.0, target_sr=16000):
        item = self.library[idx]
        audio, sr = sf.read(item["local_path"], always_2d=True, dtype="float32")
        audio = audio.mean(axis=1)

        max_samples = int(sr * max_seconds)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]

        if sr != target_sr and audio.shape[0] > 1:
            old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
            new_len = max(1, int(audio.shape[0] * target_sr / sr))
            new_x = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            audio = np.interp(new_x, old_x, audio).astype(np.float32)

        audio = audio - float(np.mean(audio))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32), target_sr

    def _estimate_pitch_stats(self, audio, sr):
        frame_len = int(sr * 0.04)
        hop = int(sr * 0.02)
        if audio.shape[0] < frame_len:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        min_lag = max(1, int(sr / 350))
        max_lag = min(frame_len - 1, int(sr / 70))
        f0s = []

        for start in range(0, audio.shape[0] - frame_len + 1, hop):
            frame = audio[start : start + frame_len]
            frame = frame - np.mean(frame)
            energy = float(np.sqrt(np.mean(frame * frame)))
            if energy < 0.025:
                continue

            corr = np.correlate(frame, frame, mode="full")[frame_len - 1 :]
            if corr[0] <= 1e-6:
                continue

            search = corr[min_lag:max_lag]
            if search.size == 0:
                continue

            lag = int(np.argmax(search) + min_lag)
            confidence = float(corr[lag] / corr[0])
            if confidence >= 0.30:
                f0s.append(sr / lag)

        if not f0s:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        f0 = np.array(f0s, dtype=np.float32)
        return np.array(
            [
                float(np.median(f0) / 250.0),
                float((np.percentile(f0, 75) - np.percentile(f0, 25)) / 150.0),
                float(min(1.0, len(f0) / 80.0)),
            ],
            dtype=np.float32,
        )

    def speaker_proxy_embedding(self, idx):
        """
        Lightweight local speaker/timbre proxy.

        This is not a true speaker verification embedding, but it gives the
        selector an acoustic identity constraint without downloading another
        model. It combines average spectral shape, pitch statistics, energy,
        and zero-crossing behavior.
        """
        if idx in self.speaker_proxy_cache:
            return self.speaker_proxy_cache[idx]

        try:
            audio, sr = self._load_reference_audio(idx)
        except Exception:
            emb = np.zeros(51, dtype=np.float32)
            self.speaker_proxy_cache[idx] = emb
            return emb

        if audio.shape[0] < int(sr * 0.5):
            emb = np.zeros(51, dtype=np.float32)
            self.speaker_proxy_cache[idx] = emb
            return emb

        n_fft = 512
        hop = 160
        frame_len = 400
        if audio.shape[0] < frame_len:
            audio = np.pad(audio, (0, frame_len - audio.shape[0]))

        frames = []
        window = np.hanning(frame_len).astype(np.float32)
        for start in range(0, audio.shape[0] - frame_len + 1, hop):
            frame = audio[start : start + frame_len] * window
            frames.append(frame)

        frame_arr = np.stack(frames, axis=0)
        spectrum = np.abs(np.fft.rfft(frame_arr, n=n_fft, axis=1)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        speech_mask = (freqs >= 80) & (freqs <= 4200)
        spectrum = spectrum[:, speech_mask] + 1e-8
        freqs = freqs[speech_mask]

        band_edges = np.linspace(0, spectrum.shape[1], 40 + 1, dtype=int)
        bands = []
        for i in range(40):
            lo, hi = band_edges[i], max(band_edges[i] + 1, band_edges[i + 1])
            bands.append(float(np.mean(np.log(spectrum[:, lo:hi]))))
        bands = np.array(bands, dtype=np.float32)
        bands = (bands - bands.mean()) / (bands.std() + 1e-6)

        power_sum = spectrum.sum(axis=1) + 1e-8
        centroid = (spectrum * freqs[None, :]).sum(axis=1) / power_sum
        bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * spectrum).sum(axis=1) / power_sum)
        cumsum = np.cumsum(spectrum, axis=1)
        rolloff_idx = np.argmax(cumsum >= (0.85 * power_sum[:, None]), axis=1)
        rolloff = freqs[rolloff_idx]

        rms = np.sqrt(np.mean(frame_arr * frame_arr, axis=1))
        zcr = np.mean(np.abs(np.diff(np.signbit(frame_arr), axis=1)), axis=1)
        pitch = self._estimate_pitch_stats(audio, sr)

        stats = np.array(
            [
                float(np.mean(centroid) / 2500.0),
                float(np.std(centroid) / 1000.0),
                float(np.mean(bandwidth) / 2000.0),
                float(np.mean(rolloff) / 4200.0),
                float(np.mean(rms)),
                float(np.std(rms)),
                float(np.mean(zcr)),
                float(np.std(zcr)),
            ],
            dtype=np.float32,
        )

        emb = np.concatenate([bands, stats, pitch]).astype(np.float32)
        emb = np.nan_to_num(emb)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm

        self.speaker_proxy_cache[idx] = emb
        return emb

    def speaker_proxy_similarity(self, idx_a, idx_b):
        emb_a = self.speaker_proxy_embedding(idx_a)
        emb_b = self.speaker_proxy_embedding(idx_b)
        if not np.any(emb_a) or not np.any(emb_b):
            return 0.0
        return float(np.dot(emb_a, emb_b))

    def split_sentences(self, text):
        """Split Hindi text into sentence-sized chunks."""
        parts = re.split(r"([।?!\n])", text)
        result = []
        current = ""

        for part in parts:
            current += part
            if part in ["।", "?", "!", "\n"]:
                sentence = current.strip()
                if sentence:
                    result.append(sentence)
                current = ""

        if current.strip():
            result.append(current.strip())

        return [sentence for sentence in result if len(sentence) > 1]

    def encode_text(self, text):
        return self.clap.encode_text_string(text, device=self.device)

    def select_reference_pool(
        self,
        full_text,
        min_global_score=0.10,
        min_ref_duration=2.0,
        max_ref_duration=12.0,
        min_bytes_per_second=16.0,
        max_bytes_per_second=75.0,
        min_speaker_similarity=0.94,
        min_speaker_pool_size=4,
    ):
        """Choose a stable top-K reference pool for the whole audiobook."""
        global_emb = self.encode_text(full_text)
        global_sims = torch.matmul(global_emb, self.lib_embs.T).squeeze(0)

        valid_indices = []
        for idx, stats in enumerate(self.reference_stats):
            if not stats["valid_audio"]:
                continue
            if stats["duration"] < min_ref_duration or stats["duration"] > max_ref_duration:
                continue
            if stats["bytes_per_second"] < min_bytes_per_second or stats["bytes_per_second"] > max_bytes_per_second:
                continue
            valid_indices.append(idx)

        if not valid_indices:
            raise RuntimeError("No reference clips passed duration/transcript quality filters.")

        valid_tensor = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
        valid_scores = global_sims[valid_tensor]
        score_mask = valid_scores >= min_global_score

        if not torch.any(score_mask):
            best_score, best_pos = torch.max(valid_scores, dim=0)
            best_idx = valid_tensor[best_pos].item()
            best_item = self.library[best_idx]
            best_stats = self.reference_stats[best_idx]
            print(
                "Warning: no CLAP references passed the minimum global score. "
                f"Best was {best_item['id']} score={best_score.item():.3f}, "
                f"duration={best_stats['duration']:.2f}s. "
                "Using ranked valid clips anyway."
            )
            filtered_indices = valid_tensor
            filtered_scores = valid_scores
        else:
            filtered_indices = valid_tensor[score_mask]
            filtered_scores = valid_scores[score_mask]

        k = min(self.pool_size, filtered_indices.numel())
        pool_scores, positions = torch.topk(filtered_scores, k=k)
        pool_indices = filtered_indices[positions]
        
        # SPEAKER-LOCK: Find the absolute best global match to use as our "Lead Actor" anchor
        anchor_idx = pool_indices[0].item()
        anchor_score = pool_scores[0].item()

        # OPTIMIZATION: Only search for the lead speaker among the top 500 global matches
        # instead of the entire library. This prevents the "hang" on large libraries.
        search_k = min(500, filtered_indices.numel())
        search_scores, search_positions = torch.topk(filtered_scores, k=search_k)
        search_indices = filtered_indices[search_positions]

        print(f"Analyzing top {search_k} candidates to lock-in Lead Actor...")
        emb_anchor = self.speaker_proxy_embedding(anchor_idx)
        all_speaker_sims = []
        for idx in tqdm(search_indices.tolist(), desc="Acoustic Profiling"):
            emb_idx = self.speaker_proxy_embedding(idx)
            all_speaker_sims.append(float(np.dot(emb_anchor, emb_idx)))

        all_speaker_sims = np.array(all_speaker_sims, dtype=np.float32)

        # We want clips that are basically the same person. 0.98 is very tight.
        strict_threshold = 0.98
        speaker_mask = all_speaker_sims >= strict_threshold

        if int(np.sum(speaker_mask)) < min_speaker_pool_size:
            print(f"Refining Speaker-Lock: Only {int(np.sum(speaker_mask))} clips met 0.98. Relaxing to 0.96...")
            speaker_mask = all_speaker_sims >= 0.96

        final_pool_indices = search_indices[speaker_mask]
        final_pool_scores = search_scores[speaker_mask]
        final_speaker_sims = torch.tensor(all_speaker_sims[speaker_mask], device=self.device)

        # Sort by global score within the speaker-locked set
        top_k = min(self.pool_size, len(final_pool_indices))
        final_scores, final_pos = torch.topk(final_pool_scores, k=top_k)
        pool_indices = final_pool_indices[final_pos]
        pool_scores = final_scores
        speaker_sims = final_speaker_sims[final_pos].cpu().numpy()

        print(f"Speaker-Lock Active: Identified Lead Actor '{self.library[anchor_idx]['id']}'")
        print(f"Filtered library to {len(pool_indices)} clips belonging to this actor.")
        print(
            "Filters: "
            f"score>={min_global_score}, "
            f"duration={min_ref_duration}-{max_ref_duration}s, "
            f"bytes_per_second={min_bytes_per_second}-{max_bytes_per_second}, "
            f"speaker_similarity>={min_speaker_similarity}"
        )
        print("Top pool references:")
        for rank, (idx, score, spk_sim) in enumerate(
            zip(pool_indices[:5].tolist(), pool_scores[:5].tolist(), speaker_sims[:5].tolist()), start=1
        ):
            item = self.library[idx]
            stats = self.reference_stats[idx]
            transcript = item["transcript"].replace("\n", " ")[:70]
            print(
                f"  {rank}. {item['id']} score={score:.3f} "
                f"speaker_sim={spk_sim:.3f} duration={stats['duration']:.2f}s transcript={transcript}..."
            )

        return {
            "global_emb": global_emb,
            "global_sims": global_sims,
            "pool_indices": pool_indices,
            "pool_speaker_sims": torch.tensor(speaker_sims, device=self.device),
            "anchor_idx": anchor_idx,
            "anchor_score": anchor_score,
        }

    def select_reference_for_sentence(
        self,
        sentence,
        pool,
        previous_idx=None,
        recent_indices=None,
        local_weight=0.58,
        global_weight=0.27,
        continuity_weight=0.10,
        speaker_weight=0.35,
        recent_penalty=0.035,
    ):
        """
        Pick a reference clip from the global pool.

        local_weight: match this sentence's content/prosody.
        global_weight: stay close to the whole audiobook tone.
        continuity_weight: avoid abrupt jumps from the previous reference.
        recent_penalty: encourage subtle variation instead of repeating one clip.
        """
        recent_indices = recent_indices or set()
        pool_indices = pool["pool_indices"]

        local_emb = self.encode_text(sentence)
        local_sims_all = torch.matmul(local_emb, self.lib_embs.T).squeeze(0)

        pool_local_sims = local_sims_all[pool_indices]
        pool_global_sims = pool["global_sims"][pool_indices]

        score = (
            local_weight * pool_local_sims
            + global_weight * pool_global_sims
            + speaker_weight * pool["pool_speaker_sims"]
        )

        if previous_idx is not None:
            previous_emb = self.lib_embs[previous_idx]
            continuity_sims = torch.matmul(self.lib_embs[pool_indices], previous_emb)
            score = score + continuity_weight * continuity_sims

        if recent_indices:
            recent_mask = torch.tensor(
                [idx.item() in recent_indices for idx in pool_indices],
                device=self.device,
                dtype=score.dtype,
            )
            score = score - recent_penalty * recent_mask

        best_pool_pos = torch.argmax(score).item()
        best_idx = pool_indices[best_pool_pos].item()

        return best_idx

    def is_dialogue(self, text):
        """Detect if a sentence is likely dialogue based on punctuation."""
        dialogue_chars = ['"', '“', '”', '‘', '’', "'"]
        return any(char in text for char in dialogue_chars)

    def generate(
        self,
        text,
        output_path="audiobook_variable.wav",
        recent_window=3,
        local_weight=0.58,
        global_weight=0.27,
        continuity_weight=0.25,
        speaker_weight=0.70,
        recent_penalty=0.035,
        min_global_score=-1.0,
        min_ref_duration=2.0,
        max_ref_duration=12.0,
        min_bytes_per_second=16.0,
        max_bytes_per_second=75.0,
        min_speaker_similarity=0.94,
        min_speaker_pool_size=4,
        reference_hold_sentences=3,
    ):
        sentences = self.split_sentences(text)
        print(f"Synthesizing {len(sentences)} sentences...")
        if not sentences:
            raise ValueError("No sentence-like text found.")

        print("Selecting stable speaker-locked reference pool...")
        pool = self.select_reference_pool(
            text,
            min_global_score=min_global_score,
            min_ref_duration=min_ref_duration,
            max_ref_duration=max_ref_duration,
            min_bytes_per_second=min_bytes_per_second,
            max_bytes_per_second=max_bytes_per_second,
            min_speaker_similarity=min_speaker_similarity,
            min_speaker_pool_size=min_speaker_pool_size,
        )

        full_audio_segment = AudioSegment.silent(duration=0)
        previous_idx = None
        previous_was_dialogue = None
        recent = deque(maxlen=recent_window)
        used_ids = []

        for sent_idx, sent in enumerate(tqdm(sentences, desc="Generating chunks")):
            current_is_dialogue = self.is_dialogue(sent)
            
            # FORCE RE-SELECTION if we transition between narration and dialogue
            # This ensures dialogue gets an expressive reference from our locked narrator's pool
            force_reselect = (previous_was_dialogue is not None and current_is_dialogue != previous_was_dialogue)
            
            if not force_reselect and previous_idx is not None and reference_hold_sentences > 1 and sent_idx % reference_hold_sentences != 0:
                best_idx = previous_idx
            else:
                best_idx = self.select_reference_for_sentence(
                    sent,
                    pool,
                    previous_idx=previous_idx,
                    recent_indices=set(recent),
                    local_weight=local_weight,
                    global_weight=global_weight,
                    continuity_weight=continuity_weight,
                    speaker_weight=speaker_weight,
                    recent_penalty=recent_penalty,
                )
            best_match = self.library[best_idx]
            used_ids.append(best_match["id"])

            audio = self.f5(
                sent,
                ref_audio_path=best_match["local_path"],
                ref_text=best_match["transcript"],
            )

            # Convert numpy array from model to pydub AudioSegment
            audio_seg = AudioSegment(
                audio.tobytes(), 
                frame_rate=24000,
                sample_width=audio.dtype.itemsize, 
                channels=1
            )

            # CROSS-FADED STITCHING
            # Hide noise-floor shifts and "mechanical" seams between sentences
            if len(full_audio_segment) > 0:
                # Ensure crossfade is not longer than the clip itself (Pydub requirement)
                # We use at most 150ms, or half the clip duration for very short clips
                actual_crossfade = min(150, len(audio_seg) // 2)
                full_audio_segment = full_audio_segment.append(audio_seg, crossfade=actual_crossfade)
            else:
                full_audio_segment = audio_seg

            previous_idx = best_idx
            previous_was_dialogue = current_is_dialogue
            recent.append(best_idx)

        # Final Cleanup & Professional Polish
        noise_gen = WhiteNoise()
        white_noise = noise_gen.to_audio_segment(duration=len(full_audio_segment), volume=-68)
        full_audio_segment = full_audio_segment.overlay(white_noise) 

        full_audio_segment.export(output_path, format="wav")
        print(f"Finished. Speaker-locked audiobook saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Hindi text string or path to a .txt file containing the script",
    )
    parser.add_argument("--output", type=str, default="output_audiobook_variable.wav", help="Output path")
    parser.add_argument("--pool_size", type=int, default=80, help="Number of globally matched reference clips to inspect")
    parser.add_argument("--recent_window", type=int, default=3, help="How many recent references to lightly penalize")
    parser.add_argument("--local_weight", type=float, default=0.58, help="Sentence-level reference match weight")
    parser.add_argument("--global_weight", type=float, default=0.27, help="Whole-text consistency weight")
    parser.add_argument("--continuity_weight", type=float, default=0.25, help="Previous-reference continuity weight")
    parser.add_argument("--speaker_weight", type=float, default=0.70, help="Anchor-speaker similarity weight")
    parser.add_argument("--recent_penalty", type=float, default=0.035, help="Penalty for reusing recently selected references")
    parser.add_argument("--min_global_score", type=float, default=-1.0, help="Reject references below this global CLAP score")
    parser.add_argument("--min_ref_duration", type=float, default=2.0, help="Reject shorter reference clips")
    parser.add_argument("--max_ref_duration", type=float, default=12.0, help="Reject longer reference clips")
    parser.add_argument("--min_bytes_per_second", type=float, default=16.0, help="Reject transcript/audio pairs with too little text")
    parser.add_argument("--max_bytes_per_second", type=float, default=75.0, help="Reject transcript/audio pairs with too much text")
    parser.add_argument("--min_speaker_similarity", type=float, default=0.94, help="Keep references acoustically close to the anchor")
    parser.add_argument("--min_speaker_pool_size", type=int, default=4, help="Minimum same-speaker-neighborhood references to keep")
    parser.add_argument("--reference_hold_sentences", type=int, default=3, help="Reuse each selected reference for this many sentences")
    args = parser.parse_args()

    # Dynamic target check: read file if path exists, otherwise treat as text string
    raw_text = args.text
    if os.path.isfile(raw_text):
        print(f"Reading script text from file: {raw_text}")
        with open(raw_text, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        input_text = raw_text

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = VariableReferenceIndicAudiobookGenerator(device=device, pool_size=args.pool_size)
    gen.generate(
        input_text,
        args.output,
        recent_window=args.recent_window,
        local_weight=args.local_weight,
        global_weight=args.global_weight,
        continuity_weight=args.continuity_weight,
        speaker_weight=args.speaker_weight,
        recent_penalty=args.recent_penalty,
        min_global_score=args.min_global_score,
        min_ref_duration=args.min_ref_duration,
        max_ref_duration=args.max_ref_duration,
        min_bytes_per_second=args.min_bytes_per_second,
        max_bytes_per_second=args.max_bytes_per_second,
        min_speaker_similarity=args.min_speaker_similarity,
        min_speaker_pool_size=args.min_speaker_pool_size,
        reference_hold_sentences=args.reference_hold_sentences,
    )


if __name__ == "__main__":
    main()