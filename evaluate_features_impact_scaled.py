
"""
Scaled Feature Impact Evaluation: Punctuation Normalization & Qwen-2.5 Prosody

This script evaluates the impact of features across multiple stories and more segments.
It focuses on the Narrator (Speaker 'N') to ensure identity consistency.
"""

import os
import json
import csv
import torch
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import AutoTokenizer, ClapModel, ClapProcessor
from parler_tts import ParlerTTSForConditionalGeneration

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "ai4bharat/indic-parler-tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "StoricoTTSDataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
PROSODY_PLANS_DIR = "prosody_plans"

OUTPUT_DIR = "feature_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STORY_IDS = ["23", "44", "120"]
LIMIT_PER_STORY = 50  # Increased limit

# ============================================================================
# Helpers
# ============================================================================

def normalize_punctuation(text: str) -> str:
    replacements = {'।': '.', '॥': '.', '，': ',', '？': '?', '！': '!', '…': '...'}
    for hindi, english in replacements.items():
        text = text.replace(hindi, english)
    return text

def get_silence_duration(audio, sr, threshold_db=-40):
    if audio.size == 0:
        return 0
    try:
        # For very short audio, use a smaller frame length
        frame_length = min(2048, audio.size)
        hop_length = min(512, frame_length // 4) if frame_length > 4 else 1
        
        non_silent_intervals = librosa.effects.split(
            audio, 
            top_db=-threshold_db, 
            frame_length=frame_length, 
            hop_length=hop_length
        )
        if len(non_silent_intervals) == 0:
            return 0
        last_end = non_silent_intervals[-1][1]
        total_samples = len(audio)
        silence_samples = total_samples - last_end
        return silence_samples / sr
    except Exception:
        return 0

def get_pitch_variance(audio, sr):
    if audio.size < 2048:  # pyin needs a minimum length
        return 0
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        f0_valid = f0[~np.isnan(f0)]
        if len(f0_valid) < 2:
            return 0
        return np.std(f0_valid)
    except Exception:
        return 0

# ============================================================================
# Core Logic
# ============================================================================

def load_models():
    print("Loading TTS Model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    
    print("Loading CLAP Model...")
    clap_model = ClapModel.from_pretrained("laion/larger_clap_general").to(DEVICE)
    clap_processor = ClapProcessor.from_pretrained("laion/larger_clap_general")
    
    return model, tokenizer, desc_tokenizer, clap_model, clap_processor

def generate_variant(model, tokenizer, desc_tokenizer, text, caption, device):
    desc_inputs = desc_tokenizer(caption, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )
    return generation.cpu().numpy().squeeze()

def compute_l_sem(clap_model, clap_processor, audio, sr, caption, device):
    if audio.size == 0:
        return 0.0
    
    try:
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        
        t_inputs = clap_processor(text=[caption], return_tensors="pt", padding=True).to(device)
        a_inputs = clap_processor(audios=[audio], return_tensors="pt", sampling_rate=48000).to(device)
        
        with torch.no_grad():
            t_feat = clap_model.get_text_features(**t_inputs)
            a_feat = clap_model.get_audio_features(**a_inputs)
        
        t_feat /= t_feat.norm(dim=-1, keepdim=True)
        a_feat /= a_feat.norm(dim=-1, keepdim=True)
        return (t_feat @ a_feat.T).item()
    except Exception:
        return 0.0

def run_evaluation():
    model, tokenizer, desc_tokenizer, clap_model, clap_processor = load_models()
    sr = model.config.sampling_rate
    
    output_path = os.path.join(OUTPUT_DIR, "scaled_impact_results.json")
    
    # Load existing results if they exist to allow resuming
    if os.path.exists(output_path):
        print(f"Loading existing results from {output_path}...")
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            overall_results = existing_data.get("details", {k: [] for k in ["baseline", "punct_only", "prosody_only", "full_pipeline"]})
            story_stats = existing_data.get("per_story_stats", {})
    else:
        overall_results = {
            "baseline": [],
            "punct_only": [],
            "prosody_only": [],
            "full_pipeline": []
        }
        story_stats = {}

    for story_id in STORY_IDS:
        if story_id in story_stats:
            print(f"Skipping Story {story_id} (already evaluated).")
            continue
            
        print(f"\nEvaluating Story {story_id}...")
        
        # Load narrator segments
        segments = []
        with open(TEST_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['story'] == story_id and row['speaker'] == 'N':
                    segments.append(row)
        
        segments = segments[:LIMIT_PER_STORY]
        print(f"Processing {len(segments)} narrator segments.")
        
        # Load Qwen Prosody Plan
        plan_path = os.path.join(PROSODY_PLANS_DIR, f"story{story_id}_qwen.json")
        if not os.path.exists(plan_path):
            print(f"Warning: Prosody plan {plan_path} not found. Skipping story.")
            continue
            
        with open(plan_path, 'r', encoding='utf-8') as f:
            prosody_plan = json.load(f)
        
        story_results = {k: [] for k in overall_results.keys()}
        full_pipeline_audio_segments = []
        
        for seg in tqdm(segments):
            text_hindi = seg['norm']
            text_eng = normalize_punctuation(text_hindi)
            segment_id = os.path.basename(seg['audio_filepath']).replace('.wav', '')
            
            # Get Qwen Caption
            prosody_caption = "moderate pitch, clear articulation, neutral narration"
            for p_seg in prosody_plan.get('segments', []):
                if p_seg['segment_id'] == segment_id:
                    prosody_caption = p_seg.get('style_caption', prosody_caption)
                    break
            
            variants = [
                ("baseline", text_hindi, "neutral narration"),
                ("punct_only", text_eng, "neutral narration"),
                ("prosody_only", text_hindi, prosody_caption),
                ("full_pipeline", text_eng, prosody_caption)
            ]
            
            for v_name, v_text, v_caption in variants:
                audio = generate_variant(model, tokenizer, desc_tokenizer, v_text, v_caption, DEVICE)
                if v_name == "full_pipeline":
                    full_pipeline_audio_segments.append(audio)
                
                metrics = {
                    "pause": get_silence_duration(audio, sr),
                    "pitch_var": get_pitch_variance(audio, sr),
                    "l_sem": compute_l_sem(clap_model, clap_processor, audio, sr, v_caption, DEVICE)
                }
                story_results[v_name].append(metrics)
                overall_results[v_name].append(metrics)

        # Save stitched audio for full_pipeline
        if full_pipeline_audio_segments:
            pause_samples = int(sr * 0.5)
            silence = np.zeros(pause_samples)
            
            stitched_audio = full_pipeline_audio_segments[0]
            for audio_seg in full_pipeline_audio_segments[1:]:
                stitched_audio = np.concatenate([stitched_audio, silence, audio_seg])
            
            audio_out_path = os.path.join(OUTPUT_DIR, f"story{story_id}_full_pipeline.wav")
            sf.write(audio_out_path, stitched_audio, sr)
            print(f"Saved stitched audio for Story {story_id} to {audio_out_path}")

        # Per-story summary
        story_stats[story_id] = {
            v: {
                "avg_pause_ms": np.mean([s['pause'] for s in scores]) * 1000,
                "avg_pitch_var": np.mean([s['pitch_var'] for s in scores]),
                "avg_l_sem": np.mean([s['l_sem'] for s in scores])
            } for v, scores in story_results.items()
        }

        # Save progress after each story
        summary = {}
        for variant, scores in overall_results.items():
            if not scores: continue
            summary[variant] = {
                "avg_pause_ms": np.mean([s['pause'] for s in scores]) * 1000,
                "avg_pitch_var": np.mean([s['pitch_var'] for s in scores]),
                "avg_l_sem": np.mean([s['l_sem'] for s in scores])
            }
        
        output_payload = {
            "global_summary": summary,
            "per_story_stats": story_stats,
            "details": overall_results
        }
        with open(output_path, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"Saved progress for story {story_id}.")

    # Final Global Summary Statistics
    summary = {}
    for variant, scores in overall_results.items():
        if not scores: continue
        summary[variant] = {
            "avg_pause_ms": np.mean([s['pause'] for s in scores]) * 1000,
            "avg_pitch_var": np.mean([s['pitch_var'] for s in scores]),
            "avg_l_sem": np.mean([s['l_sem'] for s in scores])
        }
    
    # Print Final Global comparison
    print("\n" + "="*80)
    print("GLOBAL AGGREGATED RESULTS")
    print(f"{'Variant':<20} | {'Pause (ms)':<12} | {'Pitch Var':<12} | {'L_sem (↑)':<12}")
    print("-" * 80)
    for v, s in summary.items():
        print(f"{v:<20} | {s['avg_pause_ms']:>10.1f} | {s['avg_pitch_var']:>10.2f} | {s['avg_l_sem']:>10.4f}")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()
