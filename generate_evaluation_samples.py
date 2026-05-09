import os
import json
import csv
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "ai4bharat/indic-parler-tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "StoricoTTSDataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
PROSODY_PLANS_DIR = "prosody_plans"
OUTPUT_DIR = "feature_evaluation_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STORY_IDS = ["23", "44", "120"]
LIMIT_PER_STORY = 50 
PERSONA = "Rohit"

# ============================================================================
# Helpers
# ============================================================================

def normalize_punctuation(text: str) -> str:
    """Replace Hindi punctuation with English equivalents."""
    replacements = {'।': '.', '॥': '.', '，': ',', '？': '?', '！': '!', '…': '...'}
    for hindi, english in replacements.items():
        text = text.replace(hindi, english)
    return text

def load_tts_model():
    print("Loading TTS Model...")
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="auto").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    return model, tokenizer, desc_tokenizer

def generate_audio(model, tokenizer, desc_tokenizer, text, caption, device):
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

# ============================================================================
# Main Inference Loop
# ============================================================================

def run_inference():
    model, tokenizer, desc_tokenizer = load_tts_model()
    sr = model.config.sampling_rate
    
    for story_id in STORY_IDS:
        out_path = os.path.join(OUTPUT_DIR, f"story{story_id}_inference.wav")
        if os.path.exists(out_path):
            print(f"Skipping Story {story_id} (already exists at {out_path})")
            continue

        print(f"\nGenerating Samples for Story {story_id}...")
        
        # Load narrator segments
        segments = []
        with open(TEST_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['story'] == story_id and row['speaker'] == 'N':
                    segments.append(row)
        
        segments = segments[:LIMIT_PER_STORY]
        
        # Load Qwen Prosody Plan
        plan_path = os.path.join(PROSODY_PLANS_DIR, f"story{story_id}_qwen.json")
        if not os.path.exists(plan_path):
            print(f"Warning: Prosody plan {plan_path} not found. Skipping.")
            continue
            
        with open(plan_path, 'r', encoding='utf-8') as f:
            prosody_plan = json.load(f)
        
        audio_segments = []
        
        for seg in tqdm(segments, desc=f"Story {story_id}"):
            # Use English Punctuation (Full Pipeline Logic)
            text_eng = normalize_punctuation(seg['norm'])
            segment_id = os.path.basename(seg['audio_filepath']).replace('.wav', '')
            
            # Get Qwen Caption
            prosody_caption = "moderate pitch, clear articulation, neutral narration"
            for p_seg in prosody_plan.get('segments', []):
                if p_seg['segment_id'] == segment_id:
                    prosody_caption = p_seg.get('style_caption', prosody_caption)
                    break
            
            # Prepend Persona to prevent drift
            full_caption = f"{PERSONA} speaks, {prosody_caption}"
            
            # Inference Only
            try:
                audio = generate_audio(model, tokenizer, desc_tokenizer, text_eng, full_caption, DEVICE)
                audio_segments.append(audio)
            except Exception as e:
                print(f"Error generating segment {segment_id}: {e}")

        # Stitch and Save
        if audio_segments:
            pause_samples = int(sr * 0.5)
            silence = np.zeros(pause_samples)
            
            stitched_audio = audio_segments[0]
            for audio_seg in audio_segments[1:]:
                # Ensure 1D
                if len(audio_seg.shape) > 1:
                    audio_seg = audio_seg.flatten()
                stitched_audio = np.concatenate([stitched_audio, silence, audio_seg])
            
            sf.write(out_path, stitched_audio, sr)
            print(f"✓ Saved stitched audio to {out_path}")

if __name__ == "__main__":
    run_inference()
