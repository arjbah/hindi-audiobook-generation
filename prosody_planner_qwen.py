
import os
import json
import csv
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# Configuration
# ============================================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "StoricoTTSDataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "prosody_plans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STORY_IDS = ["23", "44", "120"]

# ============================================================================
# Helpers
# ============================================================================

def load_narrator_segments(story_id: str) -> list:
    segments = []
    with open(TEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by story AND speaker 'N' (Narrator)
            if row['story'] == story_id and row['speaker'] == 'N':
                segments.append({
                    'segment_id': os.path.basename(row['audio_filepath']).replace('.wav', ''),
                    'text': row.get('norm', row.get('text', '')),
                    'character': row.get('character', ''),
                    'emotion': row.get('emotion', '')
                })
    return segments

def generate_captions():
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype="auto", 
        device_map="auto"
    )

    for story_id in STORY_IDS:
        print(f"\nProcessing Story {story_id}...")
        segments = load_narrator_segments(story_id)
        print(f"Found {len(segments)} narrator segments.")
        
        plan = {
            "story_id": story_id,
            "model": MODEL_ID,
            "segments": []
        }
        
        for seg in tqdm(segments):
            prompt = f"Analyze the following Hindi text from a children's story and provide a short, descriptive style caption in English (10-20 words) for a TTS narrator. Focus on tone, pace, and emotional delivery.\n\nHindi Text: {seg['text']}\n\nStyle Caption:"
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant specializing in audiobook production and TTS style prompting."},
                {"role": "user", "content": prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            plan["segments"].append({
                "segment_id": seg['segment_id'],
                "text": seg['text'],
                "style_caption": caption
            })
            
        output_path = os.path.join(OUTPUT_DIR, f"story{story_id}_qwen.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        print(f"Saved plan to {output_path}")

if __name__ == "__main__":
    generate_captions()
