"""
Auto-generate style captions for audio clips using Azure OpenAI GPT-4o Audio Preview.
Outputs a JSON file mapping segment_id → style_caption.
"""

import os
import json
import csv
import time
import base64
import requests
from pathlib import Path
from tqdm import tqdm

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://kiran-mmmx4mjp-eastus2.cognitiveservices.azure.com"
AZURE_DEPLOYMENT = "gpt-4o-audio-preview"
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_API_KEY = "9zzJMDMnmY6Gwqc1us3F5eWQoRurq7YUrDQWbBSGWtNEJGp6rNf0JQQJ99CCACHYHv6XJ3w3AAAAACOG5NCG"

CLIPS_DIR = "StoricoTTSDataset/clips"
OUTPUT_FILE = "StoricoTTSDataset/style_captions.json"

# Set to True to test with just 5 clips first
DRY_RUN = True
DRY_RUN_COUNT = 5

# Load test.csv to get segment metadata
def load_test_segments():
    segments = []
    with open('StoricoTTSDataset/test.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'segment_id': os.path.basename(row['audio_filepath']).replace('.wav', ''),
                'audio_filepath': row['audio_filepath'],
                'character': row.get('character', '').strip(),
                'emotion': row.get('emotion', '').strip(),
                'text': row.get('text', ''),
                'story': row.get('story', ''),
            })
    return segments

def get_style_caption(audio_path, text_hint=None):
    """Call Azure OpenAI GPT-4o Audio Preview API to get style caption for audio."""
    
    with open(audio_path, 'rb') as audio_file:
        audio_data = audio_file.read()
    
    # Encode as base64
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    
    # Build the request payload
    user_content = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": audio_base64,
                "format": "wav"
            }
        },
        {
            "type": "text",
            "text": "Describe the speaking style, emotion, and tone of this speech. Keep it to one concise sentence. Focus on prosody, emotion, and energy level."
        }
    ]
    
    if text_hint:
        user_content.append({
            "type": "text", 
            "text": f"Transcript (Hindi): {text_hint}"
        })
    
    # Azure OpenAI endpoint URL
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

def main():
    segments = load_test_segments()
    captions = {}
    
    # Filter segments that need captioning (no existing emotion)
    segments_to_process = [
        seg for seg in segments 
        if not seg['emotion']
    ]
    
    # Dry run: limit to first N segments
    if DRY_RUN:
        segments_to_process = segments_to_process[:DRY_RUN_COUNT]
        print(f"=== DRY RUN MODE: Processing only {len(segments_to_process)} segments ===")
    else:
        print(f"=== FULL RUN: Processing {len(segments_to_process)} segments ===")
    
    # Use tqdm for progress bar
    for seg in tqdm(segments_to_process, desc="Generating captions", unit="clip"):
        audio_path = os.path.join(CLIPS_DIR, os.path.basename(seg['audio_filepath']))
        
        if not os.path.exists(audio_path):
            tqdm.write(f"SKIP: {seg['segment_id']} (audio not found)")
            captions[seg['segment_id']] = {
                'style_caption': '[MISSING_AUDIO]',
                'character': seg['character'],
                'story': seg['story'],
                'had_emotion_annotation': False
            }
            continue
        
        try:
            caption = get_style_caption(audio_path, seg['text'])
            tqdm.write(f"DONE: {seg['segment_id']} → {caption[:70]}...")
        except Exception as e:
            caption = f"[ERROR] {str(e)}"
            tqdm.write(f"ERROR: {seg['segment_id']} - {e}")
        
        captions[seg['segment_id']] = {
            'style_caption': caption,
            'character': seg['character'],
            'story': seg['story'],
            'had_emotion_annotation': False
        }
        
        # Rate limiting
        time.sleep(2)
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Results ===")
    print(f"Saved captions to {OUTPUT_FILE}")
    print(f"Total processed: {len(captions)}")
    
    # Summary
    successful = sum(1 for v in captions.values() if not v['style_caption'].startswith('[ERROR]'))
    print(f"Successful: {successful}/{len(captions)}")
    
    if DRY_RUN:
        print(f"\n✓ Dry run complete! Set DRY_RUN = False to process all {len(segments) - sum(1 for s in segments if s['emotion'])} segments")

if __name__ == "__main__":
    main()
