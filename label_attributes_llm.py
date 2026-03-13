"""
LLM-Based Attribute Labeling for Training Data

Uses GPT-4o (text-only) to map style captions to structured attributes.
Much more accurate than keyword matching.

Usage:
    python label_attributes_llm.py

Requirements:
    - style_captions.json (from generate_captions.py)
    - Azure OpenAI API key
"""

import os
import json
import csv
import time
import requests
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# Azure OpenAI Configuration
# ============================================================================

AZURE_ENDPOINT = "https://kiran-mmmx4mjp-eastus2.cognitiveservices.azure.com"
AZURE_DEPLOYMENT = "gpt-4o"
AZURE_API_VERSION = "2025-01-01-preview"
AZURE_API_KEY = "9zzJMDMnmY6Gwqc1us3F5eWQoRurq7YUrDQWbBSGWtNEJGp6rNf0JQQJ99CCACHYHv6XJ3w3AAAAACOG5NCG"

DATA_DIR = "StoricoTTSDataset"
CAPTIONS_FILE = os.path.join(DATA_DIR, "style_captions.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "attributes_labeled.json")

# Set to True to test with just 20 captions first
DRY_RUN = True
DRY_RUN_COUNT = 20

# Rate limiting
REQUESTS_PER_MINUTE = 60
DELAY_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE  # ~1 second

# ============================================================================
# Prompt for Attribute Extraction
# ============================================================================

SYSTEM_PROMPT = """You are an expert at analyzing speech style descriptions and extracting structured attributes.

Your task is to map free-text style captions to discrete attribute labels.

Attributes to extract:
1. emotion: One of [neutral, excited, calm, sad, angry, frustrated, happy, surprised, fearful, disgusted]
2. rate: One of [slow, normal, fast]
3. pitch: One of [low, medium, high]
4. energy: One of [low, medium, high]

Rules:
- If the caption doesn't specify an attribute, use the default (neutral for emotion, normal/medium for others)
- Be consistent - similar captions should get similar labels
- Consider context and implied meaning, not just keywords
- Output ONLY valid JSON, no additional text"""

USER_PROMPT_TEMPLATE = """Style caption: "{caption}"

Extract attributes as JSON:
{{
    "emotion": "...",
    "rate": "...",
    "pitch": "...",
    "energy": "..."
}}"""


# ============================================================================
# LLM Labeling Function
# ============================================================================

def label_caption_with_llm(caption: str) -> dict:
    """
    Use GPT-4o (text) to extract structured attributes from a caption.
    
    Returns:
        Dict with emotion, rate, pitch, energy labels
    """
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(caption=caption)
            }
        ],
        "max_tokens": 150,
        "temperature": 0.1,  # Low temperature for consistency
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    
    # Parse JSON response
    try:
        attributes = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if LLM doesn't return valid JSON
        attributes = {
            'emotion': 'neutral',
            'rate': 'normal',
            'pitch': 'medium',
            'energy': 'medium'
        }
    
    return attributes


# ============================================================================
# Main Processing
# ============================================================================

def main():
    print("="*60)
    print("LLM-Based Attribute Labeling")
    print("="*60)
    print(f"Endpoint: {AZURE_ENDPOINT}")
    print(f"Deployment: {AZURE_DEPLOYMENT}")
    print(f"Output: {OUTPUT_FILE}")

    # Load captions
    print("\nLoading style captions...")
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        style_captions = json.load(f)

    print(f"Total captions: {len(style_captions)}")

    # Filter to only captions that need labeling
    captions_to_label = {}
    for segment_id, data in style_captions.items():
        caption = data.get('style_caption', '')

        # Skip if already an error or missing
        if not caption or caption.startswith('[ERROR]'):
            continue

        # Skip if it's an [EXISTING] annotation (already has emotion)
        if caption.startswith('[EXISTING]'):
            # Extract emotion from existing annotation
            emotion_text = caption.replace('[EXISTING]', '').strip()
            captions_to_label[segment_id] = {
                'caption': caption,
                'existing_emotion': emotion_text,
            }
        else:
            captions_to_label[segment_id] = {
                'caption': caption,
                'existing_emotion': None,
            }

    print(f"Captions to label: {len(captions_to_label)}")

    # Dry run mode
    if DRY_RUN:
        print(f"\n=== DRY RUN MODE: Processing only {DRY_RUN_COUNT} captions ===")
        captions_to_label = dict(list(captions_to_label.items())[:DRY_RUN_COUNT])
        print(f"Captions to label (dry run): {len(captions_to_label)}")
    
    # Label each caption
    labeled_data = {}
    errors = []
    
    print("\nLabeling captions with GPT-4o...")
    
    for segment_id, data in tqdm(captions_to_label.items(), desc="Labeling", unit="caption"):
        caption = data['caption']
        
        try:
            attributes = label_caption_with_llm(caption)
            
            # Validate attributes
            valid_emotions = ['neutral', 'excited', 'calm', 'sad', 'angry', 'frustrated', 'happy', 'surprised', 'fearful', 'disgusted']
            valid_rates = ['slow', 'normal', 'fast']
            valid_pitches = ['low', 'medium', 'high']
            valid_energies = ['low', 'medium', 'high']
            
            if attributes.get('emotion') not in valid_emotions:
                attributes['emotion'] = 'neutral'
            if attributes.get('rate') not in valid_rates:
                attributes['rate'] = 'normal'
            if attributes.get('pitch') not in valid_pitches:
                attributes['pitch'] = 'medium'
            if attributes.get('energy') not in valid_energies:
                attributes['energy'] = 'medium'
            
            labeled_data[segment_id] = {
                'caption': caption,
                'attributes': attributes,
                'source': 'llm_labeled',
            }
            
            tqdm.write(f"  {segment_id}: {attributes}")
            
        except Exception as e:
            errors.append({
                'segment_id': segment_id,
                'error': str(e),
            })
            tqdm.write(f"  ERROR {segment_id}: {e}")
        
        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)
        
        # Save progress every 20 captions
        if len(labeled_data) % 20 == 0:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(labeled_data, f, indent=2, ensure_ascii=False)
            tqdm.write(f"  → Saved progress ({len(labeled_data)} labeled)")
    
    # Final save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(labeled_data, f, indent=2, ensure_ascii=False)
    
    # Save errors
    if errors:
        error_file = OUTPUT_FILE.replace('.json', '_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        print(f"\nErrors saved to: {error_file}")
    
    # Summary
    print("\n" + "="*60)
    print("LABELING COMPLETE")
    print("="*60)
    print(f"Successfully labeled: {len(labeled_data)}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Show attribute distribution
    print("\n" + "="*60)
    print("ATTRIBUTE DISTRIBUTION")
    print("="*60)
    
    attr_counts = {
        'emotion': {},
        'rate': {},
        'pitch': {},
        'energy': {},
    }
    
    for data in labeled_data.values():
        attrs = data['attributes']
        for attr_name, attr_val in attrs.items():
            if attr_val not in attr_counts[attr_name]:
                attr_counts[attr_name][attr_val] = 0
            attr_counts[attr_name][attr_val] += 1
    
    for attr_name, counts in attr_counts.items():
        print(f"\n{attr_name.upper()}:")
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / len(labeled_data) * 100 if labeled_data else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:<15} {count:>5} ({pct:>5.1f}%) {bar}")

    print("\n" + "="*60)
    if DRY_RUN:
        print("✓ DRY RUN COMPLETE!")
        print(f"Set DRY_RUN = False to process all {len(style_captions)} captions")
    else:
        print("LABELING COMPLETE!")
    print("Next step: Run train_attribute_classifier.py")
    print("="*60)


if __name__ == "__main__":
    main()
