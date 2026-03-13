"""
Compute L_attr Metric Using Trained Attribute Classifier

This script computes L_attr (attribute prediction error) for evaluating
style adherence in IndicParler-TTS generated speech.

Usage:
    python compute_l_attr.py

Requirements:
    - Trained attribute classifier (from train_attribute_classifier.py)
    - Generated audio files (from evaluate_style_adherence.py)
    - style_captions.json
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from train_attribute_classifier import AttributeClassifier, Config

# ============================================================================
# Configuration
# ============================================================================

GENERATED_AUDIO_DIR = "evaluation_output/generated"
CAPTIONS_FILE = "StoricoTTSDataset/style_captions.json"
CLASSIFIER_DIR = "attribute_classifier"
OUTPUT_FILE = "evaluation_output/l_attr_results.json"


# ============================================================================
# Caption → Target Attributes
# ============================================================================

def get_target_attributes_from_caption(caption: str) -> dict:
    """
    Extract target attributes from style caption.
    Uses the same mapping as training.
    """
    from train_attribute_classifier import map_caption_to_attributes
    return map_caption_to_attributes(caption)


# ============================================================================
# L_attr Computation
# ============================================================================

def compute_attribute_error(predicted: dict, target: dict) -> dict:
    """
    Compute prediction error for each attribute.
    
    Returns:
        Dict with per-attribute errors (0 = match, 1 = mismatch)
    """
    errors = {}
    
    for attr in ['emotion', 'rate', 'pitch', 'energy']:
        if attr in predicted and attr in target:
            errors[attr] = 0 if predicted[attr] == target[attr] else 1
        else:
            errors[attr] = None  # Missing
    
    return errors


def compute_l_attr_segment(
    audio_path: str,
    target_attributes: dict,
    classifier: AttributeClassifier,
) -> dict:
    """
    Compute L_attr for a single segment.
    
    L_attr = average error across all attributes
    
    Returns:
        Dict with l_attr score and per-attribute errors
    """
    # Predict attributes from audio
    try:
        predicted = classifier.predict(audio_path)
    except Exception as e:
        return {
            'l_attr': None,
            'error': str(e),
            'predicted': None,
            'target': target_attributes,
        }
    
    # Compute error
    errors = compute_attribute_error(predicted, target_attributes)
    
    # Average error (ignoring None values)
    valid_errors = [v for v in errors.values() if v is not None]
    l_attr = np.mean(valid_errors) if valid_errors else None
    
    return {
        'l_attr': l_attr,
        'errors': errors,
        'predicted': predicted,
        'target': target_attributes,
    }


# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("="*60)
    print("L_attr Evaluation: Attribute Prediction Error")
    print("="*60)
    
    # Load classifier
    print("\nLoading attribute classifier...")
    classifier = AttributeClassifier(CLASSIFIER_DIR)
    
    if not classifier.models:
        print("ERROR: No attribute classifiers found.")
        print(f"Run train_attribute_classifier.py first to train models in {CLASSIFIER_DIR}")
        return
    
    print(f"Loaded classifiers: {list(classifier.models.keys())}")
    
    # Load captions
    print("\nLoading style captions...")
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        style_captions = json.load(f)
    
    # Find generated audio files
    print("\nSearching for generated audio...")
    generated_files = list(Path(GENERATED_AUDIO_DIR).glob("*_gen.wav"))
    
    if not generated_files:
        print(f"ERROR: No generated audio found in {GENERATED_AUDIO_DIR}")
        print("Run evaluate_style_adherence.py first to generate audio.")
        return
    
    print(f"Found {len(generated_files)} generated audio files")
    
    # Evaluate each segment
    results = []
    attribute_errors = {
        'emotion': [],
        'rate': [],
        'pitch': [],
        'energy': [],
    }
    
    print("\nComputing L_attr...")
    
    for audio_path in tqdm(generated_files, desc="Evaluating", unit="audio"):
        segment_id = audio_path.stem.replace('_gen', '')
        
        # Get caption
        caption_data = style_captions.get(segment_id, {})
        caption = caption_data.get('style_caption', 'neutral narration')
        
        # Clean caption
        if caption.startswith('[EXISTING]'):
            caption = caption.replace('[EXISTING]', '').strip()
        elif caption.startswith('[ERROR]') or not caption:
            caption = 'neutral narration'
        
        # Get target attributes
        target = get_target_attributes_from_caption(caption)
        
        # Compute L_attr
        result = compute_l_attr_segment(str(audio_path), target, classifier)
        result['segment_id'] = segment_id
        result['caption'] = caption
        
        results.append(result)
        
        # Collect per-attribute errors
        if result.get('errors'):
            for attr, err in result['errors'].items():
                if err is not None:
                    attribute_errors[attr].append(err)
    
    # Aggregate results
    print("\n" + "="*60)
    print("L_attr RESULTS")
    print("="*60)
    
    # Overall L_attr
    valid_l_attr = [r['l_attr'] for r in results if r['l_attr'] is not None]
    if valid_l_attr:
        overall_l_attr = np.mean(valid_l_attr)
        print(f"\nOverall L_attr: {overall_l_attr:.4f}")
        print(f"  (Lower = better attribute adherence)")
    else:
        print("\nOverall L_attr: N/A (no valid results)")
    
    # Per-attribute error rates
    print("\nPer-Attribute Error Rates:")
    print("-"*40)
    print(f"{'Attribute':<15} {'Error Rate':<15} {'Accuracy':<15}")
    print("-"*40)
    
    for attr in ['emotion', 'rate', 'pitch', 'energy']:
        errors = attribute_errors[attr]
        if errors:
            error_rate = np.mean(errors)
            accuracy = 1 - error_rate
            print(f"{attr:<15} {error_rate:.4f}          {accuracy:.4f}")
        else:
            print(f"{attr:<15} N/A")
    
    # By caption quality
    gpt4o_results = [r for r in results if '[ERROR]' not in r.get('caption', '') and r['l_attr'] is not None]
    if gpt4o_results:
        print(f"\nL_attr (GPT-4o captions only): {np.mean([r['l_attr'] for r in gpt4o_results]):.4f}")
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    output_data = {
        'overall_l_attr': float(np.mean(valid_l_attr)) if valid_l_attr else None,
        'per_attribute': {
            attr: {
                'error_rate': float(np.mean(attribute_errors[attr])) if attribute_errors[attr] else None,
                'accuracy': float(1 - np.mean(attribute_errors[attr])) if attribute_errors[attr] else None,
            }
            for attr in ['emotion', 'rate', 'pitch', 'energy']
        },
        'segment_results': results,
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("\n" + "="*60)
    print("L_attr evaluation complete!")
    print("="*60)
    
    # Interpretation guide
    print("\nINTERPRETATION GUIDE:")
    print("-"*40)
    print("L_attr = 0.0: Perfect attribute adherence")
    print("L_attr = 0.3: Good (70% attributes match)")
    print("L_attr = 0.5: Moderate (50% attributes match)")
    print("L_attr = 1.0: Poor (no attributes match)")
    print("\nPer-attribute accuracy shows which aspects")
    print("the model struggles with most.")


if __name__ == "__main__":
    main()
