#!/usr/bin/env python3
"""
Evaluate ASR Accuracy for IndicParler-TTS Generated Audio

Computes Word Error Rate (WER) between:
1. Original Hindi transcripts (from test.csv) - in Devanagari
2. ASR transcription of generated audio (using MMS with Hindi adapter) - in Devanagari

Also provides English translations using IndicTrans2 for verification.

Usage:
    python evaluate_asr_accuracy.py                          # Full evaluation
    python evaluate_asr_accuracy.py --quick-test             # Test with 10 samples
    python evaluate_asr_accuracy.py --audio-file <path>      # Single file
"""

import os
import csv
import json
import argparse
import torch
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2ForCTC, AutoProcessor, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import Levenshtein

# ============================================================================
# Configuration
# ============================================================================

ASR_MODEL = "facebook/mms-1b-all"
TRANSLATION_MODEL = "ai4bharat/indictrans2-indic-en-1B"
DATA_DIR = "StoricoTTSDataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
GENERATED_AUDIO_DIR = "evaluation_output/generated"


# ============================================================================
# MMS ASR with Hindi Adapter
# ============================================================================

def load_mms_asr(model_id=ASR_MODEL, device="cpu"):
    """Load MMS model with Hindi adapter."""
    print(f"Loading MMS ASR model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    
    # Set to Hindi language
    processor.tokenizer.set_target_lang("hin")
    model.load_adapter("hin")
    
    model = model.to(device)
    return model, processor


def transcribe_audio_mms(model, processor, audio_path, device="cpu"):
    """
    Transcribe audio using MMS model with Hindi adapter.
    Returns Hindi text in Devanagari script.
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Process audio
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Transcribe
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # Decode
    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    
    return transcription


# ============================================================================
# IndicTrans2 Translation (Hindi -> English)
# ============================================================================

def load_indictrans2_model(model_id=TRANSLATION_MODEL, device="cuda"):
    """Load IndicTrans2 Hindi-to-English translation model."""
    print(f"Loading IndicTrans2 model: {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    
    ip = IndicProcessor(inference=True)
    
    return model, tokenizer, ip


def translate_hindi_to_english(model, tokenizer, ip, hindi_text, device="cuda"):
    """
    Translate Hindi text (Devanagari) to English using IndicTrans2.
    """
    # Preprocess
    batch = ip.preprocess_batch(
        [hindi_text],
        src_lang="hin_Deva",
        tgt_lang="eng_Latn",
    )
    
    # Tokenize
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)
    
    # Generate translation
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )
    
    # Decode
    generated_tokens = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    
    # Postprocess
    translations = ip.postprocess_batch(generated_tokens, lang="eng_Latn")
    
    return translations[0]


# ============================================================================
# WER Computation
# ============================================================================

def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate between reference and hypothesis text.
    
    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=words in reference
    
    Returns:
        wer: Word Error Rate (0.0 = perfect, 1.0 = completely different)
        details: dict with S, D, I, N counts
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if len(ref_words) == 0:
        return 1.0, {'substitutions': 0, 'deletions': 0, 'insertions': 0, 'total_words': 0, 'wer': 1.0}
    
    # Use Levenshtein distance to compute edit operations
    edit_ops = Levenshtein.opcodes(ref_words, hyp_words)
    
    s, d, i = 0, 0, 0
    for op, i1, i2, j1, j2 in edit_ops:
        if op == 'replace':
            s += (i2 - i1)
        elif op == 'delete':
            d += (i2 - i1)
        elif op == 'insert':
            i += (j2 - j1)
    
    n = len(ref_words)
    wer = (s + d + i) / n if n > 0 else 0.0
    
    return wer, {
        'substitutions': s,
        'deletions': d,
        'insertions': i,
        'total_words': n,
        'wer': wer
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_test_segments():
    """Load test.csv segments."""
    segments = []
    with open(TEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'segment_id': os.path.basename(row['audio_filepath']).replace('.wav', ''),
                'audio_filepath': row['audio_filepath'],
                'text': row.get('text', ''),
                'story': row.get('story', ''),
            })
    return segments


def get_generated_audio_path(segment_id):
    """Get path to generated audio file."""
    return os.path.join(GENERATED_AUDIO_DIR, f"{segment_id}_gen.wav")


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_single_file(audio_file, asr_model, asr_processor, trans_model, trans_tokenizer, ip, device="cpu"):
    """Evaluate a single audio file."""
    
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return None
    
    # Extract segment_id from filename
    segment_id = audio_path.stem.replace('_gen', '')
    
    # Try to find original transcript from test.csv
    original_text = None
    segments = load_test_segments()
    for seg in segments:
        if seg['segment_id'] == segment_id:
            original_text = seg['text']
            break
    
    print(f"\n{'='*70}")
    print(f"File: {audio_path.name}")
    print(f"{'='*70}")
    
    # Transcribe generated audio
    print("\n[ASR Transcription of Generated Audio]")
    asr_text = transcribe_audio_mms(asr_model, asr_processor, str(audio_path), device)
    print(f"Hindi: {asr_text}")
    
    # Translate ASR output to English
    asr_english = translate_hindi_to_english(trans_model, trans_tokenizer, ip, asr_text, device)
    print(f"English: {asr_english}")
    
    # Compare with original if available
    if original_text:
        print(f"\n[Original Transcript]")
        print(f"Hindi: {original_text}")
        
        # Translate original to English
        orig_english = translate_hindi_to_english(trans_model, trans_tokenizer, ip, original_text, device)
        print(f"English: {orig_english}")
        
        # Compute WER (both in Devanagari)
        wer, wer_details = compute_wer(original_text, asr_text)
        
        print(f"\n[Word Error Rate]")
        print(f"WER: {wer:.2%}")
        print(f"Details: S={wer_details['substitutions']}, D={wer_details['deletions']}, "
              f"I={wer_details['insertions']}, N={wer_details['total_words']}")
        
        return {
            'segment_id': segment_id,
            'original_hindi': original_text,
            'asr_hindi': asr_text,
            'original_english': orig_english,
            'asr_english': asr_english,
            'wer': wer,
            'wer_details': wer_details
        }
    else:
        print("\n[Note] Original transcript not found in test.csv")
        return {
            'segment_id': segment_id,
            'asr_hindi': asr_text,
            'asr_english': asr_english,
            'wer': None
        }


def run_full_evaluation(quick_test=False, quick_test_limit=10, device="cpu"):
    """Run evaluation on all generated audio files."""
    
    # Load models
    asr_model, asr_processor = load_mms_asr(ASR_MODEL, device)
    trans_model, trans_tokenizer, ip = load_indictrans2_model(TRANSLATION_MODEL, device)
    
    # Load segments
    segments = load_test_segments()
    
    # Filter to only segments with generated audio
    eval_segments = []
    for seg in segments:
        gen_path = get_generated_audio_path(seg['segment_id'])
        if os.path.exists(gen_path):
            eval_segments.append(seg)
    
    if quick_test:
        print(f"\n=== QUICK TEST MODE: {quick_test_limit} samples ===")
        eval_segments = eval_segments[:quick_test_limit]
    
    print(f"\nEvaluating {len(eval_segments)} segments...")
    
    results = []
    wer_scores = []
    
    for idx, seg in enumerate(eval_segments):
        gen_path = get_generated_audio_path(seg['segment_id'])
        
        print(f"\n[{idx+1}/{len(eval_segments)}] {seg['segment_id']}")
        print("-" * 50)
        
        # Transcribe
        asr_text = transcribe_audio_mms(asr_model, asr_processor, gen_path, device)
        
        # Compute WER
        original_text = seg['text']
        wer, wer_details = compute_wer(original_text, asr_text)
        wer_scores.append(wer)
        
        # Translate both to English
        orig_english = translate_hindi_to_english(trans_model, trans_tokenizer, ip, original_text, device)
        asr_english = translate_hindi_to_english(trans_model, trans_tokenizer, ip, asr_text, device)
        
        # Print summary (truncated for long texts)
        print(f"Original (Hindi):   {original_text[:60]}...")
        print(f"Original (English): {orig_english[:60]}...")
        print(f"ASR (Hindi):        {asr_text[:60]}...")
        print(f"ASR (English):      {asr_english[:60]}...")
        print(f"WER: {wer:.2%}")
        
        results.append({
            'segment_id': seg['segment_id'],
            'story': seg['story'],
            'original_hindi': original_text,
            'original_english': orig_english,
            'asr_hindi': asr_text,
            'asr_english': asr_english,
            'wer': wer,
            'wer_details': wer_details
        })
    
    # Aggregate statistics
    print("\n" + "="*70)
    print("AGGREGATE STATISTICS")
    print("="*70)
    
    import numpy as np
    wer_array = np.array(wer_scores)
    print(f"\nSegments evaluated: {len(wer_scores)}")
    print(f"Mean WER: {wer_array.mean():.2%}")
    print(f"Median WER: {np.median(wer_array):.2%}")
    print(f"Std WER: {wer_array.std():.2%}")
    print(f"Min WER: {wer_array.min():.2%}")
    print(f"Max WER: {wer_array.max():.2%}")
    
    # WER distribution
    excellent = sum(1 for w in wer_scores if w < 0.2)
    good = sum(1 for w in wer_scores if 0.2 <= w < 0.4)
    fair = sum(1 for w in wer_scores if 0.4 <= w < 0.6)
    poor = sum(1 for w in wer_scores if w >= 0.6)
    
    print(f"\nWER Distribution:")
    print(f"  Excellent (<20%):  {excellent} ({excellent/len(wer_scores)*100:.1f}%)")
    print(f"  Good (20-40%):     {good} ({good/len(wer_scores)*100:.1f}%)")
    print(f"  Fair (40-60%):     {fair} ({fair/len(wer_scores)*100:.1f}%)")
    print(f"  Poor (>60%):       {poor} ({poor/len(wer_scores)*100:.1f}%)")
    
    # Save results
    output_path = "evaluation_output/asr_accuracy_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'segments': results,
            'aggregate': {
                'mean_wer': float(wer_array.mean()),
                'median_wer': float(np.median(wer_array)),
                'std_wer': float(wer_array.std()),
                'min_wer': float(wer_array.min()),
                'max_wer': float(wer_array.max()),
                'distribution': {
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor
                }
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR accuracy for IndicParler-TTS generated audio"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        help="Path to a single audio file to evaluate"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run on first 10 samples only"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ASR inference"
    )
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Single file mode
        asr_model, asr_processor = load_mms_asr(ASR_MODEL, args.device)
        trans_model, trans_tokenizer, ip = load_indictrans2_model(TRANSLATION_MODEL, args.device)
        evaluate_single_file(args.audio_file, asr_model, asr_processor, trans_model, trans_tokenizer, ip, args.device)
    else:
        # Full evaluation mode
        run_full_evaluation(quick_test=args.quick_test, device=args.device)


if __name__ == "__main__":
    main()
