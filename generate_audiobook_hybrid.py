"""
Hybrid Audiobook Generation Pipeline

Combines:
1. Speaker Embedding Lock (IndicParler level)
2. Voice Conversion (OpenVoice post-processing)

For perfect speaker consistency in multi-speaker audiobooks.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  Hybrid Pipeline                                            │
    │                                                             │
    │  IndicParler with Speaker Lock                              │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ Segment 1: Extract embedding → Generate audio        │  │
    │  │ Segment 2: Reuse embedding → Generate audio          │  │
    │  │ Segment 3: Reuse embedding → Generate audio          │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                          ↓                                  │
    │  OpenVoice Voice Conversion                                 │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │ Audio 1 → Convert with reference → Consistent voice  │  │
    │  │ Audio 2 → Convert with reference → Consistent voice  │  │
    │  │ Audio 3 → Convert with reference → Consistent voice  │  │
    │  └──────────────────────────────────────────────────────┘  │
    │                          ↓                                  │
    │  Stitch with Crossfade → Final Audiobook                    │
    └─────────────────────────────────────────────────────────────┘

Usage:
    python generate_audiobook_hybrid.py --story_id 23
    python generate_audiobook_hybrid.py --story_id 23 --no-vc  # Skip voice conversion
    python generate_audiobook_hybrid.py --all-stories
"""

import os
import json
import csv
import argparse
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

from speaker_embedding_lock import SpeakerLockedIndicParler
from openvoice_postprocessor import OpenVoicePostProcessor

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "ai4bharat/indic-parler-tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "StoricoTTSDataset"
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
CAPTIONS_FILE = os.path.join(DATA_DIR, "style_captions.json")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# OpenVoice paths
OPENCV_CONFIG = "narrativetts/checkpoints_v2/converter/config.json"
OPENCV_CHECKPOINT = "narrativetts/checkpoints_v2/converter/checkpoint.pth"

# Output directory
OUTPUT_DIR = "audiobooks_hybrid"

# Default 5 stories for initial generation
DEFAULT_STORY_IDS = ["23", "44", "120", "8", "30"]

# All 10 stories in test.csv
ALL_STORY_IDS = ["8", "23", "30", "44", "50", "75", "93", "119", "120", "165"]

# Voice profile templates - using female narrator consistently
NARRATOR_TEMPLATE = "spoken by a female narrator, consistent storytelling voice"
NARRATOR_MULTI_TEMPLATE = "spoken by female narrator, clear storytelling voice"

# Audio stitching parameters
SEGMENT_GAP = 0.2  # 200ms silence between segments (natural speech pause)
CROSSFADE_DURATION = 0.003  # 3ms crossfade for same-speaker transitions (prevents clicks)

# Voice conversion parameters
VC_TAU = 0.8  # 80% conversion strength


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name=MODEL_NAME, device=DEVICE):
    """Load IndicParler-TTS model and tokenizers."""
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )

    print("Model loaded successfully!")
    print(f"Sampling rate: {model.config.sampling_rate} Hz")
    return model, tokenizer, description_tokenizer, device


# ============================================================================
# Data Loading
# ============================================================================

def load_segments(story_ids):
    """Load segments from test.csv for specified stories."""
    segments_by_story = {story_id: [] for story_id in story_ids}

    with open(TEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            story = row['story']
            if story in story_ids:
                segments_by_story[story].append({
                    'segment_id': os.path.basename(row['audio_filepath']).replace('.wav', ''),
                    'audio_filepath': row['audio_filepath'],
                    'character': row.get('character', '').strip(),
                    'age': row.get('age', '').strip(),
                    'gender': row.get('gender', '').strip(),
                    'text': row.get('norm', ''),  # Use 'norm' column for punctuation
                    'story': story,
                    'speaker': row.get('speaker', ''),
                })

    # Sort each story's segments by segment number
    for story_id in segments_by_story:
        segments_by_story[story_id].sort(
            key=lambda x: int(x['segment_id'].split('_')[-1])
        )

    # Deduplicate segments with identical text
    for story_id in segments_by_story:
        seen_texts = set()
        deduplicated = []
        for seg in segments_by_story[story_id]:
            if seg['text'] in seen_texts:
                continue
            seen_texts.add(seg['text'])
            deduplicated.append(seg)
        
        original_count = len(segments_by_story[story_id])
        segments_by_story[story_id] = deduplicated
        deduplicated_count = len(segments_by_story[story_id])
        
        if deduplicated_count < original_count:
            print(f"Story {story_id}: Removed {original_count - deduplicated_count} duplicate segments")

    return segments_by_story


def load_style_captions():
    """Load GPT-4o generated style captions."""
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Caption Building
# ============================================================================

def build_caption_single_speaker(style_caption: str) -> str:
    """Build caption for single-speaker mode."""
    return f"{NARRATOR_TEMPLATE}, {style_caption}"


def build_caption_multi_speaker(
    character: str,
    age: str,
    gender: str,
    style_caption: str
) -> str:
    """Build caption for multi-speaker mode."""
    if character:
        age_str = age if age else "character"
        gender_str = gender if gender else "voice"
        voice_profile = f"spoken as {character}, a {age_str} character, {gender_str} voice"
        return f"{voice_profile}, {style_caption}"
    else:
        return f"{NARRATOR_MULTI_TEMPLATE}, {style_caption}"


# ============================================================================
# Speech Generation (IndicParler with Speaker Lock)
# ============================================================================

def generate_speech_segment(
    locked_tts: SpeakerLockedIndicParler,
    device,
    text: str,
    style_caption: str,
    speaker_id: str,
    reset_memory: bool = False,
) -> np.ndarray:
    """Generate speech for a single segment using speaker-locked IndicParler."""
    
    # Extract speaker embedding on first segment
    if reset_memory or not locked_tts.is_speaker_registered(speaker_id):
        locked_tts.extract_speaker_embedding(style_caption, speaker_id)
    
    # Generate with locked embedding
    audio = locked_tts.generate_with_lock(text, speaker_id)
    
    return audio


# ============================================================================
# Audio Stitching with Crossfade
# ============================================================================

def stitch_audio_segments_with_crossfade(
    audio_segments: list,
    character_names: list,  # Added: track which speaker each segment belongs to
    sample_rate: int,
    gap_duration: float = SEGMENT_GAP,
    crossfade_duration: float = CROSSFADE_DURATION
) -> np.ndarray:
    """
    Stitch audio segments together with smart crossfade.
    
    Crossfade is only applied when adjacent segments have the SAME speaker,
    preventing voice bleeding between different speakers.
    """
    if not audio_segments:
        return np.array([])
    
    # Filter out empty segments (keep track of indices)
    valid_segments = []
    valid_characters = []
    for seg, char in zip(audio_segments, character_names):
        if len(seg) > 0:
            valid_segments.append(seg)
            valid_characters.append(char)
    
    if not valid_segments:
        return np.array([])
    
    if len(valid_segments) == 1:
        return valid_segments[0]
    
    # Calculate gap and crossfade samples
    gap_samples = int(sample_rate * gap_duration)
    crossfade_samples = int(sample_rate * crossfade_duration)
    silence = np.zeros(gap_samples)
    
    # Concatenate with gaps and conditional crossfade
    stitched = valid_segments[0]
    
    for i, seg in enumerate(valid_segments[1:]):
        prev_char = valid_characters[i]
        curr_char = valid_characters[i + 1]
        
        # Add gap between all segments
        stitched = np.concatenate([stitched, silence])
        
        # Only apply crossfade if same speaker (prevents voice bleeding)
        if crossfade_samples > 0 and prev_char == curr_char:
            if len(stitched) > crossfade_samples and len(seg) > crossfade_samples:
                fade_out = np.linspace(1, 0, crossfade_samples)
                fade_in = np.linspace(0, 1, crossfade_samples)
                
                overlap = (
                    stitched[-crossfade_samples:] * fade_out + 
                    seg[:crossfade_samples] * fade_in
                )
                
                stitched = np.concatenate([
                    stitched[:-crossfade_samples],
                    overlap,
                    seg[crossfade_samples:]
                ])
            else:
                stitched = np.concatenate([stitched, seg])
        else:
            # Different speakers - hard cut (no crossfade)
            stitched = np.concatenate([stitched, seg])
    
    return stitched


# ============================================================================
# Story Processing
# ============================================================================

def process_story_hybrid(
    story_id: str,
    segments: list,
    style_captions: dict,
    locked_tts: SpeakerLockedIndicParler,
    vc_processor: Optional[OpenVoicePostProcessor],
    device,
    sample_rate: int,
    use_voice_conversion: bool = True,
) -> dict:
    """
    Process a single story with hybrid pipeline.
    
    Args:
        story_id: Story ID
        segments: List of segment dicts
        style_captions: Style captions dict
        locked_tts: Speaker-locked IndicParler
        vc_processor: OpenVoice post-processor (optional)
        device: Device
        sample_rate: Audio sample rate
        use_voice_conversion: If True, apply OpenVoice VC
    
    Returns:
        Dict with audio, transcript, metadata
    """
    print(f"\n{'='*70}")
    print(f"Story {story_id} - HYBRID PIPELINE")
    print(f"{'='*70}")
    print(f"Segments: {len(segments)}")
    print(f"Voice Conversion: {'ENABLED' if use_voice_conversion else 'DISABLED'}")
    
    audio_segments = []
    character_names = []
    transcripts = []
    
    # Determine if multi-speaker
    has_characters = any(seg.get('character') for seg in segments)
    print(f"Mode: {'Multi-speaker' if has_characters else 'Single-speaker'}")
    
    for i, seg in enumerate(tqdm(segments, desc=f"Story {story_id}", unit="seg")):
        segment_id = seg['segment_id']
        text = seg['text']
        character = seg.get('character', '')
        
        # Get style caption
        caption_data = style_captions.get(segment_id, {})
        style_caption = caption_data.get('style_caption', 'neutral narration')
        
        # Handle missing/error captions
        if not style_caption or style_caption.startswith('[ERROR]'):
            style_caption = 'neutral narration'
        elif style_caption.startswith('[EXISTING]'):
            style_caption = style_caption.replace('[EXISTING]', '').strip()
        
        # Build caption based on mode
        if character:
            full_caption = build_caption_multi_speaker(
                character, seg['age'], seg['gender'], style_caption
            )
            speaker_id = character
        else:
            full_caption = build_caption_single_speaker(style_caption)
            speaker_id = "narrator"
        
        # Generate with speaker-locked IndicParler
        reset_memory = (i == 0)
        try:
            audio = generate_speech_segment(
                locked_tts, device, text, full_caption, speaker_id, reset_memory=reset_memory
            )
            audio_segments.append(audio)
            character_names.append(character if character else "narrator")
            transcripts.append(text)
        except Exception as e:
            print(f"  ERROR generating {segment_id}: {e}")
    
    # Apply voice conversion if enabled
    if use_voice_conversion and vc_processor and audio_segments:
        print(f"\n🔄 Applying OpenVoice voice conversion...")
        audio_segments = vc_processor.convert_story(
            audio_segments=audio_segments,
            character_names=character_names,
            sample_rate=sample_rate,
            tau=VC_TAU
        )
    
    # Stitch with crossfade (smart: only for same-speaker transitions)
    if audio_segments:
        stitched_audio = stitch_audio_segments_with_crossfade(
            audio_segments, character_names, sample_rate
        )
    else:
        stitched_audio = np.array([])
    
    # Full transcript
    full_transcript = " ".join(transcripts)
    
    return {
        'story_id': story_id,
        'audio': stitched_audio,
        'transcript': full_transcript,
        'segment_count': len(segments),
        'character_names': character_names,
        'use_voice_conversion': use_voice_conversion,
    }


# ============================================================================
# Save Outputs
# ============================================================================

def save_story_outputs(result: dict, output_dir: str):
    """Save all outputs for a processed story."""
    os.makedirs(output_dir, exist_ok=True)
    
    story_id = result['story_id']
    
    # Save transcript
    transcript_path = os.path.join(output_dir, f"story{story_id}_hybrid_transcript.txt")
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(result['transcript'])
    print(f"  📄 Saved: {transcript_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"story{story_id}_hybrid_metadata.json")
    metadata = {
        'story_id': story_id,
        'segment_count': result['segment_count'],
        'character_names': result['character_names'],
        'use_voice_conversion': result['use_voice_conversion'],
        'vc_tau': VC_TAU,
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  📊 Saved: {metadata_path}")
    
    # Save audio
    audio_path = os.path.join(output_dir, f"story{story_id}_hybrid.wav")
    if len(result['audio']) > 0:
        sf.write(audio_path, result['audio'], result.get('sample_rate', 44100))
        print(f"  🎵 Saved: {audio_path}")
        duration = len(result['audio']) / result.get('sample_rate', 44100)
        print(f"     Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
    else:
        print(f"  ⚠️ Warning: No audio generated for {story_id}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(
    story_ids: list, 
    output_dir: str = OUTPUT_DIR,
    use_voice_conversion: bool = True,
):
    """Run the complete hybrid audiobook generation pipeline."""
    
    print("="*70)
    print("Hybrid Audiobook Generation Pipeline")
    print("Speaker Embedding Lock + OpenVoice Voice Conversion")
    print("="*70)
    print(f"\nStories to process: {story_ids}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    print(f"Voice Conversion: {'ENABLED' if use_voice_conversion else 'DISABLED'}")
    
    # Load IndicParler model
    model, tokenizer, description_tokenizer, device = load_model()
    sample_rate = model.config.sampling_rate
    
    # Initialize speaker-locked TTS
    print(f"\n🔒 Initializing Speaker-Locked IndicParler...")
    locked_tts = SpeakerLockedIndicParler(model, tokenizer, description_tokenizer, device)
    
    # Initialize OpenVoice if enabled
    vc_processor = None
    if use_voice_conversion:
        print(f"\n🎤 Initializing OpenVoice PostProcessor...")
        try:
            vc_processor = OpenVoicePostProcessor(
                config_path=OPENCV_CONFIG,
                checkpoint_path=OPENCV_CHECKPOINT,
                device=device
            )
        except Exception as e:
            print(f"⚠️ Could not load OpenVoice: {e}")
            print(f"   Continuing without voice conversion...")
            use_voice_conversion = False
            vc_processor = None
    
    # Load data
    print(f"\n📂 Loading data...")
    segments_by_story = load_segments(story_ids)
    style_captions = load_style_captions()
    print(f"Style captions loaded: {len(style_captions)}")
    
    # Process each story
    for story_id in story_ids:
        segments = segments_by_story.get(story_id, [])
        
        if not segments:
            print(f"\n⚠️ Warning: No segments found for story {story_id}")
            continue
        
        print(f"\n{'#'*70}")
        print(f"# STORY {story_id}")
        print(f"{'#'*70}")
        print(f"Total segments: {len(segments)}")
        
        # Auto-register characters from this story if using VC
        if use_voice_conversion and vc_processor:
            print(f"\n🎭 Auto-registering characters from Story {story_id}...")
            try:
                registered = vc_processor.get_library().auto_register_from_story(story_id)
                print(f"✓ Registered {len(registered)} characters: {', '.join(registered.keys())}")
            except Exception as e:
                print(f"⚠️ Could not auto-register characters: {e}")
        
        # Process story
        result = process_story_hybrid(
            story_id=story_id,
            segments=segments,
            style_captions=style_captions,
            locked_tts=locked_tts,
            vc_processor=vc_processor,
            device=device,
            sample_rate=sample_rate,
            use_voice_conversion=use_voice_conversion,
        )
        result['sample_rate'] = sample_rate
        
        # Save outputs
        save_story_outputs(result, output_dir)
        
        # Clear speaker cache for next story
        locked_tts.clear_cache()
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}/")
    print("\nFiles per story:")
    print("  - story{ID}_hybrid_transcript.txt")
    print("  - story{ID}_hybrid_metadata.json")
    print("  - story{ID}_hybrid.wav")
    print("\nCompare with:")
    print("  - audiobooks/story{ID}_single.wav (original, no memory)")
    print("  - audiobooks_memory/story{ID}_single.wav (memory only)")
    print("="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audiobooks with hybrid speaker consistency pipeline"
    )
    parser.add_argument(
        "--all-stories",
        action="store_true",
        help="Process all 10 stories (default: 5 stories)"
    )
    parser.add_argument(
        "--story_id",
        type=str,
        default=None,
        help="Process a specific story only (e.g., --story_id 23)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--no-vc",
        action="store_true",
        help="Disable voice conversion (use speaker embedding lock only)"
    )
    
    args = parser.parse_args()
    
    # Determine which stories to process
    if args.story_id:
        story_ids = [args.story_id]
    elif args.all_stories:
        story_ids = ALL_STORY_IDS
    else:
        story_ids = DEFAULT_STORY_IDS
    
    # Run pipeline
    use_vc = not args.no_vc
    run_pipeline(story_ids, args.output_dir, use_voice_conversion=use_vc)
