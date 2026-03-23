"""
Generate Audiobook from StoricoTTSDataset using IndicParler-TTS

Produces both single-speaker and multi-speaker versions of each story.

Outputs per story:
- story{ID}_single_transcript.txt: Full stitched transcript (single-speaker)
- story{ID}_single_captions.json: Captions used for each segment (single-speaker)
- story{ID}_single.wav: Continuous audio (single-speaker)
- story{ID}_multi_transcript.txt: Full stitched transcript (multi-speaker)
- story{ID}_multi_captions.json: Captions used for each segment (multi-speaker)
- story{ID}_multi.wav: Continuous audio (multi-speaker)
- story{ID}_reference.wav: Stitched original audio from clips/

Usage:
    python generate_audiobook.py                    # Default: 5 stories
    python generate_audiobook.py --all-stories      # All 10 stories
    python generate_audiobook.py --story_id 23      # Specific story only

Requirements:
    - torch
    - transformers
    - parler_tts
    - soundfile
    - librosa
    - tqdm
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
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "ai4bharat/indic-parler-tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "StoricoTTSDataset"
CLIPS_DIR = os.path.join(DATA_DIR, "clips")
CAPTIONS_FILE = os.path.join(DATA_DIR, "style_captions.json")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
OUTPUT_DIR = "audiobooks"

# Default 5 stories for initial generation
DEFAULT_STORY_IDS = ["23", "44", "120", "8", "30"]

# All 10 stories in test.csv
ALL_STORY_IDS = ["8", "23", "30", "44", "50", "75", "93", "119", "120", "165"]

# Voice profile templates - using female narrator consistently
NARRATOR_TEMPLATE = "spoken by a female narrator, consistent storytelling voice"
NARRATOR_MULTI_TEMPLATE = "spoken by female narrator, clear storytelling voice"

# Crossfade/silence between segments (in seconds)
SEGMENT_GAP = 0.05  # 50ms silence


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
    """Load segments from test.csv for specified stories.
    
    Automatically deduplicates segments with identical text to handle
    dataset issues (e.g., Story 8 has repeated segments).
    """
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

    # Deduplicate segments with identical text (handles dataset issues)
    for story_id in segments_by_story:
        seen_texts = set()
        deduplicated = []
        for seg in segments_by_story[story_id]:
            # Skip segments with text we've already seen
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
    """
    Build caption for single-speaker mode.
    Prepends consistent narrator voice profile to existing caption.
    """
    return f"{NARRATOR_TEMPLATE}, {style_caption}"


def build_caption_multi_speaker(
    character: str,
    age: str,
    gender: str,
    style_caption: str
) -> str:
    """
    Build caption for multi-speaker mode.
    - Narrator segments: Use narrator template
    - Character segments: Prepend character-specific voice profile
    """
    if character:
        # Character dialogue - prepend voice profile
        age_str = age if age else "character"
        gender_str = gender if gender else "voice"
        voice_profile = f"spoken as {character}, a {age_str} character, {gender_str} voice"
        return f"{voice_profile}, {style_caption}"
    else:
        # Narrator segment
        return f"{NARRATOR_MULTI_TEMPLATE}, {style_caption}"


# ============================================================================
# Speech Generation
# ============================================================================

def generate_speech_segment(
    model,
    tokenizer,
    description_tokenizer,
    device,
    text: str,
    style_caption: str,
) -> np.ndarray:
    """
    Generate speech for a single segment using IndicParler-TTS.
    Returns audio array (not saved to file).
    """
    # Tokenize
    description_input_ids = description_tokenizer(
        style_caption, return_tensors="pt"
    ).to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
        )

    # Convert to audio array
    audio_arr = generation.cpu().numpy().squeeze()

    # Handle empty/invalid generation
    if audio_arr.ndim == 0 or audio_arr.size == 0:
        # Return empty 1D array instead of scalar
        return np.array([])

    # Ensure 1D array for mono audio
    if audio_arr.ndim > 1:
        audio_arr = audio_arr.mean(axis=0)

    return audio_arr


def stitch_audio_segments(
    audio_segments: list,
    sample_rate: int,
    gap_duration: float = SEGMENT_GAP
) -> np.ndarray:
    """
    Stitch audio segments together with small silence gap between them.
    """
    if not audio_segments:
        return np.array([])

    # Filter out empty segments
    valid_segments = [seg for seg in audio_segments if len(seg) > 0]

    if not valid_segments:
        return np.array([])

    # Calculate gap samples
    gap_samples = int(sample_rate * gap_duration)
    silence = np.zeros(gap_samples)

    # Concatenate with gaps
    stitched = valid_segments[0]
    for seg in valid_segments[1:]:
        stitched = np.concatenate([stitched, silence, seg])

    return stitched


def load_reference_audio(audio_filepath: str) -> np.ndarray:
    """Load original audio from clips folder."""
    # Convert workspace path to local path
    local_path = audio_filepath.replace('/workspace/WORK_DIR/en_output/clips/', f'{CLIPS_DIR}/')
    local_path = local_path.replace('\\', '/')

    if os.path.exists(local_path):
        audio, sr = sf.read(local_path)
        return audio, sr
    else:
        print(f"  Warning: Reference audio not found: {local_path}")
        return None, None


# ============================================================================
# Story Processing
# ============================================================================

def process_story_single_speaker(
    story_id: str,
    segments: list,
    style_captions: dict,
    model,
    tokenizer,
    description_tokenizer,
    device,
    sample_rate: int,
) -> dict:
    """
    Process a single story in single-speaker mode.
    All segments use the same narrator voice profile.
    """
    print(f"\n{'='*60}")
    print(f"Story {story_id} - SINGLE SPEAKER MODE")
    print(f"{'='*60}")
    print(f"Segments: {len(segments)}")

    audio_segments = []
    captions_used = []
    transcripts = []

    for seg in tqdm(segments, desc=f"Story {story_id} (single)", unit="seg"):
        segment_id = seg['segment_id']
        text = seg['text']

        # Get style caption
        caption_data = style_captions.get(segment_id, {})
        style_caption = caption_data.get('style_caption', 'neutral narration')

        # Handle missing/error captions
        if not style_caption or style_caption.startswith('[ERROR]'):
            style_caption = 'neutral narration'
        elif style_caption.startswith('[EXISTING]'):
            style_caption = style_caption.replace('[EXISTING]', '').strip()

        # Build caption with single-speaker narrator voice
        full_caption = build_caption_single_speaker(style_caption)

        # Generate audio
        try:
            audio_arr = generate_speech_segment(
                model, tokenizer, description_tokenizer, device, text, full_caption
            )
            audio_segments.append(audio_arr)
            captions_used.append({
                'segment_id': segment_id,
                'text': text,
                'caption_used': full_caption,
                'original_caption': style_caption,
                'character': seg['character'],
            })
            transcripts.append(text)
        except Exception as e:
            print(f"  ERROR generating {segment_id}: {e}")
            captions_used.append({
                'segment_id': segment_id,
                'text': text,
                'error': str(e),
            })

    # Stitch audio
    if audio_segments:
        stitched_audio = stitch_audio_segments(audio_segments, sample_rate)
    else:
        stitched_audio = np.array([])

    # Full transcript
    full_transcript = " ".join(transcripts)

    return {
        'story_id': story_id,
        'mode': 'single',
        'audio': stitched_audio,
        'transcript': full_transcript,
        'captions_used': captions_used,
        'segment_count': len(segments),
    }


def process_story_multi_speaker(
    story_id: str,
    segments: list,
    style_captions: dict,
    model,
    tokenizer,
    description_tokenizer,
    device,
    sample_rate: int,
) -> dict:
    """
    Process a single story in multi-speaker mode.
    Character segments get character-specific voice profiles.
    """
    print(f"\n{'='*60}")
    print(f"Story {story_id} - MULTI SPEAKER MODE")
    print(f"{'='*60}")
    print(f"Segments: {len(segments)}")

    audio_segments = []
    captions_used = []
    transcripts = []

    # Track character voice profiles for consistency
    character_profiles = {}

    for seg in tqdm(segments, desc=f"Story {story_id} (multi)", unit="seg"):
        segment_id = seg['segment_id']
        text = seg['text']
        character = seg['character']
        age = seg['age']
        gender = seg['gender']

        # Get style caption
        caption_data = style_captions.get(segment_id, {})
        style_caption = caption_data.get('style_caption', 'neutral narration')

        # Handle missing/error captions
        if not style_caption or style_caption.startswith('[ERROR]'):
            style_caption = 'neutral narration'
        elif style_caption.startswith('[EXISTING]'):
            style_caption = style_caption.replace('[EXISTING]', '').strip()

        # Build caption with multi-speaker voice profile
        full_caption = build_caption_multi_speaker(
            character, age, gender, style_caption
        )

        # Track character profile usage
        if character:
            profile_key = f"{character}_{age}_{gender}"
            if profile_key not in character_profiles:
                character_profiles[profile_key] = {
                    'character': character,
                    'age': age,
                    'gender': gender,
                    'segments': []
                }
            character_profiles[profile_key]['segments'].append(segment_id)

        # Generate audio
        try:
            audio_arr = generate_speech_segment(
                model, tokenizer, description_tokenizer, device, text, full_caption
            )
            audio_segments.append(audio_arr)
            captions_used.append({
                'segment_id': segment_id,
                'text': text,
                'caption_used': full_caption,
                'original_caption': style_caption,
                'character': character,
                'age': age,
                'gender': gender,
            })
            transcripts.append(text)
        except Exception as e:
            print(f"  ERROR generating {segment_id}: {e}")
            captions_used.append({
                'segment_id': segment_id,
                'text': text,
                'error': str(e),
            })

    # Stitch audio
    if audio_segments:
        stitched_audio = stitch_audio_segments(audio_segments, sample_rate)
    else:
        stitched_audio = np.array([])

    # Full transcript
    full_transcript = " ".join(transcripts)

    # Character profile summary
    print(f"\nCharacter profiles used:")
    for profile_key, profile_data in character_profiles.items():
        print(f"  {profile_data['character']} ({profile_data['age']}, {profile_data['gender']}): {len(profile_data['segments'])} segments")

    return {
        'story_id': story_id,
        'mode': 'multi',
        'audio': stitched_audio,
        'transcript': full_transcript,
        'captions_used': captions_used,
        'segment_count': len(segments),
        'character_profiles': character_profiles,
    }


def process_story_reference(
    story_id: str,
    segments: list,
    sample_rate: int,
) -> dict:
    """
    Stitch together original reference audio from clips folder.
    """
    print(f"\n{'='*60}")
    print(f"Story {story_id} - REFERENCE AUDIO")
    print(f"{'='*60}")

    audio_segments = []

    for seg in tqdm(segments, desc=f"Story {story_id} (ref)", unit="seg"):
        audio_filepath = seg['audio_filepath']
        audio, sr = load_reference_audio(audio_filepath)

        if audio is not None:
            # Resample if needed
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            audio_segments.append(audio)

    # Stitch audio
    if audio_segments:
        stitched_audio = stitch_audio_segments(audio_segments, sample_rate)
    else:
        stitched_audio = np.array([])

    return {
        'story_id': story_id,
        'mode': 'reference',
        'audio': stitched_audio,
        'segment_count': len(audio_segments),
    }


# ============================================================================
# Save Outputs
# ============================================================================

def save_story_outputs(result: dict, output_dir: str):
    """Save all outputs for a processed story."""
    os.makedirs(output_dir, exist_ok=True)

    story_id = result['story_id']
    mode = result['mode']

    # Save transcript (only for single/multi modes, not reference)
    if 'transcript' in result:
        transcript_path = os.path.join(output_dir, f"story{story_id}_{mode}_transcript.txt")
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(result['transcript'])
        print(f"  Saved: {transcript_path}")

    # Save captions used (only for single/multi modes, not reference)
    if 'captions_used' in result:
        captions_path = os.path.join(output_dir, f"story{story_id}_{mode}_captions.json")
        with open(captions_path, 'w', encoding='utf-8') as f:
            json.dump(result['captions_used'], f, indent=2, ensure_ascii=False)
        print(f"  Saved: {captions_path}")

    # Save audio
    audio_path = os.path.join(output_dir, f"story{story_id}_{mode}.wav")
    if len(result['audio']) > 0:
        sf.write(audio_path, result['audio'], result.get('sample_rate', 22050))
        print(f"  Saved: {audio_path}")
        print(f"    Duration: {len(result['audio']) / result.get('sample_rate', 22050):.2f} seconds")
    else:
        print(f"  Warning: No audio generated for {story_id}_{mode}")


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(story_ids: list, output_dir: str = OUTPUT_DIR):
    """Run the complete audiobook generation pipeline."""

    print("="*70)
    print("IndicParler-TTS Audiobook Generation")
    print("StoricoTTSDataset - Single & Multi Speaker")
    print("="*70)
    print(f"\nStories to process: {story_ids}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")

    # Load model
    model, tokenizer, description_tokenizer, device = load_model()
    sample_rate = model.config.sampling_rate

    # Load data
    print("\nLoading data...")
    segments_by_story = load_segments(story_ids)
    style_captions = load_style_captions()

    print(f"Style captions loaded: {len(style_captions)}")

    # Process each story
    for story_id in story_ids:
        segments = segments_by_story.get(story_id, [])

        if not segments:
            print(f"\nWarning: No segments found for story {story_id}")
            continue

        # Check if story is already complete (resume functionality)
        single_wav_path = os.path.join(output_dir, f"story{story_id}_single.wav")
        multi_wav_path = os.path.join(output_dir, f"story{story_id}_multi.wav")
        ref_wav_path = os.path.join(output_dir, f"story{story_id}_reference.wav")

        if (os.path.exists(single_wav_path) and 
            os.path.exists(multi_wav_path) and
            os.path.exists(ref_wav_path)):
            print(f"\n{'#'*70}")
            print(f"# STORY {story_id} - ALREADY COMPLETE, SKIPPING...")
            print(f"{'#'*70}")
            continue

        print(f"\n{'#'*70}")
        print(f"# STORY {story_id}")
        print(f"{'#'*70}")
        print(f"Total segments: {len(segments)}")

        # Count narrator vs character segments
        narrator_count = sum(1 for s in segments if not s['character'])
        character_count = len(segments) - narrator_count
        print(f"Narrator segments: {narrator_count}")
        print(f"Character segments: {character_count}")

        # Single-speaker mode
        result_single = process_story_single_speaker(
            story_id, segments, style_captions,
            model, tokenizer, description_tokenizer, device, sample_rate
        )
        result_single['sample_rate'] = sample_rate
        save_story_outputs(result_single, output_dir)

        # Multi-speaker mode
        result_multi = process_story_multi_speaker(
            story_id, segments, style_captions,
            model, tokenizer, description_tokenizer, device, sample_rate
        )
        result_multi['sample_rate'] = sample_rate
        save_story_outputs(result_multi, output_dir)

        # Reference audio (original clips stitched together)
        result_ref = process_story_reference(story_id, segments, sample_rate)
        result_ref['sample_rate'] = sample_rate
        save_story_outputs(result_ref, output_dir)

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}/")
    print("\nFiles per story:")
    print("  - story{ID}_single_transcript.txt")
    print("  - story{ID}_single_captions.json")
    print("  - story{ID}_single.wav")
    print("  - story{ID}_multi_transcript.txt")
    print("  - story{ID}_multi_captions.json")
    print("  - story{ID}_multi.wav")
    print("  - story{ID}_reference.wav")
    print("\n" + "="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate audiobooks from StoricoTTSDataset using IndicParler-TTS"
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

    args = parser.parse_args()

    # Determine which stories to process
    if args.story_id:
        story_ids = [args.story_id]
    elif args.all_stories:
        story_ids = ALL_STORY_IDS
    else:
        story_ids = DEFAULT_STORY_IDS

    run_pipeline(story_ids, args.output_dir)
