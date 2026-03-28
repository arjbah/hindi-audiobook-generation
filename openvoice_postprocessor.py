"""
OpenVoice Voice Conversion Post-Processor

Applies voice conversion to IndicParler outputs for perfect speaker consistency.
Works as "identity insurance" on top of speaker embedding lock.

Usage:
    vc = OpenVoicePostProcessor(
        config_path="narrativetts/checkpoints_v2/converter/config.json",
        checkpoint_path="narrativetts/checkpoints_v2/converter/checkpoint.pth"
    )
    
    # Register reference voices
    vc.library.auto_register_from_story("23")
    
    # Convert IndicParler output
    converted = vc.convert_segment(
        source_audio=indicparler_audio,
        character_name="narrator",
        tau=0.8
    )
"""

import os
import sys
import torch
import numpy as np
import tempfile
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional

# Add OpenVoice to path manually (no pip install required)
OPENCV_PATH = os.path.join(os.path.dirname(__file__), "OpenVoice-main")
if OPENCV_PATH not in sys.path:
    sys.path.insert(0, OPENCV_PATH)

from openvoice.api import ToneColorConverter
from character_voice_library import CharacterVoiceLibrary


class OpenVoicePostProcessor:
    """
    Applies voice conversion as "identity insurance" after IndicParler generation.
    
    Why Voice Conversion?
    - Speaker embedding lock reduces drift at the source
    - Voice conversion provides final polish, eliminating any remaining inconsistencies
    - Together: Near-perfect speaker consistency
    
    Architecture:
        IndicParler Output (drifting voice)
            ↓
        OpenVoice Content Encoder (extracts linguistic content)
            ↓
        Reference Speaker Embedding (fixed identity)
            ↓
        OpenVoice Decoder (reconstructs audio with fixed voice)
            ↓
        Consistent Output (stable speaker identity)
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: Optional[str] = None
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Loading OpenVoice ToneColorConverter on {device}...")
        
        # Convert to absolute paths
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_config_path = os.path.join(script_dir, config_path)
        abs_checkpoint_path = os.path.join(script_dir, checkpoint_path)
        
        print(f"   Config: {abs_config_path}")
        print(f"   Checkpoint: {abs_checkpoint_path}")
        
        # Initialize OpenVoice converter
        self.converter = ToneColorConverter(
            config_path=abs_config_path,
            device=device
        )
        self.converter.load_ckpt(abs_checkpoint_path)
        
        # Initialize character library
        self.library = CharacterVoiceLibrary(self.converter)
        
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        print(f"✓ OpenVoice PostProcessor ready")
        print(f"  Config: {config_path}")
        print(f"  Checkpoint: {checkpoint_path}")
    
    def convert_segment(
        self,
        source_audio: np.ndarray,
        character_name: str,
        sample_rate: int = 44100,  # IndicParler's sample rate
        tau: float = 0.8
    ) -> np.ndarray:
        """
        Convert a single segment to target character's voice.
        
        Args:
            source_audio: Audio from IndicParler
            character_name: Which reference voice to apply
            sample_rate: Audio sample rate
            tau: Conversion strength (0.0-1.0)
                 - 0.0 = no conversion (keep source voice)
                 - 1.0 = full conversion (may introduce artifacts)
                 - 0.8 = recommended balance
        
        Returns:
            Converted audio array
        """
        # Get target speaker embedding
        target_embedding = self.library.get_embedding(character_name)
        
        if target_embedding is None:
            # Fallback to narrator
            target_embedding = self.library.get_embedding("narrator")
        
        if target_embedding is None:
            print("⚠️ No reference voices registered! Returning unchanged audio")
            return source_audio
        
        # Save source audio to temp file (OpenVoice requires file path)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, source_audio, sample_rate)
        
        try:
            # Extract source speaker embedding from the audio itself
            # This is required for OpenVoice to work properly
            src_se = self.converter.extract_se([temp_path])
            
            # Convert voice
            # src_se=extracted from source audio
            # tgt_se=target_embedding locks to the reference voice
            converted = self.converter.convert(
                audio_src_path=temp_path,
                src_se=src_se,  # Extract from source audio
                tgt_se=target_embedding,  # Lock to reference voice
                tau=tau,  # Conversion strength
                output_path=None  # Return as numpy array
            )
            
            # OpenVoice returns audio at its internal sample rate (hps.data.sampling_rate)
            # We need to resample to match the source audio's sample rate
            opencv_sr = self.converter.hps.data.sampling_rate  # Usually 22050 or 16000
            if opencv_sr != sample_rate:
                import librosa
                converted = librosa.resample(converted, orig_sr=opencv_sr, target_sr=sample_rate)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return converted
    
    def convert_story(
        self,
        audio_segments: List[np.ndarray],
        character_names: List[str],
        sample_rate: int = 44100,
        tau: float = 0.8
    ) -> List[np.ndarray]:
        """
        Convert all segments of a story.
        
        Args:
            audio_segments: List of audio arrays from IndicParler
            character_names: List of character names for each segment
            sample_rate: Audio sample rate
            tau: Conversion strength
        
        Returns:
            List of converted audio arrays
        """
        print(f"\n🔄 Converting {len(audio_segments)} segments with OpenVoice...")
        print(f"   Conversion strength (tau): {tau}")
        
        converted_segments = []
        
        for i, (audio, char_name) in enumerate(zip(audio_segments, character_names)):
            converted = self.convert_segment(
                source_audio=audio,
                character_name=char_name,
                sample_rate=sample_rate,
                tau=tau
            )
            converted_segments.append(converted)
        
        print(f"✓ Voice conversion complete")
        return converted_segments
    
    def get_library(self) -> CharacterVoiceLibrary:
        """Get the character voice library for registration."""
        return self.library
