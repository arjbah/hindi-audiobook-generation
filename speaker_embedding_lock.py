"""
Speaker Embedding Lock for IndicParler-TTS

Extracts and reuses speaker embeddings to maintain voice consistency
across segments without requiring voice conversion.

Usage:
    locked_tts = SpeakerLockedIndicParler(model, tokenizer, description_tokenizer, device)
    
    # Extract embedding from first segment
    locked_tts.extract_speaker_embedding(
        "spoken by a female narrator, warm voice",
        speaker_id="narrator"
    )
    
    # Generate all segments with locked embedding
    for segment in segments:
        audio = locked_tts.generate_with_lock(
            text=segment.text,
            speaker_id="narrator"
        )
"""

import torch
import numpy as np
from typing import Dict, Optional


class SpeakerLockedIndicParler:
    """
    Modifies IndicParler to support speaker embedding locking.
    
    This bypasses the style encoder for segments 2+, reusing the
    embedding from segment 1 to maintain consistent speaker identity.
    
    Architecture:
        Standard IndicParler:
            Caption → Flan T5 → Style Embedding → Decoder → Audio
            (New embedding per segment → voice drift)
        
        Speaker-Locked IndicParler:
            Caption 1 → Flan T5 → Style Embedding → Cache
            Caption 2 → [BYPASS] → Use Cached Embedding → Decoder → Audio
            (Same embedding → consistent voice)
    """
    
    def __init__(self, model, tokenizer, description_tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.description_tokenizer = description_tokenizer
        self.device = device
        
        # Cache of speaker embeddings: {speaker_id: embedding_tensor}
        self.speaker_embeddings: Dict[str, torch.Tensor] = {}
    
    def extract_speaker_embedding(
        self, 
        style_caption: str, 
        speaker_id: str = "default"
    ) -> torch.Tensor:
        """
        Extract and cache style embedding from a caption.
        
        This becomes the "locked" speaker identity for this speaker.
        
        Args:
            style_caption: Style description text
            speaker_id: Identifier for this speaker (e.g., "narrator", "character_1")
        
        Returns:
            Speaker embedding tensor
        """
        inputs = self.description_tokenizer(
            style_caption, 
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            encoder_outputs = self.model.text_encoder(**inputs)
        
        # Cache embedding: shape (1, seq_len, 1024)
        self.speaker_embeddings[speaker_id] = encoder_outputs.last_hidden_state
        
        print(f"✓ Extracted speaker embedding for '{speaker_id}': {encoder_outputs.last_hidden_state.shape}")
        return encoder_outputs.last_hidden_state
    
    def generate_with_lock(
        self,
        text: str,
        speaker_id: str = "default",
        **generate_kwargs
    ) -> np.ndarray:
        """
        Generate audio using a PRE-EXTRACTED speaker embedding.
        
        This bypasses the style encoder, ensuring consistent voice identity.
        
        Args:
            text: Hindi text to synthesize
            speaker_id: Which cached speaker embedding to use
            **generate_kwargs: Additional args for model.generate()
        
        Returns:
            Audio array
        """
        if speaker_id not in self.speaker_embeddings:
            raise ValueError(
                f"Speaker '{speaker_id}' not registered. "
                f"Call extract_speaker_embedding() first."
            )
        
        # Get cached embedding
        speaker_embedding = self.speaker_embeddings[speaker_id]
        
        # Tokenize text
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Use the full model's generate with inputs_embeds
        # The model's generate() returns audio waveform directly (already decoded)
        with torch.no_grad():
            audio_output = self.model.generate(
                inputs_embeds=speaker_embedding,  # Pass embedding directly
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
                **generate_kwargs
            )
        
        # Convert to audio array (same as original generate_audiobook.py)
        audio_arr = audio_output.cpu().numpy().squeeze()
        
        # Handle empty/invalid generation
        if audio_arr.ndim == 0 or audio_arr.size == 0:
            return np.array([])
        
        # Ensure 1D array for mono audio
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=0)
        
        return audio_arr
    
    def is_speaker_registered(self, speaker_id: str) -> bool:
        """Check if a speaker embedding is cached."""
        return speaker_id in self.speaker_embeddings
    
    def get_registered_speakers(self) -> list:
        """List all registered speaker IDs."""
        return list(self.speaker_embeddings.keys())
    
    def clear_cache(self):
        """Clear all cached speaker embeddings."""
        self.speaker_embeddings.clear()
