import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ClapAudioModelWithProjection
import math

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class CLAPModel(nn.Module):
    def __init__(self, muril_model_name="google/muril-base-cased", projection_dim=512):
        super().__init__()
        
        self.audio_encoder = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        self.text_projection = ProjectionHead(768, projection_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def clamp_logit_scale(self):
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))

    def encode_audio(self, input_features, is_longer):
        """
        Input: (batch_size, samples)
        Output: (batch_size, projection_dim) normalized
        """
        # HTS-AT forward returns a dict with 'latent_output' if enable_tscam is True
        outputs = self.audio_encoder(input_features=input_features, is_longer=is_longer)
        audio_features = outputs.audio_embeds
        audio_features = F.normalize(audio_features, dim=-1)

        return audio_features

    def encode_text(self, input_ids, attention_mask):
        """
        Input: input_ids, attention_mask
        Output: (batch_size, projection_dim) normalized
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        text_features = outputs.last_hidden_state[:, 0, :] # (batch_size, 768)
        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def forward(self, input_features, is_longer, input_ids, attention_mask):
        audio_features = self.encode_audio(input_features, is_longer)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()
        
        return logits_per_audio, logits_per_text

def contrastive_loss(logits_per_audio, logits_per_text):
    batch_size = logits_per_audio.shape[0]
    labels = torch.arange(batch_size, device=logits_per_audio.device)
    
    loss_a = F.cross_entropy(logits_per_audio, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_a + loss_t) / 2