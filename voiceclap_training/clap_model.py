import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
    def __init__(self, htsat_config, muril_model_name="google/muril-base-cased", projection_dim=768):
        super().__init__()
        
        self.audio_encoder = AutoModel.from_pretrained("laion/voiceclap-small-v2", trust_remote_code=True)
        self.audio_projection = ProjectionHead(768, projection_dim)
        
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        self.text_projection = ProjectionHead(768, projection_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def encode_audio(self, audio):
        """
        Input: (batch_size, samples)
        Output: (batch_size, projection_dim) normalized
        """
        # HTS-AT forward returns a dict with 'latent_output' if enable_tscam is True
        audio_features = self.audio_encoder.encode_waveform(audio)
        audio_features = self.audio_projection(audio_features)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
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

    def forward(self, audio, input_ids, attention_mask):
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()
        
        return logits_per_audio, logits_per_text

def contrastive_loss(logits_per_audio, logits_per_text):
    batch_size = logits_per_audio.shape[0]

    labels = torch.full_like(logits_per_audio, -1.0)
    labels.fill_diagonal_(1.0)

    loss_a = F.softplus(-(labels * logits_per_audio))
    loss_t = F.softplus(-(labels * logits_per_text))

    return (loss_a.mean() + loss_t.mean()) / 2