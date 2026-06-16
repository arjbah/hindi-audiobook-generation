import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, ClapAudioModel

from hts_repo.model.htsat import HTSAT_Swin_Transformer

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
    def __init__(self, config=None muril_model_name="google/muril-base-cased", laion_model_name="laion/clap-htsat-unfused", 
                 projection_dim=512, custom_audio_ckpt=None):
        super().__init__()
        #Text Encoder: Google MuRIL 
        print(f"Loading pre-trained Text Encoder from HF: {muril_model_name}")
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        
        #Audio Encoder: LAION-CLAP HTS-AT
        print(f"Loading pre-trained Audio Encoder from HF: {laion_model_name}")
        self.audio_encoder = ClapAudioModel.from_pretrained(laion_model_name)
        
        #[Optional] Loading HTS-AT from local .ckpt (in case HuggingFace doesn't work)
        if custom_audio_ckpt is not None:
            print(f"Overriding audio encoder with local checkpoint: {custom_audio_ckpt}")
            checkpoint = torch.load(custom_audio_ckpt, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.audio_encoder.load_state_dict(state_dict, strict=False)

        #Projection Heads, loaded from scratch
        print("Initializing Audio and Text Projection Heads from scratch...")
        self.audio_projection = ProjectionHead(768, projection_dim)
        self.text_projection = ProjectionHead(768, projection_dim)

        #Logit Scale (Temperature): Initialized from scratch 
        # (Using the standard 0.07 initial scale from CLAP/CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        #Keep all parameters fully unfrozen (requires_grad = True)
        self.unfreeze_all_parameters()

      def unfreeze_all_parameters(self):
              for param in self.text_encoder.parameters():
                  param.requires_grad = True
              for param in self.audio_encoder.parameters():
                  param.requires_grad = True
              for param in self.audio_projection.parameters():
                  param.requires_grad = True
              for param in self.text_projection.parameters():
                  param.requires_grad = True
              self.logit_scale.requires_grad = True
        
        '''
        archived CLAP_model.py code
        # 1. Audio Encoder (HTS-AT)
        # Using the HTSAT_Swin_Transformer from the cloned repo
        self.audio_encoder = HTSAT_Swin_Transformer(
            spec_size=htsat_config.htsat_spec_size,
            patch_size=htsat_config.htsat_patch_size,
            in_chans=1,
            num_classes=htsat_config.classes_num,
            window_size=htsat_config.htsat_window_size,
            config=htsat_config,
            depths=htsat_config.htsat_depth,
            embed_dim=htsat_config.htsat_dim,
            patch_stride=htsat_config.htsat_stride,
            num_heads=htsat_config.htsat_num_head,
        )
        checkpoint = torch.load("hts_repo/HTSAT_AudioSet_Saved_6.ckpt", map_location=torch.device("cpu"))
        new_checkpoint = {"state_dict": {}}

        for old_key in checkpoint["state_dict"].keys():
            new_key = old_key.replace("sed_model.", "")
            new_checkpoint["state_dict"][new_key] = checkpoint["state_dict"][old_key]
        self.audio_encoder.load_state_dict(new_checkpoint["state_dict"])
        
        # HTS-AT outputs 768-D latent vector (num_features = embed_dim * 2^(num_layers-1) = 96 * 8 = 768)
        self.audio_projection = ProjectionHead(768, projection_dim)
        
        # 2. Text Encoder (MuRIL)
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        # MuRIL base is a BERT-base architecture, outputs 768-D
        self.text_projection = ProjectionHead(768, projection_dim)
        
        # 3. Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        '''
      
    def encode_audio(self, input_features, is_longer):
      """
      Extract pre-trained LAION HTS-AT representations and map them via custom projection head
      """
      outputs = self.audio_encoder(input_features=input_features, is_longer=is_longer)
      # pooler_output has shape [batch_size, 768]
      audio_features = outputs.pooler_output
      audio_features = self.audio_projection(audio_features)
      return F.normalize(audio_features, p=2, dim=-1)
      
    def encode_text(self, input_ids, attention_mask):
        """
        Extract pre-trained Google MuRIL representations and map them via custom projection head
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, p=2, dim=-1)

    def forward(self, input_features, is_longer, input_ids, attention_mask):
        audio_features = self.encode_audio(input_features, is_longer)
        text_features = self.encode_text(input_ids, attention_mask)

        # Calculate cosine similarity scaled by the learnable temperature
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
