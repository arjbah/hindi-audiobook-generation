import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ClapAudioModel, ClapFeatureExtractor

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
    def __init__(self, htsat_config=None, muril_model_name="google/muril-base-cased", laion_model_name="laion/clap-htsat-unfused", projection_dim=512, custom_audio_ckpt=None):
        super().__init__()
        
        # Text Encoder: Google MuRIL (Downloaded automatically from HF)
        print(f"Loading Text Encoder: {muril_model_name}")
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        self.text_projection = ProjectionHead(768, projection_dim)
        
        # Audio Encoder: LAION-CLAP HTS-AT (Downloaded automatically from HF)
        print(f"Loading Audio Encoder: {laion_model_name}")
        self.audio_encoder = ClapAudioModel.from_pretrained(laion_model_name)
        self.audio_projection = ProjectionHead(768, projection_dim)
        
        # Local checkpoint override (If using .chkpt)
        if custom_audio_ckpt is not None:
            print(f"Overriding audio encoder with local checkpoint: {custom_audio_ckpt}")
            checkpoint = torch.load(custom_audio_ckpt, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Clean common state_dict wrappers/prefixes
            clean_state_dict = {}
            for k, v in state_dict.items():
                clean_k = k.replace("module.", "").replace("sed_model.", "")
                clean_state_dict[clean_k] = v
                
            msg = self.audio_encoder.load_state_dict(clean_state_dict, strict=False)
            print(f"Loaded checkpoint. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        
        # LAION-CLAP Feature Extractor (Handles DSP internally on the GPU)
        self.feature_extractor = ClapFeatureExtractor.from_pretrained(laion_model_name)
        
        # Temperature Parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        # Unfreeze all parameters 
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

    def encode_audio(self, audio_waveforms):
        """
        Input: (batch_size, samples) - Raw 1D Audio Waveforms
        Output: (batch_size, projection_dim) L2 Normalized Embeddings
        """
        # Convert batched raw audio waveforms from GPU to CPU numpy list for HF Feature Extractor
        waveforms_list = [w.detach().cpu().numpy() for w in audio_waveforms]
        
        # Extract features (mel spectrograms and length flags) using the LAION config
        extracted_features = self.feature_extractor(
            waveforms_list, 
            sampling_rate=48000, 
            return_tensors="pt"
        )
        
        # Move extracted features to the GPU
        input_features = extracted_features["input_features"].to(audio_waveforms.device)
        is_longer = extracted_features["is_longer"].to(audio_waveforms.device)
        
        # Forward pass through pre-trained unfrozen HTS-AT
        outputs = self.audio_encoder(input_features=input_features, is_longer=is_longer)
        audio_features = outputs.pooler_output
        audio_features = self.audio_projection(audio_features)
        
        return F.normalize(audio_features, p=2, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        """
        Input: input_ids, attention_mask
        Output: (batch_size, projection_dim) L2 Normalized Embeddings
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        return F.normalize(text_features, p=2, dim=-1)

    def forward(self, audio, input_ids, attention_mask):
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(input_ids, attention_mask)
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
