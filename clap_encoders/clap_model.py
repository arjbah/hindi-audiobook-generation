import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio.functional as AF
from transformers import AutoModel, AutoTokenizer

# Add project root and hts_at_repo to sys.path to handle imports correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
hts_at_path = os.path.join(project_root, "hts_at_repo")

for p in [project_root, hts_at_path]:
    if p not in sys.path:
        sys.path.append(p)

from hts_at_repo.model.htsat import HTSAT_Swin_Transformer

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
    def __init__(self, htsat_config, muril_model_name="google/muril-base-cased", projection_dim=512):
        super().__init__()
        
        # 1. Audio Encoder (HTS-AT)
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
        
        # RESTORE MANUAL MEL LOGIC: Disable on-the-fly extractors
        self.audio_encoder.spectrogram_extractor = nn.Identity()
        self.audio_encoder.logmel_extractor = nn.Identity()
        
        # Setup manual Mel transforms (Matching clap_dataset.py)
        self.target_sr = 22050
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            f_min=50,
            f_max=11025,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=None)

        # Look for checkpoint in hts_at_repo or project root
        checkpoint_path = os.path.join(project_root, "HTSAT_AudioSet_Saved_6.ckpt")
        if not os.path.exists(checkpoint_path):
             checkpoint_path = os.path.join(project_root, "hts_at_repo", "HTSAT_AudioSet_Saved_6.ckpt")
             
        if os.path.exists(checkpoint_path):
            print(f"Loading HTS-AT backbone from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            new_checkpoint = {"state_dict": {}}

            for old_key in checkpoint["state_dict"].keys():
                new_key = old_key.replace("sed_model.", "")
                new_checkpoint["state_dict"][new_key] = checkpoint["state_dict"][old_key]
            
            # Load with strict=False because we replaced extractors with Identity
            self.audio_encoder.load_state_dict(new_checkpoint["state_dict"], strict=False)
        else:
            print("Warning: HTSAT_AudioSet_Saved_6.ckpt not found.")
        
        # HTS-AT outputs 768-D latent vector
        self.audio_projection = ProjectionHead(768, projection_dim)
        
        # 2. Text Encoder (MuRIL)
        self.text_encoder = AutoModel.from_pretrained(muril_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(muril_model_name)
        self.text_projection = ProjectionHead(768, projection_dim)
        
        # 3. Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def preprocess_audio(self, audio, sr):
        """
        Implements the manual Mel-spectrogram pipeline from clap_dataset.py
        Input: (batch_size, samples) raw waveform
        Output: (batch_size, 1, Time, Freq) Mel-spectrogram
        """
        # Resample to 22050
        if sr != self.target_sr:
            audio = AF.resample(audio, sr, self.target_sr)
            
        # Ensure correct length (11.89s = 262174 samples)
        max_samples = int(self.target_sr * 11.89)
        if audio.shape[-1] > max_samples:
            audio = audio[:, :max_samples]
        else:
            padding = max_samples - audio.shape[-1]
            audio = F.pad(audio, (0, padding))
            
        # Mel Transform
        mel_spec = self.mel_transform(audio) # (B, Freq, Time)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Ensure exactly 1024 frames (Time dimension)
        if mel_spec.shape[-1] > 1024:
            mel_spec = mel_spec[:, :, :1024]
        elif mel_spec.shape[-1] < 1024:
            padding = 1024 - mel_spec.shape[-1]
            mel_spec = F.pad(mel_spec, (0, padding))
            
        # Z-score Normalization per sample
        processed_mel = []
        for i in range(mel_spec.shape[0]):
            m = mel_spec[i]
            mean = m.mean()
            std = m.std()
            m = (m - mean) / (std + 1e-6)
            # Transpose to (Time, Freq) and add channel dim
            m = m.transpose(0, 1).unsqueeze(0) # (1, 1024, 64)
            processed_mel.append(m)
            
        return torch.stack(processed_mel) # (B, 1, 1024, 64)

    def encode_audio(self, audio, sr=22050):
        """
        Input: (batch_size, samples) raw waveform OR preprocessed mel
        """
        # If input is raw waveform (2D), preprocess it
        if audio.ndim == 2:
            audio = self.preprocess_audio(audio, sr).to(audio.device)
            
        # audio is now (B, 1, 1024, 64)
        output = self.audio_encoder(audio)
        audio_features = output['latent_output']
        audio_features = self.audio_projection(audio_features)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
        return audio_features

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def encode_text_string(self, text, device="cpu"):
        if isinstance(text, str):
            text = [text]
            
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            return self.encode_text(tokens["input_ids"], tokens["attention_mask"])

    def forward(self, audio, input_ids, attention_mask, sr=22050):
        audio_features = self.encode_audio(audio, sr)
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
