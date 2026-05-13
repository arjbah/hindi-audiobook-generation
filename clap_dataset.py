import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

class IndicVoicesCLAPDataset(Dataset):
    def __init__(self, split="train", target_sr=22050, max_audio_len=11.89, max_text_len=128, muril_model_name="google/muril-base-cased"):
        self.dataset = load_dataset("ai4bharat/indicvoices_r", "Hindi", split=split)
        self.target_sr = target_sr
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        self.tokenizer = AutoTokenizer.from_pretrained(muril_model_name)
        
        # Audio length in samples (11.89s @ 22050Hz with 256 hop = 1024 frames)
        self.max_samples = int(target_sr * max_audio_len)
        
        # 1. Quality Filtering (SNR > 20, CER < 0.1)
        # We use metadata to filter out noisy or inaccurately transcribed samples
        print(f"Filtering dataset ({split})...")
        self.valid_indices = []
        total_count = len(self.dataset)
        
        # To avoid slow iteration over the whole dataset if it's large, we use select/filter or simple loop
        # Given it's a HuggingFace dataset, we can use .filter for efficiency
        def filter_fn(example):
            # Parse CER from string format "tensor(0.01)"
            cer_val = None
            if example['cer'] is not None:
                try:
                    cer_str = example['cer']
                    if "tensor(" in cer_str:
                        cer_val = float(cer_str.replace("tensor(", "").replace(")", ""))
                    else:
                        cer_val = float(cer_str)
                except (ValueError, TypeError):
                    cer_val = None

            return (example['snr'] is not None and example['snr'] > 20) and \
                   (cer_val is not None and cer_val < 0.1)
        
        filtered_ds = self.dataset.filter(filter_fn, desc="Applying Quality Filters")
        self.dataset = filtered_ds
        
        excluded_count = total_count - len(self.dataset)
        print(f"Filter Complete: Excluded {excluded_count} samples ({excluded_count/total_count:.1%}) due to low SNR/high CER.")
        
        # Mel Spectrogram parameters matching HTSATConfig
        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=64,
            f_min=50,
            f_max=11025,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=None)

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        return text.strip()

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. Process Audio
        audio_array = item['audio']['array']
        orig_sr = item['audio']['sampling_rate']
        
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Resample using stateless functional for efficiency
        if orig_sr != self.target_sr:
            audio_tensor = F.resample(audio_tensor, orig_sr, self.target_sr)
        
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor[0] # Take first channel if multi-channel

        # Pad or truncate to ensure exactly max_samples
        if audio_tensor.shape[0] > self.max_samples:
            audio_tensor = audio_tensor[:self.max_samples]
        else:
            padding = self.max_samples - audio_tensor.shape[0]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
            
        # 2. Generate Mel Spectrogram (Frequency-Domain Input)
        mel_spec = self.mel_transform(audio_tensor)
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Ensure exactly 1024 frames
        if mel_spec.shape[1] > 1024:
            mel_spec = mel_spec[:, :1024]
        elif mel_spec.shape[1] < 1024:
            padding = 1024 - mel_spec.shape[1]
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            
        # Z-score Normalization (Global for the sample)
        # Using a small epsilon to avoid division by zero
        mean = mel_spec.mean()
        std = mel_spec.std()
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        # HTS-AT expects (1, Time, Freq) for its reshape logic
        mel_spec = mel_spec.transpose(0, 1) # (Time, Freq)
        mel_spec = mel_spec.unsqueeze(0)    # (1, Time, Freq)

        # 3. Process Text
        text = self.clean_text(item['normalized'])
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_text_len, 
            return_tensors="pt"
        )
        
        return {
            "mel_spec": mel_spec,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

def get_dataloader(split="train", batch_size=32, num_workers=4):
    dataset = IndicVoicesCLAPDataset(split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)
