import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import AUDIO_DURATION, MAX_TEXT_LENGTH, load_indicvoices, load_rasa

class IndicVoicesCLAPDataset(Dataset):
    def __init__(self, dataset_name, split="train", target_sr=16000, max_audio_len=AUDIO_DURATION, max_text_len=MAX_TEXT_LENGTH, muril_model_name="google/muril-base-cased", gender="male", language="all", limit=None):
        if dataset_name == "indicvoices":
            self.dataset = load_indicvoices(split, gender, language)
        elif dataset_name == "rasa":
            self.dataset = load_rasa(split, gender, language)
        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))
        self.dataset_name = dataset_name
        self.split = split
        self.target_sr = target_sr
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        self.tokenizer = AutoTokenizer.from_pretrained(muril_model_name)
        
        # Audio length in samples (16.384s @ 16000Hz with 256 hop = 1024 frames)
        self.max_samples = int(target_sr * max_audio_len)
        self.cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "voiceclap" / f"{dataset_name}_{language}_{gender}_{split}_sr{target_sr}_seconds{AUDIO_DURATION}_txt{max_text_len}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.build_cache()

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        return text.strip()

    def cache_path(self, idx):
        return self.cache_dir / f"{idx:08d}.pt"

    def build_cache(self):
        for idx in tqdm(range(len(self.dataset)), desc=f"Caching {self.dataset_name} {self.split}"):
            path = self.cache_path(idx)
            if not path.exists():
                torch.save(self.process_item(idx), path)

    def process_item(self, idx):
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
            
        # 3. Process Text
        if self.dataset_name == "indicvoices":
            text = self.clean_text(item['normalized'])
        elif self.dataset_name == "rasa":
            text = self.clean_text(item['text'])
        tokens = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_text_len, 
            return_tensors="pt"
        )
        
        return {
            "mel_spec": audio_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0)
        }

    def __getitem__(self, idx):
        return torch.load(self.cache_path(idx), map_location="cpu", weights_only=False)

def get_dataloader(split="train", batch_size=128, num_workers=4, gender="male", language="all"):
    dataset = IndicVoicesCLAPDataset(dataset_name="rasa", split=split, gender=gender, language=language)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)
