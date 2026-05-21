import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

class IndicVoicesCLAPDataset(Dataset):
    def __init__(self, dataset_name, split="train", target_sr=32000, max_audio_len=10.24, max_text_len=128, muril_model_name="google/muril-base-cased"):
        if dataset_name == "indicvoices":
            self.dataset = load_dataset("ai4bharat/indicvoices_r", "Hindi", split=split)
        elif dataset_name == "rasa":
            self.dataset = load_dataset("ai4bharat/Rasa", "Hindi", split=split)
        self.dataset_name = dataset_name
        self.target_sr = target_sr
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        
        self.tokenizer = AutoTokenizer.from_pretrained(muril_model_name)
        
        # Audio length in samples (11.89s @ 22050Hz with 256 hop = 1024 frames)
        self.max_samples = int(target_sr * max_audio_len)

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

def get_dataloader(split="train", batch_size=32, num_workers=4):
    indicvoices = IndicVoicesCLAPDataset(dataset_name="indicvoices", split=split)
    rasa = IndicVoicesCLAPDataset(dataset_name="rasa", split=split)
    dataset = ConcatDataset([indicvoices, rasa])
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers)

def get_full_dataset():
    indicvoices_train = IndicVoicesCLAPDataset(dataset_name="indicvoices", split="train")
    indicvoices_test = IndicVoicesCLAPDataset(dataset_name="indicvoices", split="test")
    rasa_train = IndicVoicesCLAPDataset(dataset_name="rasa", split="train")
    rasa_test = IndicVoicesCLAPDataset(dataset_name="rasa", split="test")
    dataset = ConcatDataset([indicvoices_train, indicvoices_test, rasa_train, rasa_test])
    return dataset