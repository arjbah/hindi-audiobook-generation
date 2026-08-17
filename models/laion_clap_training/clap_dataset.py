from pathlib import Path

import torch
import torchaudio.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ClapProcessor
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import AUDIO_DURATION, MAX_TEXT_LENGTH, load_indicvoices, load_rasa

class IndicVoicesCLAPDataset(Dataset):
    def __init__(self, dataset_name, split="train", target_sr=48000, gender="male", language="all", limit=None):
        if dataset_name == "indicvoices":
            self.dataset = load_indicvoices(split, gender, language)
        elif dataset_name == "rasa":
            self.dataset = load_rasa(split, gender, language)
        if limit:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        self.dataset_name = dataset_name
        self.split = split
        self.target_sr = target_sr
        self.max_text_len = MAX_TEXT_LENGTH
        self.muril_model_name = "google/muril-base-cased"
        self.clap_model_name = "laion/clap-htsat-fused"
        self.tokenizer = AutoTokenizer.from_pretrained(self.muril_model_name)
        self.audio_processor = ClapProcessor.from_pretrained(self.clap_model_name)

        cache_root = Path(__file__).resolve().parents[1] / ".cache" / "laion_clap"
        self.cache_dir = cache_root / f"{dataset_name}_{language}_{gender}_{split}_sr{target_sr}_seconds{AUDIO_DURATION}_txt{self.max_text_len}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.build_cache()

    def __len__(self):
        return len(self.dataset)

    def clean_text(self, text):
        return text.strip()

    def int16_to_float32_torch(self, x):
        return (x / 32767.0).type(torch.float32)

    def float32_to_int16_torch(self, x):
        x = torch.clamp(x, min=-1.0, max=1.0)
        return (x * 32767.0).type(torch.int16)

    def cache_path(self, idx):
        return self.cache_dir / f"{idx:08d}.pt"

    def build_cache(self):
        for idx in tqdm(range(len(self.dataset)), desc=f"Caching {self.dataset_name} {self.split}"):
            path = self.cache_path(idx)
            if not path.exists():
                torch.save(self.process_item(idx), path)

    def process_item(self, idx):
        item = self.dataset[idx]

        audio_array = item["audio"]["array"]
        orig_sr = item["audio"]["sampling_rate"]
        audio_tensor = torch.from_numpy(audio_array).float()

        if orig_sr != self.target_sr:
            audio_tensor = F.resample(audio_tensor, orig_sr, self.target_sr)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor[0]

        audio_tensor = self.int16_to_float32_torch(self.float32_to_int16_torch(audio_tensor))
        audio_inputs = self.audio_processor(
            audio=audio_tensor.numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt",
        )

        if self.dataset_name == "indicvoices":
            text = self.clean_text(item["normalized"])
        elif self.dataset_name == "rasa":
            text = self.clean_text(item["text"])

        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_text_len, return_tensors="pt")

        return {
            "input_features": audio_inputs["input_features"].squeeze(0),
            "is_longer": audio_inputs["is_longer"].squeeze(0),
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def __getitem__(self, idx):
        return torch.load(self.cache_path(idx), map_location="cpu")


def get_dataloader(split="train", batch_size=128, num_workers=4, gender="male", language="all"):
    dataset = IndicVoicesCLAPDataset(dataset_name="rasa", split=split, gender=gender, language=language)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
