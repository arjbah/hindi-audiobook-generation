import logging
import random
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from lightning import LightningDataModule

sys.path.append(str(Path(__file__).resolve().parents[3]))

from common.data import load_indicvoices, load_rasa

from src.tokenizer.base import Tokenizer, HuggingFaceTokenizer

log = logging.getLogger(__name__)

TARGET_SR = 32000
MAX_DURATION = 10.24  # seconds
MAX_SAMPLES = int(TARGET_SR * MAX_DURATION)


class RasaDataset(Dataset):
    def __init__(self, split: str = "train", gender: str = "both", limit: int | None = None):
        self.split = split
        self.dataset = load_rasa(split, gender, limit)
        log.info(f"Loaded Rasa Hindi {split} (gender={gender}): {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        item = self.dataset[idx]

        audio = torch.from_numpy(item["audio"]["array"]).float()
        sr = item["audio"]["sampling_rate"]

        if audio.ndim > 1:
            audio = audio[0]

        if sr != TARGET_SR:
            audio = AF.resample(audio, sr, TARGET_SR)

        if audio.shape[0] > MAX_SAMPLES:
            start = random.randint(0, audio.shape[0] - MAX_SAMPLES) if self.split == "train" else 0
            audio = audio[start: start + MAX_SAMPLES]
        else:
            audio = F.pad(audio, (0, MAX_SAMPLES - audio.shape[0]))

        text = item["text"].strip()
        return audio, text


def _collate(batch):
    audios, texts = zip(*batch)
    return list(audios), list(texts)

class IndicVoicesDataset(Dataset):
    def __init__(self, split: str = "train", limit: int | None = None):
        self.split = split
        # Never gender-filtered: the fixed out-of-domain probe.
        self.dataset = load_indicvoices(split, limit)
        log.info(f"Loaded IndicVoices Hindi {split}: {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        item = self.dataset[idx]

        audio = torch.from_numpy(item["audio"]["array"]).float()
        sr = item["audio"]["sampling_rate"]

        if audio.ndim > 1:
            audio = audio[0]

        if sr != TARGET_SR:
            audio = AF.resample(audio, sr, TARGET_SR)

        if audio.shape[0] > MAX_SAMPLES:
            start = random.randint(0, audio.shape[0] - MAX_SAMPLES) if self.split == "train" else 0
            audio = audio[start: start + MAX_SAMPLES]
        else:
            audio = F.pad(audio, (0, MAX_SAMPLES - audio.shape[0]))

        text = item["normalized"].strip()
        return audio, text

class RasaDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        dataloader_kwargs: dict,
        gender: str = "both",
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.gender = gender
        self.limit = limit

        devices = dataloader_kwargs.pop("devices", 1)
        if not isinstance(devices, int):
            devices = len(devices)
        batch_size = dataloader_kwargs.pop("batch_size", 32) // devices

        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader_kwargs["batch_size"] = batch_size

        self.train_dataset = None
        self.val_dataset_1 = None
        self.val_dataset_2 = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        if self.train_dataset is not None:
            return
        self.train_dataset = RasaDataset(split="train", gender=self.gender, limit=self.limit)
        self.val_dataset_1 = RasaDataset(split="test", gender=self.gender, limit=self.limit)
        # Evaluate retrieval on the IndicVoices test split during validation too.
        self.val_dataset_2 = IndicVoicesDataset(split="test", limit=self.limit)
        self.test_dataset = RasaDataset(split="test", gender=self.gender, limit=self.limit)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=_collate,
            drop_last=True,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return [DataLoader(
            self.val_dataset_1,
            shuffle=False,
            collate_fn=_collate,
            **self.dataloader_kwargs,
        ), DataLoader(
            self.val_dataset_2,
            shuffle=False,
            collate_fn=_collate,
            **self.dataloader_kwargs,
        )]

    def test_dataloader(self):
        return [DataLoader(
            self.test_dataset,
            shuffle=False,
            collate_fn=_collate,
            **self.dataloader_kwargs,
        )]

    def on_before_batch_transfer(
        self,
        batch: Tuple[List[torch.Tensor], List[str]],
        dataloader_idx: int,
    ) -> Tuple[torch.Tensor, Any, List[str]]:
        audios, descriptions = batch

        audio_batch = pad_sequence(
            [a for a in audios],
            batch_first=True,
            padding_value=0.0,
        )

        text_tokens = self.tokenizer(descriptions) if self.tokenizer is not None else torch.zeros(1)
        return audio_batch, text_tokens, descriptions
