#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
from pathlib import Path
import torch
import random
import time
import librosa
import torchlibrosa
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler, BatchSampler
from data_handling.sampler import BySequenceLengthSampler, BySequenceBatchSampler
from data_handling.text_transform import text_preprocess
import torch.nn.functional as F
import torchaudio.transforms as T
from datasets import load_dataset, concatenate_datasets
from common.data import filter_by_gender
from torch.utils.data import ConcatDataset
from tqdm import tqdm


HF_CACHE_DIR = Path("/mnt/huggingface")
DATA_CACHE_DIR = Path("/mnt/mga_clap_training_cache")

class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, audio_config, dataset_name, language, split=None,
                 gender="male", limit=None):
        if dataset_name == "indicvoices":
            dataset_split = "train"
            self.dataset = load_dataset(
                "ai4bharat/indicvoices_r",
                language,
                split=dataset_split,
                cache_dir=str(HF_CACHE_DIR),
            )
            # IndicVoices stays male-only and capped regardless of --gender:
            # this is the eval set the existing MGA numbers were measured on.
            self.dataset = self.dataset.filter(
                lambda item: item["gender"] == "Male",
                desc=f"Filtering {language} IndicVoices to Male",
            )
            self.dataset = self.dataset.select(
                range(min(2000, len(self.dataset)))
            )
            gender = "male"
        elif dataset_name == "rasa":
            dataset_split = split
            self.dataset = load_dataset(
                "ai4bharat/Rasa",
                language,
                split=dataset_split,
                cache_dir=str(HF_CACHE_DIR),
            )
            self.dataset = filter_by_gender(
                self.dataset, gender, label=f"{language} Rasa {split}")

        if limit is not None:
            self.dataset = self.dataset.select(range(min(limit, len(self.dataset))))

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = int(audio_config["max_length"] * self.sr)
        else:
            self.max_length = 0
        self.dataset_name = dataset_name
        self.language = language
        # Gender belongs in the cache key: entries are named by row index, so a
        # male-filtered cache would silently serve wrong audio to a female run.
        self.cache_dir = (
            DATA_CACHE_DIR
            / f"{dataset_name}_{language}_{dataset_split}_sr{self.sr}_max{self.max_length}_gender-{gender}"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.build_cache()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return torch.load(self.cache_path(index), map_location="cpu")

    def cache_path(self, index):
        return self.cache_dir / f"{index:08d}.pt"

    def build_cache(self):
        for index in tqdm(
            range(len(self.dataset)),
            desc=f"Caching {self.dataset_name} {self.language}",
        ):
            path = self.cache_path(index)
            if not path.exists():
                torch.save(self.process_item(index), path)

    def process_item(self, index):
        item = self.dataset[index]
        
        audio_array = item['audio']['array']
        orig_sr = item['audio']['sampling_rate']
        
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Resample using stateless functional for efficiency
        if orig_sr != self.sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, self.sr)
        
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor[0] # Take first channel if multi-channel

        # Pad or truncate to ensure exactly max_samples
        if audio_tensor.shape[0] > self.max_length:
            audio_tensor = audio_tensor[:self.max_length]
        else:
            padding = self.max_length - audio_tensor.shape[0]
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

        if self.dataset_name == "indicvoices":
            text = self.clean_text(item['normalized'])
        elif self.dataset_name == "rasa":
            text = self.clean_text(item['text'])

        return audio_tensor, text, index
        # return duration, caption, audio_id

    def clean_text(self, text):
        return text.strip()

def collate_fn(batch):
    wav_list = []
    text_list = []
    audio_idx_list = []
    max_length = max([i[0].shape[-1] for i in batch])
    for waveform, text, audio_idx in batch:
        if waveform.shape[-1] < max_length:
            pad_length = max_length - waveform.shape[-1]
            waveform = F.pad(waveform, [0, pad_length], "constant", 0.0)
        wav_list.append(waveform)
        text_list.append(text)
        audio_idx_list.append(audio_idx)
    waveforms = torch.stack(wav_list, dim=0)
    audio_idx = torch.tensor(audio_idx_list).type(torch.long)
    return waveforms, text_list, audio_idx

def pretrain_dataloader(config,
                        bucket: bool = False,
                        bucket_boundaries: tuple = (5, 30, 6),
                        is_distributed: bool = False,
                        num_tasks: int = 0,
                        global_rank: int = 0,
                        split: str = "train",
                        language: str = "Hindi",
                        gender: str = "male",
                        limit=None):
    dataset = AudioLanguagePretrainDataset(
        config["audio_args"], "rasa", language, split=split, gender=gender, limit=limit
    )

    if split == "train":
        return DataLoader(
            dataset,
            batch_size=config["data_args"]["batch_size"],
            num_workers=config["data_args"]["num_workers"],
            pin_memory=True,
            sampler=None,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    elif split == "test":
        return DataLoader(dataset,
            batch_size=32,
            num_workers=8,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )

def indicvoices_dataloader(config,
                        bucket: bool = False,
                        bucket_boundaries: tuple = (5, 30, 6),
                        is_distributed: bool = False,
                        num_tasks: int = 0,
                        global_rank: int = 0,
                        language: str = "Hindi",
                        limit=None):
    dataset = AudioLanguagePretrainDataset(
        config["audio_args"], "indicvoices", language, limit=limit
    )
    return DataLoader(
        dataset,
        batch_size=config["data_args"]["batch_size"],
        num_workers=config["data_args"]["num_workers"],
        pin_memory=True,
        sampler=None,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
