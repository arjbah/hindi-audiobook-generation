#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import json
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
from torch.utils.data import ConcatDataset

class AudioLanguagePretrainDataset(Dataset):

    def __init__(self, audio_config, split, dataset_name):
        if dataset_name == "indicvoices":
            self.dataset = load_dataset("ai4bharat/indicvoices_r", "Hindi", split="test")
        elif dataset_name == "rasa":
            self.dataset = load_dataset("ai4bharat/Rasa", "Hindi", split=split)

        self.sr = audio_config["sr"]
        if audio_config["max_length"] != 0:
            self.max_length = int(audio_config["max_length"] * self.sr)
        else:
            self.max_length = 0
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
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
                        split: str = "train"):
    dataset = AudioLanguagePretrainDataset(config["audio_args"], split, "rasa")

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
                        global_rank: int = 0):
    dataset = AudioLanguagePretrainDataset(config["audio_args"], "", "indicvoices")
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