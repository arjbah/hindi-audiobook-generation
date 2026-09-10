#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

import sys
from pathlib import Path

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from data import MAX_TEXT_LENGTH

class TextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_encoder_args"]["type"])
        self.text_encoder = AutoModel.from_pretrained(config["text_encoder_args"]["type"], add_pooling_layer=False)

        if config["text_encoder_args"]["freeze"]:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = False

        self.text_width = 768

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, text):
        text_input = self.tokenizer(text,
                                    padding='longest',
                                    truncation=True,
                                    max_length=MAX_TEXT_LENGTH,
                                    return_tensors="pt",
                                    return_special_tokens_mask=True)
        text_input = {
            key: value.to(self.device, non_blocking=True)
            for key, value in text_input.items()
        }
        text_output = self.text_encoder(input_ids=text_input["input_ids"],
                                        attention_mask=text_input["attention_mask"])[0]
        return text_output, (1 - text_input["special_tokens_mask"][:, 1:]).contiguous()
