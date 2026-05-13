import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse

from clap_model import CLAPModel, contrastive_loss
from clap_dataset import get_dataloader

class HTSATConfig:
    # HTS-AT hyperparamaters as per CLAP Plan and HTS-AT defaults
    htsat_window_size = 8
    htsat_spec_size = 256
    htsat_patch_size = 4 
    htsat_stride = (4, 4)
    htsat_num_head = [4, 8, 16, 32]
    htsat_dim = 96 
    htsat_depth = [2, 2, 6, 2]
    
    # Signal processing parameters from CLAP Plan.md
    sample_rate = 22050
    window_size = 1024
    hop_size = 256
    mel_bins = 64
    fmin = 50
    fmax = 11025 # sr // 2
    
    classes_num = 527 # Default for HTS-AT, but we'll use latent_output
    enable_tscam = True
    htsat_attn_heatmap = False
    loss_type = "clip_bce" # For internal HTS-AT logic if needed
    enable_repeat_mode = False

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Config and Model
    config = HTSATConfig()
    model = CLAPModel(config).to(device)

    # 2. Data
    train_loader = get_dataloader(split="train", batch_size=24, num_workers=4)
    # val_loader = get_dataloader(split="validation", batch_size=24, num_workers=4)

    # 3. Optimizer and Scheduler
    # Lower LR for backbones, higher for projection heads
    optimizer = optim.AdamW([
        {'params': model.audio_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 1e-5},
        {'params': model.audio_projection.parameters(), 'lr': 1e-4},
        {'params': model.text_projection.parameters(), 'lr': 1e-4},
        {'params': [model.logit_scale], 'lr': 1e-4}
    ], weight_decay=0.01)

    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            mel_spec = batch['mel_spec'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            
            logits_per_audio, logits_per_text = model(mel_spec, input_ids, attention_mask)
            loss = contrastive_loss(logits_per_audio, logits_per_text)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"clap_checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
