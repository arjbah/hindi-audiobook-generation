

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import argparse

from clap_model import CLAPModel, contrastive_loss
from clap_dataset import get_dataloader
'''
past config
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
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000 # sr // 2
    
    classes_num = 527 # Default for HTS-AT, but we'll use latent_output
    enable_tscam = True
    htsat_attn_heatmap = False
    loss_type = "clip_bce" # For internal HTS-AT logic if needed
    enable_repeat_mode = False
'''

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize custom hybrid model
    # To use local checkpoint, change custom_audio_ckpt=None to custom filename (e.g., "hts_weights.ckpt")
    model = CLAPModel(
        muril_model_name="google/muril-base-cased",
        laion_model_name="laion/clap-htsat-unfused",
        custom_audio_ckpt=None 
    ).to(device)

    # Data Loader (uses  zero-change dataset structure at 48kHz)
    train_loader = get_dataloader(split="train", batch_size=128, num_workers=4)

    # Optimizer with learning rates
    optimizer = optim.AdamW([
        {'params': model.audio_encoder.parameters(), 'lr': 1e-5},    # Pretrained, unfrozen
        {'params': model.text_encoder.parameters(), 'lr': 1e-5},     # Pretrained, unfrozen
        {'params': model.audio_projection.parameters(), 'lr': 1e-4},    # Scratch Projection Head
        {'params': model.text_projection.parameters(), 'lr': 1e-4},     # Scratch Projection Head
        {'params': [model.logit_scale], 'lr': 1e-4}                     # Scratch Temperature scale
    ], weight_decay=0.01)

    num_epochs = 120
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training Loop 
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0
        for batch in pbar:
            # Unpack keys
            mel_spec = batch['mel_spec'].to(device) # Raw 1D Audio tensor
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_per_audio, logits_per_text = model(mel_spec, input_ids, attention_mask)
                loss = contrastive_loss(logits_per_audio, logits_per_text)
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

        total_loss /= len(train_loader)

        # Save checkpoint
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss,
            }, f"clap_checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
