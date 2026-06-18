import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from clap_model import CLAPModel, contrastive_loss
from clap_dataset import IndicVoicesCLAPDataset, get_dataloader

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


def retrieval_metrics(similarity_matrix, ks=(1, 5, 10)):
    ranks = []
    for i in range(similarity_matrix.shape[0]):
        sorted_indices = torch.argsort(similarity_matrix[i], descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks)
    metrics = {}
    for k in ks:
        metrics[f"R@{k}"] = (ranks <= k).float().mean().item()
    metrics["MedianRank"] = ranks.median().item()
    metrics["MeanRank"] = ranks.float().mean().item()
    metrics["mAP"] = sum(1.0 / rank.item() for rank in ranks) / len(ranks)
    return metrics


@torch.no_grad()
def evaluate_rasa(model, dataloader, device):
    model.eval()
    all_audio_embs = []
    all_text_embs = []

    for batch in tqdm(dataloader, desc="Rasa eval", leave=False):
        mel_spec = batch["mel_spec"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            audio_embs = model.encode_audio(mel_spec)
            text_embs = model.encode_text(input_ids, attention_mask)

        all_audio_embs.append(audio_embs.cpu())
        all_text_embs.append(text_embs.cpu())

    all_audio_embs = torch.cat(all_audio_embs)
    all_text_embs = torch.cat(all_text_embs)
    similarity_a2t = model.logit_scale.exp().cpu() * (all_audio_embs @ all_text_embs.T)

    metrics_a2t = retrieval_metrics(similarity_a2t)
    metrics_t2a = retrieval_metrics(similarity_a2t.T)
    return metrics_t2a, metrics_a2t

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Config and Model
    config = HTSATConfig()
    model = CLAPModel(config).to(device)

    # 2. Data
    train_loader = get_dataloader(split="train", batch_size=128, num_workers=4)
    rasa_eval_dataset = IndicVoicesCLAPDataset(dataset_name="rasa", split="test")
    rasa_eval_loader = DataLoader(rasa_eval_dataset, batch_size=512, shuffle=False, num_workers=4)

    # 3. Optimizer and Scheduler
    # Lower LR for backbones, higher for projection heads
    optimizer = optim.AdamW([
        {'params': model.audio_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 1e-5},
        {'params': model.audio_projection.parameters(), 'lr': 1e-4},
        {'params': model.text_projection.parameters(), 'lr': 1e-4},
        {'params': [model.logit_scale], 'lr': 1e-4}
    ], weight_decay=0.01)

    num_epochs = 120
    total_steps = len(train_loader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    #scaler = GradScaler()

    # 4. Training Loop
    best_rasa_r1 = -1.0
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0.0

        for batch in pbar:
            mel_spec = batch['mel_spec'].to(device)
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
        
        model.eval()
        metrics_t2a, metrics_a2t = evaluate_rasa(model, rasa_eval_loader, device)
        rasa_r1 = metrics_t2a["R@1"]
        print(
            f"Epoch {epoch+1}: Rasa T2A R@1={metrics_t2a['R@1']:.4f}, "
            f"A2T R@1={metrics_a2t['R@1']:.4f}, best={best_rasa_r1:.4f}"
        )

        if rasa_r1 > best_rasa_r1:
            best_rasa_r1 = rasa_r1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': total_loss,
                'rasa_t2a_metrics': metrics_t2a,
                'rasa_a2t_metrics': metrics_a2t,
                'best_rasa_r1': best_rasa_r1,
            }, "clap_checkpoint_best_rasa.pt")
            print(f"Saved new best Rasa checkpoint at epoch {epoch+1}.")

if __name__ == "__main__":
    train()
