import contextlib
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from clap_model import CLAPModel, contrastive_loss
from clap_dataset import IndicVoicesCLAPDataset, get_dataloader
from common.data import add_shared_args, set_seed


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
        input_features = batch["input_features"].to(device)
        is_longer = batch["is_longer"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with amp(device):
            audio_embs = model.encode_audio(input_features, is_longer)
            text_embs = model.encode_text(input_ids, attention_mask)

        all_audio_embs.append(audio_embs.cpu())
        all_text_embs.append(text_embs.cpu())

    all_audio_embs = torch.cat(all_audio_embs)
    all_text_embs = torch.cat(all_text_embs)
    similarity_a2t = model.logit_scale.exp().cpu() * (all_audio_embs @ all_text_embs.T)

    metrics_a2t = retrieval_metrics(similarity_a2t)
    metrics_t2a = retrieval_metrics(similarity_a2t.T)
    return metrics_t2a, metrics_a2t

def amp(device):
    """bf16 autocast on CUDA, no-op elsewhere so CPU smoke tests run."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def setup_logging(log_output_dir):
    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('train_log.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
                filter=lambda record: record['extra']['indent'] == 1)
    return logger.bind(indent=1)


def train(args):
    output_dir = Path(args.output_dir) if args.output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    main_logger = setup_logging(output_dir)
    seed = set_seed(args.seed)
    main_logger.info(f"Run config | gender={args.gender}, seed={seed or 'unseeded'}")
    device = torch.device(args.device if getattr(args, "device", None) else ("cuda" if torch.cuda.is_available() else "cpu"))
    main_logger.info(f"Using device: {device}")

    # 1. Config and Model
    model = CLAPModel().to(device)
    main_logger.info(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    main_logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. Data
    batch_size = args.batch_size or 128
    train_loader = get_dataloader(split="train", batch_size=batch_size, num_workers=4, gender=args.gender, limit=args.limit)
    rasa_eval_dataset = IndicVoicesCLAPDataset(dataset_name="rasa", split="test", gender=args.gender, limit=args.limit)
    rasa_eval_loader = DataLoader(rasa_eval_dataset, batch_size=32, shuffle=False, num_workers=4)
    indicvoices_eval_dataset = IndicVoicesCLAPDataset(dataset_name="indicvoices", split="test", limit=args.limit)
    indicvoices_eval_loader = DataLoader(indicvoices_eval_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. Optimizer and Scheduler
    # Lower LR for backbones, higher for projection heads
    optimizer = optim.AdamW([
        {'params': model.audio_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_encoder.parameters(), 'lr': 1e-5},
        {'params': model.text_projection.parameters(), 'lr': 1e-4},
        {'params': [model.logit_scale], 'lr': 1e-4}
    ], weight_decay=0.01)

    num_epochs = args.epochs or 120
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
            input_features = batch['input_features'].to(device)
            is_longer = batch["is_longer"].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()

            with amp(device):
                logits_per_audio, logits_per_text = model(input_features, is_longer, input_ids, attention_mask)
                loss = contrastive_loss(logits_per_audio, logits_per_text)
            
            loss.backward()
            optimizer.step()
            model.clamp_logit_scale()
            scheduler.step()

            total_loss += loss.item()
            avg_loss = total_loss / (pbar.n + 1)
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})

        total_loss /= len(train_loader)
        
        main_logger.info(f"Epoch {epoch + 1} loss: {total_loss}")

        model.eval()
        metrics_t2a, metrics_a2t = evaluate_rasa(model, rasa_eval_loader, device)
        metrics_t2a_indicvoices, metrics_a2t_indicvoices = evaluate_rasa(model, indicvoices_eval_loader, device)
        rasa_r1 = metrics_t2a["R@1"]
        main_logger.info(
            f"Epoch [{epoch+1}] | Rasa metrics | "
            f"T2A: R@1={metrics_t2a['R@1']:.3f}, R@5={metrics_t2a['R@5']:.3f}, R@10={metrics_t2a['R@10']:.3f}, "
            f"MedR={metrics_t2a['MedianRank']:.3f}, MeanR={metrics_t2a['MeanRank']:.3f}, mAP={metrics_t2a['mAP']:.3f} | "
            f"A2T: R@1={metrics_a2t['R@1']:.3f}, R@5={metrics_a2t['R@5']:.3f}, R@10={metrics_a2t['R@10']:.3f}, "
            f"MedR={metrics_a2t['MedianRank']:.3f}, MeanR={metrics_a2t['MeanRank']:.3f}, mAP={metrics_a2t['mAP']:.3f}"
        )
        main_logger.info(
            f"Epoch [{epoch+1}] | IndicVoices metrics | "
            f"T2A: R@1={metrics_t2a_indicvoices['R@1']:.3f}, R@5={metrics_t2a_indicvoices['R@5']:.3f}, R@10={metrics_t2a_indicvoices['R@10']:.3f}, "
            f"MedR={metrics_t2a_indicvoices['MedianRank']:.3f}, MeanR={metrics_t2a_indicvoices['MeanRank']:.3f}, mAP={metrics_t2a_indicvoices['mAP']:.3f} | "
            f"A2T: R@1={metrics_a2t_indicvoices['R@1']:.3f}, R@5={metrics_a2t_indicvoices['R@5']:.3f}, R@10={metrics_a2t_indicvoices['R@10']:.3f}, "
            f"MedR={metrics_a2t_indicvoices['MedianRank']:.3f}, MeanR={metrics_a2t_indicvoices['MeanRank']:.3f}, mAP={metrics_a2t_indicvoices['mAP']:.3f}"
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
            }, output_dir / "clap_checkpoint_best_rasa.pt")
            main_logger.info(f"Saved new best Rasa checkpoint at epoch {epoch+1}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--device", default=None)
    train(parser.parse_args())
