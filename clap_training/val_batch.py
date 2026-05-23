"""
1. Load clap model
2. Make the dataset of test datasets of Rasa and IndicVoices
3. Calculate loss
"""
from transformers import logging
from clap_model import CLAPModel, contrastive_loss
from train_clap import HTSATConfig
from clap_dataset import get_dataloader
import torch
from tqdm import tqdm

logging.set_verbosity_error()

def retrieval_metrics(similarity_matrix, ks=[1,5,10]):
    B = similarity_matrix.shape[0]

    # audio -> text
    ranks = []

    for i in range(B):
        sims = similarity_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)

        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    ranks = torch.tensor(ranks)

    metrics = {}

    for k in ks:
        metrics[f"R@{k}"] = (ranks <= k).float().mean().item()

    metrics["MedianRank"] = ranks.median().item()
    metrics["MeanRank"] = ranks.float().mean().item()

    # mAP
    APs = []
    for r in ranks:
        APs.append(1.0 / r.item())

    metrics["mAP"] = sum(APs) / len(APs)

    return metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = HTSATConfig()

loader = get_dataloader(split="test", batch_size=512, num_workers=4)

for i in range(50, 121, 10):
    model = CLAPModel(config).to(device)
    checkpoint = torch.load(f"clap_checkpoint_epoch_{i}.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    
    total_val = 0.0
    all_r1 = []
    all_r5 = []
    all_r10 = []
    all_mean = []
    all_median = []
    all_map = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Checkpoint {i}")
        for batch in pbar:
            mel_spec = batch['mel_spec'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_per_audio, logits_per_text = model(mel_spec, input_ids, attention_mask)
                loss = contrastive_loss(logits_per_audio, logits_per_text)
            
            metrics = retrieval_metrics(logits_per_audio)
            all_r1.append(metrics["R@1"])
            all_r5.append(metrics["R@5"])
            all_r10.append(metrics["R@10"])
            all_mean.append(metrics["MeanRank"])
            all_median.append(metrics["MedianRank"])
            all_map.append(metrics["mAP"])

            total_val += loss.item()
            avg_loss = total_val / (pbar.n + 1)
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

        total_val /= len(loader)

    print(f"Loss for checkpoint {i:}: {total_val:.4f}")
    print(f"R@1: {sum(all_r1) / len(all_r1)}")
    print(f"R@5: {sum(all_r5) / len(all_r5)}")
    print(f"R@10: {sum(all_r10) / len(all_r10)}")
    print(f"Mean Rank: {sum(all_mean) / len(all_mean)}")
    print(f"Median Rank: {sum(all_median) / len(all_median)}")
    print(f"mAP: {sum(all_map) / len(all_map)}")