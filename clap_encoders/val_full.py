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
    
    all_audio_embs = []
    all_text_embs = []

    with torch.no_grad():
        for batch in loader:
            mel_spec = batch['mel_spec'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                audio_embs = model.encode_audio(mel_spec)
                text_embs = model.encode_text(input_ids, attention_mask)

            all_audio_embs.append(audio_embs.cpu())
            all_text_embs.append(text_embs.cpu())

    all_audio_embs = torch.cat(all_audio_embs)
    all_text_embs = torch.cat(all_text_embs)
    
    similarity_a2t = (
        model.logit_scale.exp().cpu()
        * all_audio_embs @ all_text_embs.T
    )

    metrics_a2t = retrieval_metrics(similarity_a2t)
    metrics_t2a = retrieval_metrics(similarity_a2t.T)
    print(f"Checkpoint {i}")
    print(f"T2A metrics: {metrics_t2a}")
    print(f"A2T metrics: {metrics_a2t}")
    
