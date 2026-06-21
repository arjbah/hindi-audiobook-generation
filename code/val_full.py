"""
1. Load clap model
2. Make the dataset of test datasets of Rasa and IndicVoices
3. Calculate loss
"""
import os
from transformers import logging
from clap_model import CLAPModel, contrastive_loss
#from train_clap import HTSATConfig
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

    '''
    # mAP@10 (Official LAION-CLAP Metric)
    ranks_zero_indexed = ranks - 1
    mAP_10 = torch.where(
        ranks_zero_indexed < 10, 
        1.0 / ranks.float(), 
        torch.tensor(0.0, device=ranks.device)
    )
    metrics["mAP@10"] = mAP_10.mean().item()
    '''

    return metrics

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

#config = HTSATConfig()

loader = get_dataloader(split="test", batch_size=512, num_workers=4)

for i in range(120, 121, 10):
    # Initialize model
    model = CLAPModel().to(device)
    
    # Check both potential checkpoint file names to prevent FileNotFoundError
    checkpoint_path = f"clap_checkpoint_epoch_{i}.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = f"clap_model_epoch_{i}.pt"
        
    print(f"\n==========================================")
    print(f"Evaluating Checkpoint: {checkpoint_path}")
    print(f"==========================================")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Retrieve state-dict 
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval() #swithced from .train() to .eval()
    
    all_audio_embs = []
    all_text_embs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            mel_spec = batch['mel_spec'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    audio_embs = model.encode_audio(mel_spec)
                    text_embs = model.encode_text(input_ids, attention_mask)
            else:
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
    
    print(f"\n--- Checkpoint {i} Global Evaluation Results ---")
    print(f"T2A metrics (Text-to-Audio): {metrics_t2a}")
    print(f"A2T metrics (Audio-to-Text): {metrics_a2t}")
