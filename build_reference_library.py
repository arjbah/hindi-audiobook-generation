import os
import torch
import torch.nn.functional as F
import soundfile as sf
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import sys

# Add project root and clap_encoders to sys.path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "clap_encoders"))

from clap_model import CLAPModel
from train_clap import HTSATConfig

def build_library(max_samples_per_dataset=500, device="cpu"):
    # 1. Setup Models
    config = HTSATConfig()
    model = CLAPModel(config).to(device)
    
    checkpoint_path = "clap_model_epoch_120.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading CLAP weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Error: {checkpoint_path} not found. Cannot build library without trained weights.")
        return

    model.eval()

    # 2. Prepare Output Directory
    lib_dir = "reference_library"
    audio_dir = os.path.join(lib_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    metadata = []
    
    # 3. Process Datasets
    datasets_to_load = [
        ("indicvoices", "ai4bharat/indicvoices_r", "Hindi", "normalized"),
        ("rasa", "ai4bharat/Rasa", "Hindi", "text")
    ]
    
    for ds_id, path, name, text_key in datasets_to_load:
        print(f"\nProcessing {ds_id}...")
        try:
            ds = load_dataset(path, name, split="train", streaming=True)
        except Exception as e:
            print(f"Failed to load {ds_id}: {e}")
            continue
            
        count = 0
        for item in tqdm(ds, desc=f"Indexing {ds_id}"):
            if count >= max_samples_per_dataset:
                break
                
            audio_array = item['audio']['array']
            sr = item['audio']['sampling_rate']
            text = item[text_key].strip()
            
            if not text:
                continue
                
            # 1. Save Audio Locally (Persistence)
            audio_filename = f"{ds_id}_{count}.wav"
            local_path = os.path.join(audio_dir, audio_filename)
            sf.write(local_path, audio_array, sr)
            
            # 2. Compute Embedding
            # Preprocess audio using CLAPModel's internal logic
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0).to(device)
            
            with torch.no_grad():
                # encode_audio handles resampling, padding, and Mel extraction
                audio_emb = model.encode_audio(audio_tensor, sr=sr)
            
            # 3. Store Metadata
            metadata.append({
                "id": f"{ds_id}_{count}",
                "local_path": local_path,
                "transcript": text,
                "embedding": audio_emb.squeeze(0).cpu().tolist()
            })
            
            count += 1
            
    # 4. Save Final Library
    lib_path = os.path.join(lib_dir, "voice_library.pt")
    torch.save(metadata, lib_path)
    print(f"\n✓ Library built with {len(metadata)} samples.")
    print(f"✓ Saved to {lib_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500, help="Max samples per dataset")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_library(max_samples_per_dataset=args.samples, device=device)
