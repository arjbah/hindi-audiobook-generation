import torch
import os
import sys

# Ensure we can import from clap_encoders
sys.path.append(os.path.join(os.getcwd(), "clap_encoders"))

from clap_model import CLAPModel
from train_clap import HTSATConfig

def extract_backbone(checkpoint_path, output_path):
    device = torch.device("cpu")
    config = HTSATConfig()
    
    print(f"Initializing model (ignoring backbone warning)...")
    model = CLAPModel(config).to(device)
    
    print(f"Loading full checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both wrapped and unwrapped state dicts
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    print(f"Loading state dict (strict=False to see what matches)...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys[:5]}... ({len(missing_keys)} total)")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys[:5]}... ({len(unexpected_keys)} total)")
    
    print(f"Extracting audio encoder (HTS-AT) weights...")
    audio_encoder_state = model.audio_encoder.state_dict()
    
    # Wrap in 'state_dict' and add 'sed_model.' prefix to match original HTS-AT format
    backbone_state = {"state_dict": {}}
    for k, v in audio_encoder_state.items():
        backbone_state["state_dict"][f"sed_model.{k}"] = v
        
    print(f"Saving backbone to {output_path}...")
    torch.save(backbone_state, output_path)
    print("✓ Success!")

if __name__ == "__main__":
    extract_backbone("clap_model_epoch_120.pt", "HTSAT_AudioSet_Saved_6.ckpt")
