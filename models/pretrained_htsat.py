from pathlib import Path

import torch

CHECKPOINT = Path(__file__).parent / "pretrained" / "HTSAT_AudioSet_Saved_6.ckpt"

def load_pretrained_htsat(model):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    target = model.state_dict()
    weights = {}
    for name, value in checkpoint.items():
        name = name.removeprefix("sed_model.")
        matches = [key for key, target_value in target.items() if key.endswith(name) and target_value.shape == value.shape]
        if len(matches) == 1:
            weights[matches[0]] = value
    if not weights:
        raise RuntimeError(f"No compatible HTS-AT weights found in {CHECKPOINT}")
    model.load_state_dict(weights, strict=False)
