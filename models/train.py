import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from data import INDICVOICES_EVAL_SAMPLES, LANGUAGES

MODELS = ["voiceclap", "slap", "mga_clap", "laion_clap"]
ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gender", choices=["male", "female", "all"], default="male")
    parser.add_argument("--language", choices=["all", *LANGUAGES], default="all")
    parser.add_argument("--indicvoices_limit", type=int, default=INDICVOICES_EVAL_SAMPLES)
    parser.add_argument("--model", choices=[*MODELS, "all"], required=True)
    return parser.parse_args()

def run_module(directory, module_name, function_name, args):
    path = ROOT / directory
    previous_directory = Path.cwd()
    previous_modules = set(sys.modules)
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        module = importlib.import_module(module_name)
        getattr(module, function_name)(args.seed, args.epochs, args.batch_size, args.gender, args.language, args.indicvoices_limit)
    finally:
        os.chdir(previous_directory)
        sys.path.remove(str(path))
        for name in set(sys.modules) - previous_modules:
            module = sys.modules.get(name)
            module_path = getattr(module, "__file__", None)
            if module_path and Path(module_path).resolve().is_relative_to(path):
                del sys.modules[name]

def run_slap(args):
    path = ROOT / "slap"
    command = [
        sys.executable,
        "src/train.py",
        "data=rasa",
        "model=slap",
        "model/audio_encoder=htsat_audioset_slap",
        "model/text_encoder=muril_slap",
        "trainer=hindi",
        f"paths.pretrained={ROOT / 'pretrained'}",
        f"seed={args.seed}",
        f"trainer.max_epochs={args.epochs}",
        f"data.dataloader_kwargs.batch_size={args.batch_size}",
        f"data.gender={args.gender}",
        f"data.language={args.language}",
        f"data.indicvoices_limit={args.indicvoices_limit}",
    ]
    subprocess.run(command, cwd=path, check=True)

def run_model(model, args):
    if model == "laion_clap":
        run_module("laion_clap_training", "train_clap", "train", args)
    elif model == "voiceclap":
        run_module("voiceclap_training", "train_clap", "train", args)
    elif model == "mga_clap":
        run_module("mga_clap_training", "pretrain", "main", args)
    else:
        run_slap(args)

def main():
    args = parse_args()
    if args.indicvoices_limit < 1:
        raise ValueError("--indicvoices_limit must be at least 1")
    models = MODELS if args.model == "all" else [args.model]
    for model in models:
        run_model(model, args)

if __name__ == "__main__":
    main()
