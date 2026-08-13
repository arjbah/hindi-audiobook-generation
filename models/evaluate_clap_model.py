import argparse
import importlib
import os
import random
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

from data import INDICVOICES_EVAL_SAMPLES, LANGUAGES

MODELS = ["laion_clap", "voiceclap", "mga_clap", "slap"]
ROOT = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gender", choices=["male", "female", "all"], default="male")
    parser.add_argument("--language", choices=["all", *LANGUAGES], default="all")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--indicvoices_limit", type=int, default=INDICVOICES_EVAL_SAMPLES)
    return parser.parse_args()

@contextmanager
def model_directory(name):
    path = ROOT / name
    previous_directory = Path.cwd()
    sys.path.insert(0, str(path))
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous_directory)
        sys.path.remove(str(path))

def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def checkpoint_path(path, default):
    checkpoint = path if path else default
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint

def retrieval_metrics(similarities):
    import torch

    targets = torch.arange(similarities.shape[0]).unsqueeze(1)
    ranks = (similarities.argsort(dim=1, descending=True) == targets).long().argmax(dim=1) + 1
    return {
        "R@1": (ranks <= 1).float().mean().item(),
        "R@5": (ranks <= 5).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MedianRank": ranks.median().float().item(),
        "MeanRank": ranks.float().mean().item(),
        "mAP": ranks.float().reciprocal().mean().item(),
    }

def evaluate_embeddings(audio, text):
    import torch.nn.functional as F

    similarities = F.normalize(audio, dim=-1) @ F.normalize(text, dim=-1).t()
    return {
        "T2A": retrieval_metrics(similarities.t()),
        "A2T": retrieval_metrics(similarities),
    }

def evaluate_clap(name, args):
    import torch
    from torch.utils.data import DataLoader

    directory = "laion_clap_training" if name == "laion_clap" else "voiceclap_training"
    with model_directory(directory) as path:
        dataset_module = importlib.import_module("clap_dataset")
        model_module = importlib.import_module("clap_model")
        model = model_module.CLAPModel()
        checkpoint = checkpoint_path(args.path, path / f"{args.language}_best_rasa.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        datasets = {
            "Rasa": dataset_module.IndicVoicesCLAPDataset("rasa", split="test", gender=args.gender, language=args.language),
            "IndicVoices": dataset_module.IndicVoicesCLAPDataset("indicvoices", split="train", gender=args.gender, language=args.language, limit=args.indicvoices_limit),
        }
        results = {}
        for dataset_name, dataset in datasets.items():
            audio_embeddings = []
            text_embeddings = []
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            for batch in loader:
                context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with torch.no_grad(), context:
                    if name == "laion_clap":
                        audio = model.encode_audio(batch["input_features"].to(device), batch["is_longer"].to(device))
                    else:
                        audio = model.encode_audio(batch["mel_spec"].to(device))
                    text = model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                audio_embeddings.append(audio.cpu())
                text_embeddings.append(text.cpu())
            results[dataset_name] = {"Embeddings": evaluate_embeddings(torch.cat(audio_embeddings), torch.cat(text_embeddings))}
    return checkpoint, results

def evaluate_mga(args):
    import torch
    from ruamel.yaml import YAML

    with model_directory("mga_clap_training") as path:
        dataset_module = importlib.import_module("data_handling.pretrain_dataset")
        model_module = importlib.import_module("models.ase_model")
        validation_module = importlib.import_module("pretrain")
        with (path / "settings/pretrain.yaml").open() as config_file:
            config = YAML(typ="safe", pure=True).load(config_file)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config["device"] = str(device)
        model = model_module.ASE(config).to(device)
        checkpoint = checkpoint_path(args.path, path / f"{args.language}_best_rasa.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()
        loaders = {
            "Rasa": dataset_module.pretrain_dataloader(config, split="test", language=args.language, gender=args.gender),
            "IndicVoices": dataset_module.indicvoices_dataloader(config, language=args.language, gender=args.gender, limit=args.indicvoices_limit),
        }
        results = {}
        keys = ["R@1", "R@5", "R@10", "R@50", "MedianRank", "MeanRank", "mAP"]
        for dataset_name, loader in loaders.items():
            metrics = validation_module.validate_re(model, loader, device)
            results[dataset_name] = {"Embeddings": {
                "T2A": dict(zip(keys, metrics["t2a"])),
                "A2T": dict(zip(keys, metrics["a2t"])),
            }}
    return checkpoint, results

def evaluate_slap(args):
    import hydra
    import torch
    from hydra import compose, initialize_config_dir

    with model_directory("slap") as path:
        utils = importlib.import_module("src.utils")
        utils.register_resolvers()
        overrides = [
            "data=rasa",
            "model=slap",
            "model/audio_encoder=htsat_audioset_slap",
            "model/text_encoder=muril_slap",
            "trainer=hindi",
            f"paths.pretrained={ROOT / 'pretrained'}",
            f"seed={args.seed}",
            f"data.gender={args.gender}",
            f"data.language={args.language}",
            f"data.indicvoices_limit={args.indicvoices_limit}",
        ]
        with initialize_config_dir(config_dir=str(path / "configs"), version_base="1.3"):
            config = compose(config_name="train", overrides=overrides)
        model = hydra.utils.instantiate(config.model)
        checkpoint = checkpoint_path(args.path, path / f"{args.language}_best_rasa.pt")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        model.on_load_checkpoint(state)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        datamodule = hydra.utils.instantiate(config.data)
        datamodule.setup()
        datamodule.tokenizer.tokenizer = model.text_encoder.encoder.tokenizer
        datamodule.tokenizer.update_options(datamodule.tokenizer.tokenizer_options)
        loaders = dict(zip(["Rasa", "IndicVoices"], datamodule.val_dataloader()))
        results = {}
        for dataset_index, (dataset_name, loader) in enumerate(loaders.items()):
            spaces = {name: {"audio": [], "text": []} for name in model.retrieval_spaces}
            for batch in loader:
                audio, text, descriptions = datamodule.on_before_batch_transfer(batch, dataset_index)
                audio = audio.to(device)
                text = text.to(device)
                with torch.no_grad():
                    _, audio_projection, audio_prediction = model.audio_encoder(audio)
                    _, text_projection, text_prediction = model.text_encoder(text)
                    _, audio_ema_projection, audio_ema_prediction = model.audio_ema(audio)
                    _, text_ema_projection, text_ema_prediction = model.text_ema(text)
                values = {
                    "Projections": (audio_projection, text_projection),
                    "Predictions": (audio_prediction, text_prediction),
                    "EMA Projections": (audio_ema_projection, text_ema_projection),
                    "EMA Predictions": (audio_ema_prediction, text_ema_prediction),
                }
                for space, (audio_value, text_value) in values.items():
                    spaces[space]["audio"].append(audio_value.cpu())
                    spaces[space]["text"].append(text_value.cpu())
            results[dataset_name] = {
                space: evaluate_embeddings(torch.cat(values["audio"]), torch.cat(values["text"]))
                for space, values in spaces.items()
            }
    return checkpoint, results

def format_metrics(metrics):
    return ", ".join(f"{name}={value:.3f}" for name, value in metrics.items())

def write_evaluation(args, checkpoint, results):
    lines = [
        f"Model: {args.model}",
        f"Checkpoint: {checkpoint}",
        f"Seed: {args.seed}",
        f"Gender: {args.gender}",
        f"Language: {args.language}",
        f"IndicVoices limit: {args.indicvoices_limit}",
    ]
    for dataset_name, spaces in results.items():
        for space, directions in spaces.items():
            lines.append("")
            lines.append(f"{dataset_name} {space} metrics")
            lines.append(f"T2A: {format_metrics(directions['T2A'])}")
            lines.append(f"A2T: {format_metrics(directions['A2T'])}")
    (ROOT / "evaluation.txt").write_text("\n".join(lines) + "\n")

def main():
    args = parse_args()
    if args.indicvoices_limit < 1:
        raise ValueError("--indicvoices_limit must be at least 1")
    if args.path:
        args.path = args.path.expanduser().resolve()
    set_seed(args.seed)
    if args.model in ["laion_clap", "voiceclap"]:
        checkpoint, results = evaluate_clap(args.model, args)
    elif args.model == "mga_clap":
        checkpoint, results = evaluate_mga(args)
    else:
        checkpoint, results = evaluate_slap(args)
    write_evaluation(args, checkpoint, results)

if __name__ == "__main__":
    main()
