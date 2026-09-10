import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


MODELS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = MODELS_DIR / "hindi_model_comparison.txt"
SEPARATOR = "-" * 80

# SLAP checkpoints contain objects pickled under the local `src` package.
SLAP_DIR = MODELS_DIR / "slap"
if str(SLAP_DIR) not in sys.path:
    sys.path.insert(0, str(SLAP_DIR))


@dataclass(frozen=True)
class ModelFiles:
    name: str
    checkpoint: Path
    log: Path
    epoch_offset: int


MODELS = (
    ModelFiles(
        "mga_clap",
        MODELS_DIR / "mga_clap_training/checkpoints/hindi_best_rasa.pt",
        MODELS_DIR / "mga_clap_training/outputs/hindi/logging/output.txt",
        0,
    ),
    ModelFiles(
        "laion_clap",
        MODELS_DIR / "laion_clap_training/checkpoints/hindi_best_rasa.pt",
        MODELS_DIR / "laion_clap_training/train_log.txt",
        1,
    ),
    ModelFiles(
        "voiceclap",
        MODELS_DIR / "voiceclap_training/checkpoints/hindi_best_rasa.pt",
        MODELS_DIR / "voiceclap_training/train_log.txt",
        1,
    ),
    ModelFiles(
        "slap",
        MODELS_DIR / "slap/checkpoints/hindi_best_rasa-v1.pt",
        MODELS_DIR / "slap/logs/xps/08dea1a4/train_log.txt",
        1,
    ),
)


def checkpoint_epoch(path: Path) -> int:
    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if "epoch" not in checkpoint:
        raise KeyError(f"Checkpoint has no epoch field: {path}")
    epoch = int(checkpoint["epoch"])
    del checkpoint
    gc.collect()
    return epoch


def metric_title(label: str) -> str:
    if " " not in label:
        return f"{label} Embeddings metrics"

    dataset, space = label.split(" ", 1)
    words = ["EMA" if word == "ema" else word.capitalize() for word in space.split()]
    return f"{dataset} {' '.join(words)} metrics"


def stats_for_epoch(log_path: Path, epoch: int):
    metric_pattern = re.compile(
        rf"Epoch \[{epoch}\] \| (.+?) metrics \| (T2A: .+?) \| (A2T: .+)$"
    )
    loss_patterns = (
        re.compile(rf"Epoch {epoch} loss: ([0-9.eE+-]+)"),
        re.compile(rf"loss for epoch \[{epoch}\]: ([0-9.eE+-]+)"),
    )

    metrics = {}
    loss = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        metric_match = metric_pattern.search(line)
        if metric_match:
            label, t2a, a2t = metric_match.groups()
            metrics[label] = (t2a, a2t)

        for pattern in loss_patterns:
            loss_match = pattern.search(line)
            if loss_match:
                loss = float(loss_match.group(1))

    if not metrics:
        raise ValueError(f"No metrics for epoch {epoch} in {log_path}")
    return loss, metrics


def format_model(model: ModelFiles) -> str:
    stored_epoch = checkpoint_epoch(model.checkpoint)
    epoch = stored_epoch + model.epoch_offset
    loss, metrics = stats_for_epoch(model.log, epoch)

    lines = [
        f"Model: {model.name}",
        f"Checkpoint: {model.checkpoint}",
        f"Epoch: {epoch}",
        "Language: hindi",
    ]
    if loss is not None:
        lines.append(f"Training loss: {loss:.6f}")

    for label, (t2a, a2t) in metrics.items():
        lines.extend(("", metric_title(label), t2a, a2t))
    return "\n".join(lines)


def main() -> None:
    sections = [format_model(model) for model in MODELS]
    OUTPUT_PATH.write_text(
        f"\n\n{SEPARATOR}\n\n".join(sections) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
