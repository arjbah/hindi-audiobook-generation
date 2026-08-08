#!/usr/bin/env python3
"""Unified entrypoint for the four retrieval models.

    python models/train.py --model laion_clap --gender both
    python models/train.py --model all --gender female --seed 42

Each model runs in a subprocess with its own working directory. That is
required, not incidental:

* ``laion_clap`` and ``voiceclap`` both contain ``clap_model.py``,
  ``clap_dataset.py`` and ``train_clap.py``. In one process the second model
  would silently receive the first's cached modules.
* ``mga_clap`` imports its own subpackage as ``from models.ase_model import ASE``,
  which collides with this top-level ``models`` directory.
* ``slap`` needs its own directory as cwd for Hydra config discovery and for the
  ``.project-root`` marker ``rootutils`` searches for.

The launcher only translates flags and reports failures; each model's own
training script is otherwise unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent
RUNS_DIR = MODELS_DIR.parent / "runs"

GENDERS = ("male", "female", "both")

HTSAT_NOTE = (
    "HTS-AT AudioSet checkpoint, not vendored here. Obtain it from the HTS-AT "
    "release (RetroCirce/HTS-Audio-Transformer); WavCaps uses the same file."
)

# name -> (script, hydra config-group selections, required files)
MODELS: dict[str, tuple[str, tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "laion_clap": ("train_clap.py", (), ()),
    "voiceclap": ("train_clap.py", (), ()),
    "mga_clap": (
        "pretrain.py",
        (),
        (("pretrained_models/HTSAT_AudioSet_Saved_6.ckpt", HTSAT_NOTE),),
    ),
    "slap": (
        "src/train.py",
        (
            "data=rasa",
            "model=slap",
            "model/audio_encoder=htsat_audioset_slap",
            "model/text_encoder=muril_slap",
            "trainer=hindi",
        ),
        (("pretrained/HTSAT_AudioSet_Saved_6.ckpt", HTSAT_NOTE),),
    ),
}

# MGA-CLAP filtered Rasa to Male in a16a513; omitting --gender reproduces that.
# The other three never filtered, so they default to the full dataset.
DEFAULT_GENDER = {"mga_clap": "male"}

# Cheapest first, so a broken environment surfaces in minutes rather than
# after a 25-hour SLAP run.
ALL_ORDER = ("voiceclap", "mga_clap", "laion_clap", "slap")


def build_command(
    name: str, args: argparse.Namespace, output_dir: Path, gender: str
) -> list[str]:
    """Translate the shared flags into that model's own argument style."""
    script, hydra_overrides, _ = MODELS[name]
    command = [sys.executable, script]

    if hydra_overrides:
        command.extend(hydra_overrides)
        command += [f"data.gender={gender}", f"paths.output_dir={output_dir}"]
        if args.seed is not None:
            command.append(f"seed={args.seed}")
        if args.epochs is not None:
            command.append(f"trainer.max_epochs={args.epochs}")
        if args.batch_size is not None:
            command.append(f"data.dataloader_kwargs.batch_size={args.batch_size}")
        if args.limit is not None:
            command.append(f"data.limit={args.limit}")
        return command

    command += ["--gender", gender, "--output-dir", str(output_dir)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.epochs is not None:
        command += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        # MGA-CLAP keeps its original -s flag; the other two take --batch-size.
        command += (["-s"] if name == "mga_clap" else ["--batch-size"]) + [
            str(args.batch_size)
        ]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    # Only MGA-CLAP loops over languages; the other three are Hindi-only.
    if name == "mga_clap" and args.languages:
        command += ["--languages", *args.languages]
    return command


def run_model(name: str, args: argparse.Namespace) -> int:
    """Run one model to completion, returning its exit code."""
    _, _, required = MODELS[name]
    directory = MODELS_DIR / name
    gender = args.gender or DEFAULT_GENDER.get(name, "both")

    missing = [(p, why) for p, why in required if not (directory / p).exists()]
    if missing:
        print(
            f"\n[{name}] CANNOT RUN -- {len(missing)} required file(s) missing:",
            file=sys.stderr,
        )
        for path, why in missing:
            print(f"  - {directory / path}\n      {why}", file=sys.stderr)
        return 78  # EX_CONFIG

    output_dir = (
        Path(args.output_dir) if args.output_dir else RUNS_DIR / f"{name}_{gender}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # models/ on PYTHONPATH makes `common.data` importable; the model's own
    # directory stays ahead of it via cwd, so mga_clap's `models.*` imports
    # still resolve to its local subpackage.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(MODELS_DIR), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    command = build_command(name, args, output_dir, gender)
    print(
        f"\n{'=' * 70}\n[{name}] {' '.join(command)}\n  cwd: {directory}\n"
        f"  out: {output_dir}\n{'=' * 70}",
        flush=True,
    )

    started = time.monotonic()
    result = subprocess.run(command, cwd=directory, env=env, check=False)
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(
        f"[{name}] {status} in {(time.monotonic() - started) / 60:.1f} min", flush=True
    )
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate T2A/A2T retrieval models on Rasa Hindi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=[*MODELS, "all"])
    parser.add_argument(
        "--gender",
        choices=GENDERS,
        default=None,
        help="Rasa subset for training and in-domain eval; IndicVoices is never "
        "filtered. Omit to keep each model's own default (mga_clap: male, "
        "others: both).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed RNGs. Omit to keep each model's own default.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate splits to N rows (smoke tests only).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="MGA-CLAP only: which languages to train. Omit for all nine.",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    names = ALL_ORDER if args.model == "all" else (args.model,)
    failures = [name for name in names if run_model(name, args) != 0]

    if failures:
        print(f"\nFailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll runs completed: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
