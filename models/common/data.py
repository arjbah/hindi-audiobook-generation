"""Rasa/IndicVoices loading with gender filtering, shared by all four models.

This is the only shared module. Each model keeps its own metrics, logging,
optimizer and preprocessing exactly as it had them.

Rasa exposes a free-text ``gender`` column. Two places in this repo match it as
the exact string ``"Male"`` -- ``rasa_model_comparison.py``, and MGA-CLAP's
dataset since a16a513 -- but the full set of values per language config is still
unverified. So matching here is case- and whitespace-insensitive, every filter
logs its before/after counts, and an empty result is a hard error rather than a
silent zero-row training set.
"""

from __future__ import annotations

import logging
from typing import Literal, get_args

from datasets import Dataset, load_dataset

log = logging.getLogger(__name__)

Gender = Literal["male", "female", "both"]
GENDER_CHOICES: tuple[str, ...] = get_args(Gender)

RASA_REPO = "ai4bharat/Rasa"
INDICVOICES_REPO = "ai4bharat/indicvoices_r"
LANGUAGE = "Hindi"
GENDER_COLUMN = "gender"


class GenderFilterError(RuntimeError):
    """Raised when a gender subset cannot be produced from a dataset."""


def normalize_gender_value(value: object) -> str:
    """Fold a raw ``gender`` cell to a comparable token."""
    return str(value).strip().lower()


def gender_distribution(dataset: Dataset) -> dict[str, int]:
    """Count rows per normalized gender value."""
    if GENDER_COLUMN not in dataset.column_names:
        raise GenderFilterError(
            f"Dataset has no {GENDER_COLUMN!r} column; found {dataset.column_names}"
        )
    counts: dict[str, int] = {}
    for value in dataset[GENDER_COLUMN]:
        token = normalize_gender_value(value)
        counts[token] = counts.get(token, 0) + 1
    return counts


def filter_by_gender(dataset: Dataset, gender: str, *, label: str) -> Dataset:
    """Restrict ``dataset`` to one gender, or return it untouched for ``both``."""
    if gender not in GENDER_CHOICES:
        raise GenderFilterError(
            f"Unknown gender {gender!r}; expected one of {GENDER_CHOICES}"
        )

    before = len(dataset)
    if gender == "both":
        log.info("%s: %d rows (gender=both, unfiltered)", label, before)
        return dataset

    distribution = gender_distribution(dataset)
    if gender not in distribution:
        raise GenderFilterError(
            f"{label}: no rows with gender={gender!r}. Present values: {distribution}. "
            f"If this is Rasa Hindi it may be single-gender; see models/README.md."
        )

    filtered = dataset.filter(
        lambda row: normalize_gender_value(row[GENDER_COLUMN]) == gender,
        desc=f"Filtering {label} to gender={gender}",
    )
    if len(filtered) == 0:
        raise GenderFilterError(f"{label}: gender={gender!r} produced an empty split")

    log.info("%s: %d -> %d rows after gender=%s", label, before, len(filtered), gender)
    return filtered


def _truncate(dataset: Dataset, limit: int | None, *, label: str) -> Dataset:
    """Take the first ``limit`` rows, for smoke tests that must not train fully."""
    if limit is None or limit >= len(dataset):
        return dataset
    log.warning(
        "%s: TRUNCATED to %d rows (--limit); not comparable to baselines", label, limit
    )
    return dataset.select(range(limit))


def load_rasa(split: str, gender: str = "both", limit: int | None = None) -> Dataset:
    """Load a Rasa Hindi split, gender-filtered.

    Applied to train and test alike, so in-domain evaluation matches the
    population the model was trained on.
    """
    label = f"Rasa/{LANGUAGE}/{split}"
    dataset = load_dataset(RASA_REPO, LANGUAGE, split=split)
    return _truncate(filter_by_gender(dataset, gender, label=label), limit, label=label)


def _drop_other_genders(dataset: Dataset, *, label: str) -> Dataset:
    """Drop rows whose gender is neither male nor female.

    IndicVoices Hindi train carries 72 rows labelled ``other``. No ``--gender``
    setting selects them, so leaving them in would make ``both`` quietly differ
    from male + female. Per William, they are excluded outright.
    """
    if GENDER_COLUMN not in dataset.column_names:
        return dataset
    before = len(dataset)
    filtered = dataset.filter(
        lambda row: normalize_gender_value(row[GENDER_COLUMN]) in {"male", "female"},
        desc=f"Dropping non-male/female rows from {label}",
    )
    dropped = before - len(filtered)
    if dropped:
        log.info("%s: dropped %d rows with gender outside male/female", label, dropped)
    return filtered


def load_indicvoices(split: str = "test", limit: int | None = None) -> Dataset:
    """Load an IndicVoices Hindi split.

    Never filtered to a single gender: it is the fixed out-of-domain probe, so
    holding it constant keeps that column comparable across gender settings.
    Rows outside male/female are dropped -- see ``_drop_other_genders``.
    """
    label = f"IndicVoices/{LANGUAGE}/{split}"
    dataset = load_dataset(INDICVOICES_REPO, LANGUAGE, split=split)
    dataset = _drop_other_genders(dataset, label=label)
    return _truncate(dataset, limit, label=label)


def add_shared_args(
    parser, include_batch_size: bool = True, gender_default: str = "both"
) -> None:
    """Attach the shared flags to a model's own argparse parser.

    :param include_batch_size: MGA-CLAP already defines ``-s/--batch_size``.
        argparse would not error -- it keys conflicts on option strings, and
        ``--batch_size`` and ``--batch-size`` differ -- but both write to
        ``args.batch_size``, so whichever appeared last on the command line
        would silently win. MGA-CLAP opts out and keeps its original flag.
    :param gender_default: what ``--gender`` falls back to when the flag is
        absent. MGA-CLAP passes ``"male"`` so that omitting the flag reproduces
        the hardcoded ``item["gender"] == "Male"`` filter it carried before this
        flag existed; the other three models never filtered and default to
        ``"both"``.
    """
    parser.add_argument(
        "--gender",
        choices=GENDER_CHOICES,
        default=gender_default,
        help="Rasa subset for training and in-domain eval. IndicVoices is never filtered.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed RNGs. Omit to keep this model's default.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override the epoch count."
    )
    if include_batch_size:
        parser.add_argument(
            "--batch-size", type=int, default=None, help="Override the batch size."
        )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate splits to N rows (smoke tests only).",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Where to write logs and checkpoints."
    )


def set_seed(seed: int | None) -> int | None:
    """Seed python/numpy/torch. ``None`` leaves each model's default behavior."""
    if seed is None:
        return None
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Seeded python/numpy/torch with %d", seed)
    return seed
