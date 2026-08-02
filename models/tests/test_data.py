"""Gender-filter behavior checks.

Runs against a synthetic in-memory dataset, so it needs no HuggingFace
credentials and no download. Run with:

    python models/tests/test_data.py
"""

import sys
from pathlib import Path

from datasets import Dataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.data import (
    GenderFilterError,
    _drop_other_genders,
    add_shared_args,
    filter_by_gender,
    gender_distribution,
    normalize_gender_value,
)


def make_dataset(genders: list[str]) -> Dataset:
    return Dataset.from_dict(
        {
            "gender": genders,
            "text": [f"utterance {i}" for i in range(len(genders))],
        }
    )


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}"
    )
    return condition


def main() -> int:
    results = []

    print("normalization is case- and whitespace-insensitive")
    results.append(check("'MALE' -> 'male'", normalize_gender_value("MALE") == "male"))
    results.append(
        check("' Female ' -> 'female'", normalize_gender_value(" Female ") == "female")
    )

    print("\nmixed dataset, varied spellings")
    mixed = make_dataset(["MALE", "male", "Female", " FEMALE ", "Male"])
    dist = gender_distribution(mixed)
    results.append(
        check(
            "distribution counts fold spellings",
            dist == {"male": 3, "female": 2},
            str(dist),
        )
    )
    results.append(
        check(
            "gender=both is a passthrough",
            len(filter_by_gender(mixed, "both", label="t")) == 5,
        )
    )
    results.append(
        check(
            "gender=male selects 3",
            len(filter_by_gender(mixed, "male", label="t")) == 3,
        )
    )
    results.append(
        check(
            "gender=female selects 2",
            len(filter_by_gender(mixed, "female", label="t")) == 2,
        )
    )

    print("\nsingle-gender dataset (the Rasa Hindi scenario)")
    male_only = make_dataset(["Male"] * 10)
    results.append(
        check(
            "gender=male still works",
            len(filter_by_gender(male_only, "male", label="t")) == 10,
        )
    )
    results.append(
        check(
            "gender=both still works",
            len(filter_by_gender(male_only, "both", label="t")) == 10,
        )
    )
    try:
        filter_by_gender(male_only, "female", label="Rasa/Hindi/train")
        results.append(check("gender=female raises", False, "no exception raised"))
    except GenderFilterError as exc:
        results.append(
            check(
                "gender=female raises with a diagnostic",
                "Rasa/Hindi/train" in str(exc) and "male" in str(exc),
                str(exc)[:120],
            )
        )

    print("\nerror handling")
    try:
        gender_distribution(Dataset.from_dict({"text": ["a"]}))
        results.append(
            check("missing gender column raises", False, "no exception raised")
        )
    except GenderFilterError:
        results.append(check("missing gender column raises", True))
    try:
        filter_by_gender(mixed, "nonbinary", label="t")  # type: ignore[arg-type]
        results.append(check("unknown gender raises", False, "no exception raised"))
    except GenderFilterError:
        results.append(check("unknown gender raises", True))

    print("\nIndicVoices drops genders outside male/female")
    with_other = make_dataset(["Male"] * 5 + ["Female"] * 4 + ["other"] * 3)
    kept = _drop_other_genders(with_other, label="IndicVoices/Hindi/train")
    results.append(
        check("3 'other' rows dropped, 9 kept", len(kept) == 9, f"{len(kept)} rows")
    )
    results.append(
        check(
            "both now equals male + female",
            len(filter_by_gender(kept, "both", label="t"))
            == len(filter_by_gender(kept, "male", label="t"))
            + len(filter_by_gender(kept, "female", label="t")),
        )
    )
    results.append(
        check(
            "no gender column is a passthrough",
            len(_drop_other_genders(Dataset.from_dict({"text": ["a", "b"]}), label="t"))
            == 2,
        )
    )

    print("\nper-model CLI defaults")
    import argparse

    other = argparse.ArgumentParser()
    add_shared_args(other)
    results.append(
        check(
            "default is both when unspecified",
            other.parse_args([]).gender == "both",
        )
    )
    mga = argparse.ArgumentParser()
    mga.add_argument("-s", "--batch_size", type=int, default=128)
    add_shared_args(mga, include_batch_size=False, gender_default="male")
    results.append(
        check(
            "mga_clap defaults to male (reproduces a16a513)",
            mga.parse_args([]).gender == "male",
        )
    )
    results.append(
        check(
            "explicit --gender still overrides the default",
            mga.parse_args(["--gender", "both"]).gender == "both",
        )
    )

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
