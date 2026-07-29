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

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
