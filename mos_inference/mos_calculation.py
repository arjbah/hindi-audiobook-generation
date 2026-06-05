import json
from pathlib import Path
from collections import defaultdict
import statistics

folders = {
    "MOS": "data/hindi_raw_mos/hindi-audiobook-mos-v1",
    "NMOS": "data/hindi_raw_mos/hindi-audiobook-nmos-v3",
    "EMOS": "data/hindi_raw_mos/hindi-audiobook-emos"
}

results = defaultdict(dict)

for metric_name, folder in folders.items():
    results_path = Path(folder) / "results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_scores = defaultdict(list)

    stems = data.get("stems", {})

    for stem_data in stems.values():
        for model, scores in stem_data.items():
            if scores:
                model_scores[model].extend(scores)

    for model, scores in model_scores.items():
        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0

        results[model][metric_name] = {
            "mean": avg,
            "std": std,
            "length": len(scores)
        }

with open("metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(dict(results), f, indent=4)

models = sorted(results.keys())

print(
    f"{'Model':<20} "
    f"{'MOS':>8} {'Std':>8} {'N':>6} "
    f"{'NMOS':>8} {'Std':>8} {'N':>6} "
    f"{'EMOS':>8} {'Std':>8} {'N':>6}"
)

print("-" * 100)

for model in models:
    mos = results[model].get("MOS", {})
    nmos = results[model].get("NMOS", {})
    emos = results[model].get("EMOS", {})

    print(
        f"{model:<20} "
        f"{mos.get('mean', 0):>8.2f} {mos.get('std', 0):>8.2f} {mos.get('length', 0):>6} "
        f"{nmos.get('mean', 0):>8.2f} {nmos.get('std', 0):>8.2f} {nmos.get('length', 0):>6} "
        f"{emos.get('mean', 0):>8.2f} {emos.get('std', 0):>8.2f} {emos.get('length', 0):>6}"
    )