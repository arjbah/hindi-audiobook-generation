# Unified training and evaluation

```bash
python models/train.py --model laion_clap                 # gender=both
python models/train.py --model mga_clap                   # gender=male (see below)
python models/train.py --model voiceclap --gender female
python models/train.py --model all --seed 42
```

`--gender {male,female,both}` filters **Rasa train and Rasa test**; IndicVoices
is never filtered, so it stays a fixed out-of-domain probe. Matching is case-
and whitespace-insensitive, and an empty result is a hard error rather than a
silent zero-row training set.

**Omitting `--gender` preserves each model's prior behavior, which is not the
same across models.** MGA-CLAP hardcoded `item["gender"] == "Male"` in a16a513,
so it defaults to `male`; the other three never filtered and default to `both`.
Pass `--gender` explicitly whenever you compare models to each other.

MGA-CLAP has two further filters that `--gender` does **not** reach, both kept
deliberately so its numbers stay comparable to the existing runs: its
IndicVoices eval split is male-only and capped at 2000 rows, and it loops over
nine languages, not just Hindi.

`--seed` defaults to unset, keeping each model's existing behavior: LAION-CLAP
and VoiceCLAP unseeded (they have no seeding code, so their baselines are not
reproducible run-to-run), MGA-CLAP 20, SLAP 1. Pass `--seed` for anything you
intend to report.

| Model | Runs today | Baseline Rasa / IndicVoices T2A R@1 |
|---|---|---|
| `laion_clap` | yes | 0.847 / 0.133 |
| `voiceclap` | yes | 0.929 / 0.348 |
| `mga_clap` | needs `HTSAT_AudioSet_Saved_6.ckpt` | 0.968 / 0.548 |
| `slap` | needs `HTSAT_AudioSet_Saved_6.ckpt` | 0.953 / 0.391 |

`train.py` reports any missing file up front with the reason.

**Prerequisites:** `hf auth login` plus accepted gates on `ai4bharat/Rasa` and
`ai4bharat/indicvoices_r` (both gated); ~200 GB disk (Rasa is ~8–17 GB and
LAION-CLAP caches ~26 GB of features); a 24 GB GPU. MGA-CLAP additionally
expects `/mnt/huggingface` and `/mnt/mga_clap_training_cache` to be writable —
absolute paths from the AWS box it was last run on.

**Shared code** is only `common/data.py` — dataset loading, gender filtering,
seeding, and the shared CLI flags. Each model keeps its own metrics, logging,
optimizer and preprocessing untouched. `models/tests/test_data.py` covers the
filtering, including the single-gender case, with no GPU or credentials needed.

**Whether Rasa Hindi has female samples is unsettled.** Two places assume
male-only: `rasa_model_comparison/rasa_model_comparison.py:35`, and MGA-CLAP's
dataset. If they are right, `--gender female` fails immediately listing the
values actually present — that is intended, not a bug. To check:
`python -c "from common.data import *; print(gender_distribution(load_rasa('train')))"`

**Unverified:** SLAP's `paths.output_dir` normally resolves through Dora
(`${dora:xp.folder}`). The launcher overrides it; that path has not been run.

**Known divergence, left alone:** VoiceCLAP never clamps `logit_scale` while
LAION-CLAP does. Real bug, but fixing it would move VoiceCLAP's numbers for a
reason unrelated to the gender work.
