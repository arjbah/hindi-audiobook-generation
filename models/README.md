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
same across models.** MGA-CLAP defaults to `male`; the other three never
filtered and default to `both`. Per William the male-only choice is deliberate,
not an artifact of the incorrect comment below. Pass `--gender` explicitly
whenever you compare models to each other.

**MGA-CLAP is not doing the same job as the other three.** It trains nine
models — one per language, Assamese through Telugu — on the Rasa *male* subset.
The other three train a single model on the full Rasa Hindi data. `--gender`
also does not reach its IndicVoices eval split, which stays male-only and
capped at 2000 rows.

Rows whose gender is neither male nor female are dropped from IndicVoices
(Hindi train has 72), so `both` always equals male + female.

`--seed` defaults to unset, keeping each model's existing behavior: LAION-CLAP
and VoiceCLAP unseeded (they have no seeding code, so their baselines are not
reproducible run-to-run), MGA-CLAP 20, SLAP 1. Pass `--seed` for anything you
intend to report.

| Model | Runs today | Baseline Rasa / IndicVoices T2A R@1 |
|---|---|---|
| `laion_clap` | yes | 0.847 / 0.133 |
| `voiceclap` | yes | 0.929 / 0.348 |
| `mga_clap` | yes | 0.968 / 0.548 — **needs `--gender both`**, see below |
| `slap` | yes | 0.953 / 0.391 |

`train.py` reports any missing file up front with the reason.

**MGA-CLAP's baseline does not reproduce under the default.** Its logged run
(`mga_clap/outputs/exp_name_lr_5e-05_seed_20/logging/output.txt`, 2026-06-18)
records `Size of training set: 25713` — the full Hindi train split, both
genders. The male-only filter landed six weeks later, and male is 12,116 of
those rows. To reproduce `0.968 / 0.548` you must pass `--gender both`; the
default `male` is the current intended training configuration, not the
configuration that produced the number in this table.

**Prerequisites:** `hf auth login` plus accepted gates on `ai4bharat/Rasa` and
`ai4bharat/indicvoices_r` (both gated); ~200 GB disk (Rasa is ~8–17 GB and
LAION-CLAP caches ~26 GB of features); a 24 GB GPU. MGA-CLAP additionally
expects `/mnt/huggingface` and `/mnt/mga_clap_training_cache` to be writable —
absolute paths from the AWS box it was last run on.

**Shared code** is only `common/data.py` — dataset loading, gender filtering,
seeding, and the shared CLI flags. Each model keeps its own metrics, logging,
optimizer and preprocessing untouched. `models/tests/test_data.py` covers the
filtering, including the single-gender case, with no GPU or credentials needed.

**Measured gender distribution** (counted 2026-08-02 by reading the `gender`
column straight from the Hub parquet; spellings are exactly `Male`/`Female`):

| Dataset / split | Rows | Male | Female | Other |
|---|---|---|---|---|
| Rasa Hindi train | 25,713 | 12,116 | 13,597 | — |
| Rasa Hindi test | 2,858 | 1,348 | 1,510 | — |
| IndicVoices Hindi train | 26,318 | 14,209 | 12,037 | 72 |
| IndicVoices Hindi test | 376 | 205 | 171 | — |

Note `rasa_model_comparison/rasa_model_comparison.py:35` claims Hindi has no
female samples. **That is incorrect** — female is the slight majority. The
comment is left in place because that script is outside this directory's scope,
but do not rely on it.

**Unverified:** SLAP's `paths.output_dir` normally resolves through Dora
(`${dora:xp.folder}`). The launcher overrides it; that path has not been run.

**Known divergence, left alone:** VoiceCLAP never clamps `logit_scale` while
LAION-CLAP does. Real bug, but fixing it would move VoiceCLAP's numbers for a
reason unrelated to the gender work.
