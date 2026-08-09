# Phase A: environment verification

Goal: prove all four training scripts start and run without dependency errors
on a clean GPU box, then freeze the environment that worked. **Not** a
convergence run — no number here is comparable to any baseline.

Run on 2026-08-09, `g6e.2xlarge` (1x NVIDIA L40S 46 GB, 8 vCPU, 61 GiB RAM),
Ubuntu 22.04 Deep Learning Base AMI `ami-0326665395a428ccf`, driver 595.91.07.

## Reproducing the environment

```bash
uv venv --python 3.11 venv-clap && source venv-clap/bin/activate
uv pip install -r models/requirements-clap.lock

uv venv --python 3.11 venv-slap && source venv-slap/bin/activate
uv pip install -r models/requirements-slap.lock
```

Two environments, not one. SLAP's `requirements.txt` and the CLAP models
resolve to compatible versions today, but SLAP pulls a much larger graph and
keeping them apart means one model's upgrade cannot silently move another's
numbers.

## Gotchas, in the order they bite

**1. `setuptools<81` is mandatory.** `psds_eval` and `sed_scores_eval` pull
`dcase_util`, which imports `pkg_resources` — removed in setuptools 81. Without
the pin, `from models.ase_model import ASE` dies with
`ModuleNotFoundError: No module named 'pkg_resources'`. MGA-CLAP only uses those
packages in `zero_shot_grounding.py`, off the training path, but
`tools/utils.py`'s wildcard import drags them in anyway. The pin is in both lock
files.

**2. The repo is private, so the box cannot clone it.** `git clone` over HTTPS
fails with `could not read Username for 'https://github.com'`. Options: a
read-only deploy token, or ship the code another way. A full `git bundle` is
610 MB because the history carries ~250 MB of `.wav`/`.flac` under
`long_story_generation/` and `sovits_voice_conversion/`; `git archive HEAD
models/` is 4.4 MB and enough to run everything here.

**3. `HF_HOME` and the token location must agree.** Setting
`HF_HOME=/mnt/huggingface` makes the libraries look for the token at
`$HF_HOME/token`, *not* `~/.cache/huggingface/token`. Getting this wrong fails
late and confusingly, as `DatasetNotFoundError: ... is a gated dataset`, which
reads like a permissions problem rather than a path problem. Verify auth with
`HF_HOME` set to the same value the training run will use — otherwise the check
passes and the run still fails.

**4. Budget 400 GB of disk, not 200.** Hugging Face keeps both the downloaded
parquet and the generated Arrow copy, so on-disk is roughly 2x the Hub size.

| Dataset (Hindi) | Hub | On disk |
|---|---|---|
| Rasa | 16.4 GiB | ~33 GB |
| IndicVoices | 44.0 GiB | ~88 GB |

Plus ~10 GB of venvs and LAION-CLAP's feature cache. 200 GB runs out partway
through IndicVoices.

**5. `--limit` does not shorten the download.** `common/data.py` calls
`load_dataset` for the whole split and truncates after, so `--limit 50` still
pulls all 44 GiB of IndicVoices. Pre-fetch both datasets once into a shared
`HF_HOME` before running anything, or every model re-triggers it.

**6. Every model loads IndicVoices, not just Rasa.** It is the fixed
out-of-domain probe. Staging only Rasa leaves each run stalling on a 99-shard
download that looks like a hang.

**7. MGA-CLAP's `cache_dir` bypasses the shared cache.** `pretrain_dataset.py`
passes `cache_dir="/mnt/huggingface"`, which makes `datasets` treat that as the
cache *root* — so it misses a cache built under `HF_HOME` at
`/mnt/huggingface/datasets/` and regenerates from scratch at 10–25 examples/s,
20–40 minutes on the full split. Fixed without touching the file:

```bash
ln -sfn /mnt/huggingface/datasets/ai4bharat___rasa          /mnt/huggingface/ai4bharat___rasa
ln -sfn /mnt/huggingface/datasets/ai4bharat___indicvoices_r /mnt/huggingface/ai4bharat___indicvoices_r
```

**8. `--limit` below the batch size trains nothing, silently.** MGA-CLAP and
SLAP both default to `batch_size: 128`. At `--limit 50`, `drop_last` leaves zero
batches and the training loop is skipped — MGA still logs
`loss for epoch [1]: 0.000` and a full set of eval metrics, SLAP still exits 0.
Nothing in either log says "no gradient step ran" unless you read
`size of batches: 0` or `Trainer.fit stopped: No training batches`. **Exit code
0 is not evidence of training.** Pass `--batch-size 8` with a small `--limit`.

**9. MGA-CLAP's gender filter scans the full split before `--limit`.** It walks
all 25,713 rows at ~90/s (~5 min) before truncating to 50. Same shape as the
download: `--limit` shortens training, never the pipeline ahead of it.

## Results

Smoke runs, `--limit 50 --epochs 1 --batch-size 8`, 2026-08-09. **None of these
numbers mean anything** — 50 rows, one epoch. R@1 ≈ 0.02 is 1/50, i.e. chance,
which is the correct outcome and the only thing being checked.

| Model | Result | Evidence it really trained |
|---|---|---|
| `laion_clap` | runs | loss 3.930, checkpoint saved, both evals ran |
| `voiceclap` | runs | loss 3.914, checkpoint saved, both evals ran |
| `mga_clap` | runs | `size of batches: 6`, loss 2.113 |
| `slap` | runs | 6/6 batches, `global step 6`, checkpoint saved, test 7/7 |

Every import resolves, every model constructs, Rasa and IndicVoices both load,
evaluation runs, checkpoints write. `setuptools<81` was the only genuine
dependency fix required.

At the default `batch_size: 128`, `mga_clap` and `slap` both reported success
while training zero batches. They are listed as passing only because they were
rerun at `--batch-size 8` and produced real gradient steps.

**Benign, do not chase:** SLAP logs
`Error(s) in loading state_dict for HTSATSwinTransformer: Unexpected key(s):
tscam_conv.weight, tscam_conv.bias, head.weight, head.bias`. Those are AudioSet's
classification head, which a retrieval encoder discards. The encoder weights
load correctly.

## Not verified by Phase A

- Convergence, or any number comparable to a baseline.
- MGA-CLAP's nine-language loop — only Hindi was run (`--languages Hindi`).
- Multi-GPU or distributed anything. `Not using distributed mode` throughout.
- SLAP's `paths.output_dir` override against a real Dora sweep. It works when
  the launcher sets it, which is all that was tested.
- Whether `transformers` 5.x changes any *output* — the classes import and MuRIL
  loads, but no forward-pass shape was compared against 4.x.
