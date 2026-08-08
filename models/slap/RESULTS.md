# SLAP on Rasa (Hindi) — Results Log

## Run 1: baseline reproduction (2026-07-08 to 2026-07-10)

**Config:** `data=rasa model=slap model/audio_encoder=htsat_audioset_slap model/text_encoder=muril_slap trainer=hindi logger=csv`

| Setting | Value |
|---|---|
| Train samples | 25,713 (Rasa Hindi train split) |
| Test/val samples | 2,858 (same split used for both val and test — see caveat below) |
| Physical batch size | 8 |
| Effective batch size | 64 (accumulate_grad_batches=8) |
| EMA tau | 0.995 (fixed) |
| Epochs | 120 (completed in full) |
| Audio augmentation | none |
| Audio caching | none (decoded/resampled fresh every epoch) |
| Checkpoints | `logs/xps/e85927c5/checkpoints/` (epoch89/99/109/119, best_99, last) |
| Metrics CSV | `logs/xps/e85927c5/csv/version_0/metrics.csv` |

### Val loss trajectory

| Epoch | Val loss |
|---|---|
| 9 | 0.636 |
| 19 | 0.577 |
| 29 | 0.509–0.512 |
| 39 | 0.484 |
| 49 | 0.475 |
| 59 | 0.464 |
| 69 | 0.455 (best in that window) |
| 89 | 0.463 |
| 99 | **0.452 (best overall)** |
| 109 | not top-1 (regressed slightly) |
| 119 | not top-1 (regressed slightly) |

### Final test retrieval metrics (2,858-sample test split)

Using **predictions** (BYOL predictor head output — the one to use for actual retrieval):

| Metric | Epoch 99 ckpt | Epoch 119 ckpt (final) |
|---|---|---|
| A→T R@1 | 56.2% | 56.0% |
| A→T R@5 | 80.8% | 81.0% |
| A→T R@10 | 87.0% | 87.5% |
| T→A R@1 | 57.2% | 57.5% |
| T→A R@10 | 86.6% | 86.8% |
| AUC (both directions) | ~0.993 | ~0.993 |

Using **raw projections** (before the predictor head) — notably weaker:

| Metric | Epoch 119 |
|---|---|
| A→T R@1 | 34.5% |
| A→T R@10 | 76.2% |
| T→A R@1 | 34.4% |
| T→A R@10 | 73.0% |

**Modality gap** (lower = more aligned): projections have `Linear Separability ≈ 1.0` (audio/text fully separable — large gap), predictions have `≈ 0.62-0.70` (much better aligned). Confirms the predictor head is doing real cross-modal alignment work that the raw projector isn't.

Epoch 99 vs 119 are essentially tied — the "best val loss" checkpoint didn't translate to meaningfully better retrieval than the final epoch.

### Comparison to in-house baselines (same Rasa test set)

| Model | T2A R@1 | A2T R@1 | R@5 |
|---|---|---|---|
| VoiceCLAP (`models/voiceclap/train_log.txt`) | ~0.93 | ~0.93 | ~0.997 |
| MGA-CLAP (`models/mga_clap/outputs/.../output.txt`) | ~0.97 | ~0.97 | ~1.00 |
| **SLAP (this run, predictions)** | 0.57 | 0.56 | 0.81 |

Both contrastive baselines are near-saturated; SLAP is well behind. Root causes (see conversation for full discussion):
1. BYOL has no negative-pair repulsion term — the mechanism contrastive losses use to directly optimize retrieval separability. On a highly-separable dataset like Rasa, that gap is structural, not just undertraining.
2. This run used ~10x less data (25.7k vs the paper's 260k pairs), ~12x smaller effective batch (64 vs 768), no SpecAugment
3. SLAP's own paper reports modest absolute retrieval numbers (single-digit % R@1) on its target domain (music-caption retrieval) — it was never shown to approach saturation anywhere, so this gap isn't a sign of a broken reproduction on its own.




