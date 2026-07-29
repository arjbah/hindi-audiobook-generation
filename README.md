# Hindi Audiobook Generation

## Retrieval models

All four contrastive audio-text models live under [`models/`](models/), with one
subdirectory each, behind a single entrypoint:

```bash
python models/train.py --model laion_clap                  # defaults: gender=both
python models/train.py --model voiceclap --gender female
python models/train.py --model all --seed 42
```

| Model | Directory | Runs today |
|---|---|---|
| LAION-CLAP | `models/laion_clap` | yes |
| VoiceCLAP | `models/voiceclap` | yes |
| MGA-CLAP | `models/mga_clap` | needs `HTSAT_AudioSet_Saved_6.ckpt` |
| SLAP | `models/slap` | needs its data module, config, marker file and checkpoint |

See [`models/README.md`](models/README.md) for prerequisites, the `--gender` and
`--seed` semantics, baseline results, and the full list of what SLAP is missing.

VoiceCLAP is the LAION-CLAP pipeline with the audio encoder swapped for
`laion/voiceclap-small-v2`. MGA-CLAP and SLAP are modified versions of the
upstream MGA-CLAP and Guinot SLAP repositories.

### On gender subsets

Training now defaults to `--gender both` — no gender bias — with `male` and
`female` available per run. Note that before this change **no model filtered by
gender at all**; all four loaded the Rasa Hindi split unmodified.

Whether Rasa Hindi even contains both genders is unsettled:
`rasa_model_comparison/rasa_model_comparison.py:35` asserts it has no female
samples. If that is right, `--gender female` fails immediately with a message
listing the values actually present. `models/README.md` explains how to check.

## Other folders

| Folder | Purpose |
|---|---|
| `clap_inference` | Inference with trained CLAP models |
| `clap_training` | Earlier CLAP training code |
| `long_story_generation` | Long-form story audio generation |
| `mos_inference` | MOS evaluation inference |
| `mos_server` | MOS listening-test server and configs |
| `rasa_model_comparison` | Comparison of TTS systems on Rasa |
| `sovits_voice_conversion` | SoVITS voice conversion |
