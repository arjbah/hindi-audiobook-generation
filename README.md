# Hindi Audiobook Generation

## Retrieval models

All four contrastive audio-text models live under [`models/`](models/), with one
subdirectory each, behind a single entrypoint:

```bash
python models/train.py --model laion_clap                  # gender=both
python models/train.py --model voiceclap --gender female
python models/train.py --model all --seed 42
```
See [`models/README.md`](models/README.md) for prerequisites, the `--gender` and
`--seed` semantics, baseline results, and MGA-CLAP's remaining hardcoded paths.

VoiceCLAP is the LAION-CLAP pipeline with the audio encoder swapped for
`laion/voiceclap-small-v2`. MGA-CLAP and SLAP are modified versions of the
upstream MGA-CLAP and Guinot SLAP repositories.

### On gender subsets

`--gender {male,female,both}` selects the Rasa subset used for training and
in-domain eval; IndicVoices is never filtered.

**Omitting the flag preserves each model's prior behavior, which differs between
them.** LAION-CLAP, VoiceCLAP and SLAP never filtered by gender and default to
`both`. MGA-CLAP filters Rasa to male — a deliberate choice, and note it trains
nine models, one per language, rather than one Hindi model like the others. Its
IndicVoices eval split stays male-only and capped at 2000 rows. Pass `--gender`
explicitly whenever you compare models against each other.

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
