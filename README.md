# Speaker Drift & Character Identity Evaluation

Minimal README for the evaluation scripts in this repository. The tools compute speaker identity drift and (optionally) auto-cluster character voices using Resemblyzer embeddings.

## Dependencies (concise)

| Package | Install |
|---|---|
| `resemblyzer` | `pip install resemblyzer` |
| `webrtcvad-wheels` | `pip install webrtcvad-wheels` |
| `librosa` | `pip install librosa` |
| `numpy` | `pip install numpy` |
| `scipy` | `pip install scipy` |
| `matplotlib` | `pip install matplotlib` |
| `scikit-learn` | `pip install scikit-learn` |
| `torch` | `pip install torch` |
| `umap-learn` (optional) | `pip install umap-learn` |
| `soundfile` | `pip install soundfile` |
| `praat-parselmouth` | `pip install praat-parselmouth` |
| `speechbrain` | `pip install speechbrain` |

Notes:
- `umap-learn` is optional; scripts fall back to t-SNE or PCA.
- Silero-VAD is loaded via `torch.hub` when available (no extra pip package required).

Full install suggestion:

```
pip install resemblyzer webrtcvad-wheels librosa numpy scipy matplotlib scikit-learn torch umap-learn soundfile praat-parselmouth speechbrain
```

## Quick overview
- `speaker_drift_resemblyzer.py`: compute anchor deviation and sequential drift; outputs `{name}_drift.png`.
- `speaker_drift_resemblyzer_withcharacters.py`: drift + optional character clustering (`--n_characters`); outputs `{name}_drift.png` and `{name}_characters.png`.
- `speaker_cluster_eval.py`: evaluate k-means clustering against ground-truth JSON segments (uses silence split / Silero-VAD).
- `indic_narrator_pipeline.py`: streaming narrator pipeline using Praat F0 checks and ECAPA (speechbrain) embeddings.

## Usage

```
# Drift only
python speaker_drift_resemblyzer.py --audio story.wav

# Drift + character clustering (provide number of clusters)
python speaker_drift_resemblyzer_withcharacters.py --audio story.wav --n_characters 4

# Compare multiple files
python speaker_drift_resemblyzer.py --audio human.wav tts_output.wav --compare
```

## Output
- `{name}_drift.png` — drift panels
- `{name}_characters.png` — character panels (when using `--n_characters`)
- `comparison_drift.png` — overlay of multiple drift curves (when using `--compare`)

## Reference

Data source: [YouTube Link](https://www.youtube.com/@storicokids)