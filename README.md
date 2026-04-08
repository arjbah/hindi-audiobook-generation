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
- `{name}_drift.png` � drift panels
- `{name}_characters.png` � character panels (when using `--n_characters`)
- `comparison_drift.png` � overlay of multiple drift curves (when using `--compare`)

---

## Sentiment Alignment Evaluation (`sentiment_eval/sentiment_eval.py`)

Evaluates how well the **emotional tone of synthesised audio** matches the **sentiment of the source text**, on a per-segment basis.

### Models

| Modality | Model | Output |
|---|---|---|
| Text | `tabularisai/multilingual-sentiment-analysis` | 5-class label (Very Negative → Very Positive) |
| Audio | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Continuous valence / arousal / dominance ∈ [0, 1] |

Both are collapsed to a **3-class polarity** for direct comparison:

| Class | Text (stars) | Audio (valence) |
|---|---|---|
| Negative | 1–2 stars | < 0.4 |
| Neutral | 3 stars | 0.4 – 0.6 |
| Positive | 4–5 stars | > 0.6 |

### Usage

```bash
# From a captions JSON (Hindi story, one entry per segment)
python sentiment_eval/sentiment_eval.py \
    --json  transcripts/story120_multi_captions.json \
    --audio Audios/story120_multi.wav \
    --out_dir sentiment_eval/results

# From a plain text file (split on . ! ? ।)
python sentiment_eval/sentiment_eval.py \
    --text  transcripts/bad_blood_eng.txt \
    --audio Audios/Inference/IndicParler/ENG/elevenlabs_vampire_story.mp3 \
    --out_dir sentiment_eval/results
```

Additional flags:

| Flag | Default | Effect |
|---|---|---|
| `--min_silence_ms` | 300 | Silence gap used to split audio into segments (ms) |
| `--silence_thresh` | 40 | Energy threshold (top_db) for librosa silence split |
| `--valence_low` | 0.4 | Valence below this → audio polarity = negative |
| `--valence_high` | 0.6 | Valence above this → audio polarity = positive |
| `--save_segments` | off | Dump split WAV files to `<out_dir>/segments/` for inspection |

### Outputs

Four plots and one JSON file are written to `--out_dir`.

#### `<stem>_sentiment_timeline.png`

Two horizontal colour strips — one for text, one for audio — running left-to-right across the audio duration. Each segment is coloured **red (negative)**, **grey (neutral)**, or **green (positive)**. Look for consistent colour agreement between the two strips; mismatches (e.g. green on top, red on bottom) highlight segments where the synthesiser's emotional delivery diverges from the source text's tone.

#### `<stem>_sentiment_scatter.png`

Scatter plot with the text numeric score on the X-axis (−1 = very negative, +1 = very positive) and the audio valence score on the Y-axis (same scale). Points are coloured by the **text** polarity class. A dashed regression line is drawn when the data has sufficient variance.

- **Ideal result:** points cluster along the diagonal (high text score → high audio valence).
- **Flat / horizontal spread:** the synthesiser is not capturing sentiment variation from the text.
- **Pearson r and Spearman ρ** (shown in the title) quantify the linear and rank correlation respectively. Values above ~0.4 indicate meaningful alignment.

#### `<stem>_sentiment_confusion.png`

3×3 confusion matrix where **rows = text polarity** and **columns = audio polarity**. Counts on the diagonal are matches; off-diagonal cells are disagreements.

- High values on the diagonal → good agreement.
- A column dominated by "neutral" (e.g. most audio segments land in the neutral column regardless of text polarity) is a common sign that the TTS model is under-expressing emotional range.
- **Cohen's κ** (shown in the title) corrects for chance agreement: κ > 0.4 = moderate, > 0.6 = substantial.

#### `<stem>_sentiment_distribution.png`

Grouped bar chart comparing the proportion of negative / neutral / positive segments between text and audio. Use this to spot **systematic bias**: if the audio bar for "neutral" is much taller than the text bar, the model is collapsing emotional range toward a flat delivery. The gap between the two bars for each polarity class is the distributional mismatch.

#### `<stem>_sentiment_eval.json`

Structured results file with two sections:

- **`aggregate`** — all scalar metrics: `agreement`, `cohen_kappa`, `pearson_r`, `pearson_p`, `spearman_r`, `spearman_p`, polarity distribution counts, and the raw confusion matrix.
- **`segments`** — one entry per aligned segment containing the source text, the full text sentiment result (label, stars, confidence, numeric score, polarity), the full audio sentiment result (valence, arousal, dominance, numeric score, polarity), and a boolean `sentiment_match` flag.

### Interpreting aggregate metrics

| Metric | What it measures | Good threshold |
|---|---|---|
| Agreement | Fraction of segments where text and audio polarity match | > 0.5 |
| Cohen's κ | Agreement corrected for chance | > 0.4 (moderate) |
| Pearson r | Linear correlation of continuous scores | > 0.4 |
| Spearman ρ | Rank correlation (robust to outliers) | > 0.4 |

Low agreement with high neutral-audio proportion typically means the TTS model is producing emotionally flat audio regardless of the text's tone — a useful signal for prompt engineering or caption tuning.

---

## Reference

Data source: [YouTube Link](https://www.youtube.com/@storicokids)