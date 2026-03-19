# Speaker Drift & Character Identity Evaluation
### Resemblyzer-based pipeline for long-form expressive TTS — StoriCo Benchmark

---

## Dependencies

| Package | Install | What it does in this project |
|---|---|---|
| `resemblyzer` | `pip install resemblyzer` | The core speaker encoder. Converts audio segments into 256-dimensional "voice fingerprints" (d-vectors) using a model trained to distinguish speakers. Every metric in this pipeline is computed from these fingerprints. |
| `webrtcvad-wheels` | `pip install webrtcvad-wheels` | Required by Resemblyzer internally for voice activity detection. Install **this** instead of `webrtcvad` — it ships pre-compiled binaries that work on Python 3.13 without breaking. |
| `librosa` | `pip install librosa` | Loads audio files and resamples them to 16 kHz mono, which is the format Resemblyzer expects. Handles any input format — mp3, wav, m4a, etc. |
| `numpy` | `pip install numpy` | Core math library. Used for all embedding arithmetic, distance calculations, running averages, and linear regression on the drift curve. |
| `scipy` | `pip install scipy` | Provides the cosine distance function used to measure how different two voice fingerprints are from each other. |
| `matplotlib` | `pip install matplotlib` | Generates all output charts — the drift curves, scatter plots, bar charts, timeline strip, and summary tables. |
| `scikit-learn` | `pip install scikit-learn` | Powers two things: k-means clustering (auto-detecting character voices) and t-SNE/PCA projection (visualising embeddings in 2D if UMAP is not installed). |
| `torch` | `pip install torch` | Required by Resemblyzer to run the speaker encoder model. Automatically uses your GPU if one is available. |
| `umap-learn` | `pip install umap-learn` | Optional but recommended. Produces better 2D projections of the voice embeddings than t-SNE. If not installed, the pipeline falls back to t-SNE, then PCA. |

**Full install command:**
```bash
pip install resemblyzer webrtcvad-wheels librosa numpy scipy matplotlib scikit-learn torch umap-learn
```

---

## Metrics Explained

The pipeline produces two sets of outputs depending on whether you run drift-only or with `--n_characters`. Here is every number and chart explained in plain terms.

---

### Drift Metrics
*These run on every audio file regardless of mode.*

---

**Anchor Embedding**

Before any drift is measured, the pipeline listens to the first ~20 seconds of audio (the first 5 segments by default) and averages those voice fingerprints together into a single reference point. This is the "anchor" — it represents what the speaker is supposed to sound like. Everything else is measured against it.

---

**Anchor Deviation** *(the main signal)*

For every 4-second window of audio, the pipeline asks: *how different does the voice sound right now compared to how it sounded at the start?*

The answer is a number between 0 and 1. Zero means the voice is identical to the anchor — no drift. Values around 0.10–0.15 are where drift starts to become subtly noticeable to a human listener. Values above 0.25–0.30 indicate substantial identity change — the voice genuinely sounds like it belongs to a different person.

---

**Mean Anchor Deviation**

The average drift across the entire audio. This is the single headline number for how consistent the speaker identity was overall. Lower is better.

*Example: a score of 0.05 means the voice barely moved from its starting identity throughout the story. A score of 0.20 means there was meaningful drift on average.*

---

**Max Anchor Deviation**

The worst single moment of drift across the whole file. Useful for knowing whether problems are isolated (a spike at one point) or widespread (a high max alongside a high mean).

---

**Std Anchor Deviation**

How much the drift fluctuated up and down. A low standard deviation alongside a high mean means the voice drifted smoothly and steadily — it quietly became a different voice over time. A high standard deviation means the drift was erratic — the voice jumped around, which usually points to chunk boundary issues where the TTS model is being fed the audio in pieces and losing track of the speaker between chunks.

---

**Peak Drift Time**

The exact timestamp (in seconds) where drift was at its worst. This is marked on the chart with a yellow dotted line. You can seek to this point in the audio file and listen to confirm whether the drift is actually audible.

---

**Drift Rate (per second)**

A single number describing the *direction and speed* of drift over time, calculated by fitting a straight line through the drift curve.

- **Positive** ? the voice is gradually drifting away from the anchor as the audio progresses. The larger the number, the faster the drift.
- **Near zero** ? the voice is stable throughout.
- **Negative** ? the voice is actually getting more consistent over time (rare, but possible if the model "settles in").

This is the clearest indicator of whether drift is a systematic problem with the model's long-range conditioning, versus random noise.

---

**Sequential Drift**

Instead of comparing each segment to the fixed anchor, this compares each segment to the one *immediately before it*. It measures local turbulence — how much the voice jumps between adjacent windows.

Sudden spikes in the sequential drift chart at regular intervals are a diagnostic red flag for chunking artifacts: the TTS model is generating audio in pieces, and the speaker identity resets slightly at each boundary.

---

**Mean / Max Sequential Drift**

Summary statistics for the local jump measurements. The mean captures overall volatility; the max captures the worst single jump.

---

**Instability Score**

The variance (spread) of the sequential drift values. This distinguishes two different failure modes that can look similar in the headline numbers:

- **Low instability score + rising anchor deviation** ? the voice is slowly, smoothly becoming someone else. This is a context window or conditioning problem in the model.
- **High instability score + moderate anchor deviation** ? the voice is jumping around erratically. This is a chunking or stitching problem.

These two failure modes have different causes and require different fixes, so this metric is useful for diagnosis even when the mean drift scores look similar.

---

**Cumulative Drift (Running Mean)**

A smoothed trend line that shows the average drift up to each point in time. Unlike the raw anchor deviation curve which is noisy, this one only ever moves in the direction of the overall trend, making it the clearest visual for presentations — a flat line means a stable voice, an upward curve means the identity is slowly degrading.

---

**Drift Distribution (Histogram)**

Shows all the individual segment drift scores spread out as a distribution. A tight cluster near zero means the voice was consistently stable throughout. A wide spread or a long tail on the right means there were many moments where the voice moved significantly away from the anchor identity.

---

### Character Identity Metrics
*These only appear when running with `--n_characters N`.*

---

**Auto Character Clustering**

The pipeline runs k-means clustering on all the voice fingerprints, grouping similar-sounding segments together. It does not need you to label when each character speaks — it finds the groups automatically. This matches exactly how the StoriCo paper identified the 4 natural voice clusters in their dataset.

Clusters are labelled `character_0`, `character_1`, etc. in the order they first appear in the audio, so `character_0` will almost always be the narrator since they speak first.

---

**Embedding Projection Scatter Plot**

Each dot is one audio segment, coloured by which character cluster it belongs to. The star marker is the centroid (average position) of each cluster.

*How to read it:* tight, well-separated clusters of the same colour mean the model is producing consistent, distinct voices for each character. Overlapping clusters of different colours mean two characters are being rendered with voices that are too acoustically similar to tell apart. This is the visual equivalent of the StoriCo Figure 1 in the paper.

---

**Cluster Sequence Timeline Strip**

A colour-coded strip along the bottom of the character panel showing which cluster was active at each moment in the audio. This makes it immediately visible how frequently the narrator switches between characters and whether the clustering is picking up natural dialogue patterns or behaving erratically.

---

**Intra-Character Consistency**

For each character, this measures how similar that character's voice fingerprints are to each other across all their appearances in the story.

A score of 1.0 means the character always sounded exactly the same every time they spoke. A score of 0.75 is a reasonable minimum threshold — below this, a character's voice is inconsistent enough that listeners would notice something is off. This is the key metric for evaluating whether a TTS model maintains character identity across a long story.

---

**Inter-Character Distance**

Measures how acoustically different each pair of characters is from each other, by comparing their centroid fingerprints.

Higher distance = more distinct voices. The important thing to understand here is not the absolute number but the **relative comparison**: character voices should be meaningfully separated from each other (you want non-trivial distance) but still much closer to each other than two completely different speakers would be. They are voice modulations by the same person, not separate identities — so some similarity is expected and correct.

In practice for the StoriCo narrator, two different character voices from the same narrator typically land around 0.10–0.25 cosine distance. Two completely unrelated speakers would typically be 0.35–0.65.

---

## Output Files

| File | Contents |
|---|---|
| `{name}_drift.png` | 5-panel drift chart: anchor deviation over time, summary stats table, segment-to-segment drift, cumulative drift, drift distribution histogram |
| `{name}_characters.png` | 4-panel character chart: embedding projection scatter, timeline colour strip, intra-consistency bars, inter-character distance table |
| `comparison_drift.png` | Overlaid drift curves across multiple files (only with `--compare`) |

---

## Usage

```bash
# Drift only
python speaker_drift_resemblyzer.py --audio story.wav

# With auto character detection
python speaker_drift_resemblyzer.py --audio story.wav --n_characters 4

# Compare two files side by side
python speaker_drift_resemblyzer.py --audio human.wav tts_output.wav --compare
```

---

## Reference

Data source: [YouTube Link](https://www.youtube.com/@storicokids)