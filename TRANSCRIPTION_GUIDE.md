# Hindi TTS Transcription Guide

## Overview
Evaluate IndicParler-TTS generated audio by transcribing with MMS ASR and translating with IndicTrans2. Computes Word Error Rate (WER) against original transcripts.

## Setup

### Install Dependencies
```bash
pip install torch torchaudio transformers Levenshtein IndicTransToolkit sentencepiece
```

### Authenticate Hugging Face
```bash
huggingface-cli login
# Required for IndicTrans2 (gated model)
```

## Configuration

Edit `transcribe.py` to match my setup:

```python
# Path to test.csv with original Hindi transcripts
DATA_DIR = "StoricoTTSDataset"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Path to generated audio files
GENERATED_AUDIO_DIR = "evaluation_output/generated"

# Models (default works well)
ASR_MODEL = "facebook/mms-1b-all"
TRANSLATION_MODEL = "ai4bharat/indictrans2-indic-en-1B"
```

## Usage

### Single Audio File
```bash
python transcribe.py --audio-file <path_to_audio>.wav
```

**Example:**
```bash
python transcribe.py --audio-file evaluation_output/generated/8000_story119_0000_gen.wav
```

### Use CPU (slower)
```bash
python transcribe.py --audio-file <path>.wav --device cpu
```

## Output

### Console Output
```
======================================================================
File: 8000_story119_0000_gen.wav
======================================================================

[ASR Transcription of Generated Audio]
Hindi: हलो बच्चो आज जो कहानी मैं आपको सुनाने जा रही हूँ उसका टॉपिक है सबने की सिख
English: Hello children, the topic of the story I am going to tell you today is Everyone's Sikh.

[Original Transcript]
Hindi: हेलो बच्चों आज जो कहानी मैं आपको सुनाने जा रही हूं उसका टॉपिक है सपने की सीख
English: Hello children, the topic of the story I am going to tell you today is learning to dream.

[Word Error Rate]
WER: 29.41%
Details: S=5, D=0, I=0, N=17
```

### JSON Results
Saved to: `evaluation_output/asr_accuracy_results.json`

Contains per-segment transcriptions, translations, WER scores, and aggregate statistics.

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| `facebook/mms-1b-all` | Hindi ASR (Devanagari output) | 1B |
| `ai4bharat/indictrans2-indic-en-1B` | Hindi→English translation | 1B |

## Troubleshooting

**IndicTrans2 access denied:** Request access at https://huggingface.co/ai4bharat/indictrans2-indic-en-1B

**Missing test.csv:** Ensure `DATA_DIR` points to correct location
