from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import librosa
from evaluate import load
from pathlib import Path
import json
import re

def normalize(text):
    text = text.replace("\n", " ")
    text = text.replace("।", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
model = model.to(device)
processor.tokenizer.set_target_lang("hin")
model.load_adapter("hin")

transcripts = {}
transcript_path = Path("story_transcripts")
for file in transcript_path.iterdir():
    with open(file, "r") as f:
        transcripts[file.stem] = f.read()

wer_metric = load("wer")
cer_metric = load("cer")

mos_folder = Path("mos")
metrics = {}
for folder in mos_folder.iterdir():
    if not folder.is_dir():
        continue

    folder_name = folder.name
    curr_wers = []
    curr_cers = []
    print(f"Starting {folder_name}")
    for audio_sample in folder.iterdir():
        print(f"Starting {audio_sample.stem}")
        if audio_sample.suffix.lower() not in [".wav", ".mp3"]:
            continue

        audio, sr = librosa.load(audio_sample, sr=16000)
        inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs).logits
        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = processor.decode(ids)

        reference = normalize(transcripts[audio_sample.stem])
        prediction = normalize(transcription)
        wer = wer_metric.compute(references=[reference], predictions=[prediction])
        cer = cer_metric.compute(references=[reference], predictions=[prediction])
        print(f"WER: {wer}")
        print(f"CER: {cer}")
        curr_wers.append(wer)
        curr_cers.append(cer)

    metrics[folder_name] = {}
    metrics[folder_name]["wers"] = curr_wers
    metrics[folder_name]["cers"] = curr_cers
    metrics[folder_name]["wer"] = sum(curr_wers) / len(curr_wers)
    metrics[folder_name]["cer"] = sum(curr_cers) / len(curr_cers)

with open("metrics2.json", "w") as f:
    json.dump(metrics, f, indent=4)