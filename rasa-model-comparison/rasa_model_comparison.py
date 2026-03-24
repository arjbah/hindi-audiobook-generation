import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from datasets import load_dataset
import pandas as pd
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
folder_name = "../data/tts_model_out_v3"
os.makedirs(folder_name, exist_ok=True)

# Model Setup
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY_2")
client = ElevenLabs(api_key=elevenlabs_api_key)

# Dataset Setup
data = load_dataset("ai4bharat/Rasa", "Hindi")
data = data["train"]
data = data.to_pandas()
data["duration"] = data["duration"].astype(float)
data = data[data["duration"] >= 3]

# Extracting relevant Rasa values
genders = ["Male"] # Rasa Hindi has no female samples
emotions = ["HAPPY", "ANGER", "SURPRISE", "SAD", "FEAR", "DISGUST"]
prompt_emotions = ["happiness", "anger", "surprise", "sadness", "fear", "disgust"]
elevenlabs_prompt_emotions = ["happy", "angry", "surprised", "sad", "fearful", "disgusted"]
df = pd.DataFrame(columns=["transcript", "gender", "emotion", "description", "rasa_audio", "indicparler_audio", "elevenlabs_description", "elevenlabs_audio"])
index = 0
for gender in genders:
    for i in range(len(emotions)):
        emotion = emotions[i]
        prompt_emotion = prompt_emotions[i]
        elevenlabs_prompt_emotion = elevenlabs_prompt_emotions[i]
        subset = data[(data["gender"] == gender) & (data["style"] == emotion)]
        first_rows = subset[:8]
        for _, row in first_rows.iterrows():
            df.loc[index] = [row["text"], gender, emotion,
                             f"A {gender.lower()} speaker expressing a lot of {prompt_emotion}. Use clear articulation, precise pronunciation, and expressive pitch variation where necessary.",
                             row["audio"], None, f"very {elevenlabs_prompt_emotion}", None]
            index += 1

# IndicParler-TTS + ElevenLabs Eval
index = 0
for row in df.itertuples():
    # IndicParler
    transcript = row.transcript
    indicparler_path = f"{folder_name}/indic_tts_out_{index}.wav"
    if not Path(indicparler_path).exists():
        indicparler_description = row.description

        description_input_ids = description_tokenizer(indicparler_description, return_tensors="pt").to(device)
        prompt_input_ids = tokenizer(transcript, return_tensors="pt").to(device)

        generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(indicparler_path, audio_arr, model.config.sampling_rate)
    with open(indicparler_path, "rb") as f:
        indicparler_audio_bytes = f.read()
    df.at[index, "indicparler_audio"] = {
        "bytes": indicparler_audio_bytes,
        "path": indicparler_path
    }
    print(f"Finished {index} IndicParler")

    # ElevenLabs
    elevenlabs_path = f"{folder_name}/elevenlabs_out_{index}.wav"
    if not Path(elevenlabs_path).exists():
        elevenlabs_description = row.elevenlabs_description
        audio = client.text_to_speech.convert(
            text=f"[{elevenlabs_description}] {transcript}",
            voice_id="JBFqnCBsd6RMkjVDRZzb", # Voice: George - Warm, Captivating Storyteller
            model_id = "eleven_v3"
        )
        with open(elevenlabs_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
    with open(elevenlabs_path, "rb") as f:
        elevenlabs_audio_bytes = f.read()
    df.at[index, "elevenlabs_audio"] = {
        "bytes": elevenlabs_audio_bytes,
        "path": elevenlabs_path
    }
    print(f"Finished {index} ElevenLabs")
    index += 1

df.to_csv(f"{folder_name}/rasa_model_comparison.csv", index=False)