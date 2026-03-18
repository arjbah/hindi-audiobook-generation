import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from datasets import load_dataset
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model Setup
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

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
df = pd.DataFrame(columns=["transcript", "gender", "emotion", "description", "rasa_audio", "indicparler_audio"])
index = 0
for gender in genders:
    for i in range(len(emotions)):
        emotion = emotions[i]
        prompt_emotion = prompt_emotions[i]
        subset = data[(data["gender"] == gender) & (data["style"] == emotion)]
        first_rows = subset[:8]
        for _, row in first_rows.iterrows():
            df.loc[index] = [row["text"], gender, emotion, f"A {gender.lower()} speaker expressing a lot of {prompt_emotion}. Use clear articulation, precise pronunciation, and expressive pitch variation where necessary.", row["audio"], None]
            index += 1

# IndicParler-TTS Eval
index = 0
for row in df.itertuples():
    prompt = row.transcript
    description = row.description

    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
    audio_arr = generation.cpu().numpy().squeeze()
    path = f"indicparler_out_v2/indic_tts_out_{index}.wav"
    sf.write(path, audio_arr, model.config.sampling_rate)
    with open(path, "rb") as f:
        audio_bytes = f.read()
    df.at[index, "indicparler_audio"] = {
        "bytes": audio_bytes,
        "path": path
    }
    index += 1
    print(f"Finished {index} audios")

df.to_csv("rasa_indicparler_comparison_v2.csv", index=False)