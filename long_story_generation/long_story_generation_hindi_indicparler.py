# This is also the ablation used to compare against Pranav's audio

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from dotenv import load_dotenv
import os
load_dotenv()
from indicnlp.tokenize import sentence_tokenize

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
speaker_name = "Rohit"
folder_name = f"../data/hindi_story_output_{speaker_name}"
os.makedirs(folder_name, exist_ok=True)
max_chars = 280

with open("../data/hindi_story_transcript.txt", "r") as f:
    transcript = f.read()
transcript = transcript.replace("\n", " ")

# Chunking the text
sentences = sentence_tokenize.sentence_split(transcript, lang="hi")
chunks = []
current_chunk = ""
current_len = 0

for sentence in sentences:
    sentence_length = len(sentence)
    if current_len + sentence_length > max_chars:
        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
            current_len = sentence_length
        else:
            chunks.append(sentence)
            current_chunk = ""
            current_len = 0
    else:
        if current_chunk:
            current_chunk += " " + sentence
            sentence_length += 1 # Accounting for the space
        else:
            current_chunk = sentence
        current_len += sentence_length
    
if current_chunk:
    chunks.append(current_chunk.strip())

print(f"# of chunks: {len(chunks)}")

# IndicParler model setup
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

batch_size = 8
batch_index = 0
description_input_ids = description_tokenizer(f"{speaker_name} speaks at a slightly slow pace with a deep, low-pitched voice and a narrow pitch range in a very close-sounding studio environment. The audio is of excellent quality with no background noise. The intended style is Narration. Delivered in a calm, clear narrative voice.", return_tensors="pt").to(device)
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    prompt_input_ids = tokenizer(batch, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids.repeat(len(batch), 1),
            attention_mask=description_input_ids.attention_mask.repeat(len(batch), 1),
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )
    print(f"Finished generating batch {batch_index}. Shape: {generation.shape}")
    batch_index += 1
    for j, audio in enumerate(generation):
        audio_arr = audio.cpu().numpy().squeeze()
        index = i + j
        indicparler_path = f"{folder_name}/indicparler_generation_{index}_{max_chars}.wav"
        sf.write(indicparler_path, audio_arr, model.config.sampling_rate)
        print(f"Finished IndicParler {index}")