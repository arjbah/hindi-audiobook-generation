import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from indicnlp.tokenize import sentence_tokenize
from pydub import AudioSegment
import soundfile as sf
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

speaker_name = "Rohit"
max_chars = 200
batch_size = 8

description = f"{speaker_name} speaks at a slightly slow pace with a deep, low-pitched voice and a narrow pitch range in a very close-sounding studio environment. The audio is of excellent quality with no background noise. The intended style is Narration. Delivered in a calm, clear narrative voice."

description_inputs = description_tokenizer(
    description,
    return_tensors="pt"
).to(device)

def inference(transcript, text_file_name, folder):
    os.makedirs(folder, exist_ok=True)

    transcript = transcript.replace("\n", " ")

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
                sentence_length += 1
            else:
                current_chunk = sentence

            current_len += sentence_length

    if current_chunk:
        chunks.append(current_chunk.strip())

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        prompt_inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            generation = model.generate(
                input_ids=description_inputs.input_ids.repeat(len(batch), 1),
                attention_mask=description_inputs.attention_mask.repeat(len(batch), 1),
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask
            )

        for j, audio in enumerate(generation):
            audio_arr = audio.cpu().numpy().squeeze()

            index = i + j

            output_path = f"{folder}/{index}.wav"

            sf.write(
                output_path,
                audio_arr,
                model.config.sampling_rate
            )

            print(f"Finished {index}")

    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".wav")],
        key=lambda x: int(x.replace(".wav", ""))
    )

    combined = AudioSegment.empty()

    for file_name in files:
        combined += AudioSegment.from_wav(
            f"{folder}/{file_name}"
        )

    combined.export(
        f"indicparler/{text_file_name}.wav",
        format="wav"
    )

text_folder = "../mos_eval/story_transcripts"

for file_name in os.listdir(text_folder):
    if not file_name.endswith(".txt"):
        continue

    file_path = os.path.join(text_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    text_file_name = os.path.splitext(file_name)[0]

    inference(
        transcript,
        text_file_name,
        f"indicparler_temp/{text_file_name}"
    )

    print(f"Finished {file_name}")