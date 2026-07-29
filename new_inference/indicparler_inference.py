import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from indicnlp.tokenize import sentence_tokenize
from pydub import AudioSegment
import soundfile as sf
from pathlib import Path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

speaker_name = "Rohit"
max_chars = 200
batch_size = 8

description = f"{speaker_name} speaks at a slightly slow pace with a deep, low-pitched voice and a narrow pitch range in a very close-sounding studio environment. The audio is of excellent quality with no background noise. The intended style is Narration. Delivered in a calm, clear narrative voice."

description_inputs = description_tokenizer(description, return_tensors="pt").to(device)

def inference(transcript, output_folder, lang_code):
    transcript = transcript.replace("\n", " ")

    sentences = sentence_tokenize.sentence_split(transcript, lang=lang_code)

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

        prompt_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

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

            output_path = output_folder / f"{index}.wav"

            sf.write(output_path, audio_arr, model.config.sampling_rate)

    files = sorted(output_folder.glob("*.wav"), key=lambda p: int(p.stem))

    combined = AudioSegment.empty()

    for file_name in files:
        combined += AudioSegment.from_wav(file_name)

    combined.export(output_folder / f"{output_folder.name}.wav", format="wav")

root = Path("text")
folders = [p for p in root.iterdir() if p.is_dir()]
LANG_CODES = {
    "assamese": "as",
    "bengali": "bn",
    "gujarati": "gu",
    "hindi": "hi",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "tamil": "ta",
    "telugu": "te",
}

for folder in folders:
    files = list(folder.rglob("*.txt"))

    for file in files:
        print(f"Starting {file}")

        with file.open("r", encoding="utf-8") as f:
            text = f.read()

        output_folder = Path("audio") / folder.name / "indicparler" / file.stem
        output_folder.mkdir(parents=True, exist_ok=True)

        inference(text, output_folder, LANG_CODES[folder.name])

        print(f"Finished {file}")