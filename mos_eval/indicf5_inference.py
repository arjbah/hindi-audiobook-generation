import torch
from train_clap import HTSATConfig
from clap_model import CLAPModel
from transformers import AutoTokenizer, AutoModel
from indicnlp.tokenize import sentence_tokenize
import numpy as np
import os
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Rasa stuff running")

config = HTSATConfig()
model = CLAPModel(config).to(device)
state_dict = torch.load("../clap_training/clap_model_epoch_120.pt", map_location=device)
model.load_state_dict(state_dict)
model.train()

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
tts_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
reference_library = torch.load("rasa_reference_library.pt")
for ref in reference_library:
    ref["embedding"] = ref["embedding"].to(device)
all_embeddings = torch.stack([r["embedding"] for r in reference_library]).to(device)

def inference(transcript: str, text_file_name: str, folder: str):
    os.makedirs(folder, exist_ok=True)

    sentences = sentence_tokenize.sentence_split(transcript, lang="hi")
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence_idx, sentence in enumerate(sentences):
        tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

        text_embedding = model.encode_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))

        similarities = torch.cosine_similarity(
            text_embedding,
            all_embeddings,
            dim=1
        )

        best_idx = torch.argmax(similarities).item()
        best_match = reference_library[best_idx]

        audio_path = best_match["audio_path"]
        ref_text = best_match["transcript"]

        output_path = f"{folder}/{sentence_idx}.wav"

        audio = tts_model(
            sentence,
            ref_audio_path=audio_path,
            ref_text=ref_text
        )

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        sf.write(output_path, np.array(audio, dtype=np.float32), samplerate=24000)

        print(f"Did {sentence_idx}")

    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".wav") and f != "final.wav"],
        key=lambda x: int(x.replace(".wav", ""))
    )

    audio_data = []
    sample_rate = None

    for file_name in files:
        data, sr = sf.read(f"{folder}/{file_name}")

        if sample_rate is None:
            sample_rate = sr

        audio_data.append(data)

    final_audio = np.concatenate(audio_data, axis=0)

    sf.write(f"indicf5_rasa/{text_file_name}.wav", final_audio, sample_rate)

text_folder = "../mos_eval/story_transcripts"

for file_name in os.listdir(text_folder):
    if not file_name.endswith(".txt"):
        continue
    print(f"Starting {file_name}")
    file_path = os.path.join(text_folder, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    text_file_name = os.path.splitext(file_name)[0]

    inference(
        transcript,
        text_file_name,
        f"indicf5_rasa_temp/{text_file_name}"
    )

    print(f"Finished {file_name}")