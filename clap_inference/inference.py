from indicnlp.tokenize import sentence_tokenize
import torch
from train_clap import HTSATConfig
from clap_model import CLAPModel
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
import soundfile as sf
from TTS.api import TTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = HTSATConfig()
model = CLAPModel(config).to(device)
state_dict = torch.load("../clap_training/clap_model_epoch_120.pt", map_location=device)
model.load_state_dict(state_dict)
model.train()

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
reference_library = torch.load("reference_library.pt")
for ref in reference_library:
    ref["embedding"] = ref["embedding"].to(device)
all_embeddings = torch.stack([r["embedding"] for r in reference_library]).to(device)

folder_name = "speaker_clap"
def inference(transcript: str, is_rasa: bool = False, specified_output_path: str = None):
    os.makedirs(folder_name, exist_ok=True)
    if is_rasa:
        tokens = tokenizer(transcript, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        text_embedding = model.encode_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))

        similarities = torch.cosine_similarity(
            text_embedding,
            all_embeddings,
            dim=1
        )

        best_idx = torch.argmax(similarities).item()
        best_match = reference_library[best_idx]

        audio_path = best_match["audio_path"]

        tts_model.tts_to_file(
            text=transcript,
            speaker_wav=audio_path,
            language="hi",
            file_path=specified_output_path
        )
        return
    
    sentences = sentence_tokenize.sentence_split(transcript, lang="hi")
    sentences = [s.strip() for s in sentences if s.strip()]
    """chunks = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences):
            chunks.append(sentences[i] + " " + sentences[i + 1])
            i += 2
        else:
            chunks.append(sentences[i])
            i += 1

    print(f"# of chunks: {len(chunks)}")"""

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

        output_path = f"{folder_name}/{sentence_idx}.wav"

        tts_model.tts_to_file(
            text=sentence,
            speaker_wav=["test.wav", audio_path],
            language="hi",
            file_path=output_path
        )
        print(f"Did {sentence_idx}")
    
    files = sorted(
        [f for f in os.listdir(folder_name) if f.endswith(".wav") and f != "final.wav"],
        key=lambda x: int(x.replace(".wav", ""))
    )

    audio_data = []
    sample_rate = None

    for file_name in files:
        data, sr = sf.read(f"{folder_name}/{file_name}")

        if sample_rate is None:
            sample_rate = sr

        audio_data.append(data)

    final_audio = np.concatenate(audio_data, axis=0)
    sf.write(f"{folder_name}/final.wav", final_audio, sample_rate)

with open("hindi_story_transcript.txt", "r") as f:
    text = f.read()
inference(text)