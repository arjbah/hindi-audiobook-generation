import torch
import soundfile as sf
from dotenv import load_dotenv
import os
load_dotenv()
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize
import httpx
import io

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
folder_name = f"../data/english_story_output_voxtral"
os.makedirs(folder_name, exist_ok=True)
max_chars = 200

with open("../data/english_story_transcript.txt", "r") as f:
    transcript = f.read()
transcript = transcript.replace("\n", " ")

# Chunking the text
sentences = sent_tokenize(transcript)
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

index = 0
for chunk in chunks:
    path = f"{folder_name}/voxtral_generation_{index}_{max_chars}.wav"
    base_url = "http://localhost:8000/v1"
 
    payload = {
        "input": chunk,
        "model": "mistralai/Voxtral-4B-TTS-2603",
        "response_format": "wav",
        "voice": "casual_male",
    }
    
    response = httpx.post(f"{base_url}/audio/speech", json=payload, timeout=120.0)
    response.raise_for_status()
    
    audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
    sf.write(path, audio_array, sr)
    print(f"Finished Voxtral {index}")
    index += 1