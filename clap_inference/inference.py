import os
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer
from clap_model import CLAPModel  # Config import removed
from TTS.api import TTS

# Using Indic NLP sentence tokenizer
try:
    from indicnlp.tokenize import sentence_tokenize
except ImportError:
    # Fallback if indicnlp is not installed in the venv
    os.system("pip install indicnlp")
    from indicnlp.tokenize import sentence_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize custom CLAP model cleanly without HTSATConfig
model = CLAPModel().to(device)


checkpoint_path = "../clap_training/clap_checkpoint_epoch_120.pt"
if not os.path.exists(checkpoint_path):
    checkpoint_path = "../clap_training/clap_model_epoch_120.pt"

print(f"Loading CLAP weights from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict, strict=False)

model.eval()

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

# Initialize the XTTS-v2 zero-shot model
print("Loading XTTS-v2 zero-shot synthesis model...")
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Load precomputed reference library embeddings (Rasa only)
print("Loading precomputed reference library...")
reference_library = torch.load("reference_library.pt", map_location=device)
for ref in reference_library:
    ref["embedding"] = ref["embedding"].to(device)
all_embeddings = torch.stack([r["embedding"] for r in reference_library]).to(device)

folder_name = "speaker_clap"

def inference(transcript: str, is_rasa: bool = False, specified_output_path: str = None):
    os.makedirs(folder_name, exist_ok=True)
    
    if is_rasa:
        tokens = tokenizer(transcript, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            text_embedding = model.encode_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))

        similarities = torch.cosine_similarity(
            text_embedding,
            all_embeddings,
            dim=1
        )

        best_idx = torch.argmax(similarities).item()
        best_match = reference_library[best_idx]
        audio_path = best_match["audio_path"]

        print(f"Matched Rasa Clip: {audio_path}")
        tts_model.tts_to_file(
            text=transcript,
            speaker_wav=audio_path,
            language="hi",
            file_path=specified_output_path
        )
        return
    
    # Segment target story transcript into sentences
    sentences = sentence_tokenize.sentence_split(transcript, lang="hi")
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence_idx, sentence in enumerate(sentences):
        tokens = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        
        # Calculate semantic embedding using your newly trained CLAP
        with torch.no_grad():
            text_embedding = model.encode_text(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))

        # Retrieve the best matching expressive audio clip from the reference library
        similarities = torch.cosine_similarity(
            text_embedding,
            all_embeddings,
            dim=1
        )

        best_idx = torch.argmax(similarities).item()
        best_match = reference_library[best_idx]
        audio_path = best_match["audio_path"]

        output_path = f"{folder_name}/{sentence_idx}.wav"

        # Check if local 'test.wav' exists to prevent FileNotFoundError
        if os.path.exists("test.wav"):
            speaker_references = ["test.wav", audio_path]
        else:
            # Fallback to using only the retrieved clip as the style reference
            speaker_references = audio_path

        # Generate zero-shot sentence-level narration
        tts_model.tts_to_file(
            text=sentence,
            speaker_wav=speaker_references,
            language="hi",
            file_path=output_path
        )
        print(f"Generated sentence {sentence_idx}/{len(sentences)-1} with style reference: {audio_path}")
    
    # Concatenate all sentence wav files into a single, seamless audiobook
    files = sorted(
        [f for f in os.listdir(folder_name) if f.endswith(".wav") and f != "final.wav"],
        key=lambda x: int(x.replace(".wav", ""))
    )

    audio_data = []
    sample_rate = None

    print("Concatenating clips into final audiobook...")
    for file_name in files:
        data, sr = sf.read(f"{folder_name}/{file_name}")

        if sample_rate is None:
            sample_rate = sr

        audio_data.append(data)

    final_audio = np.concatenate(audio_data, axis=0)
    sf.write(f"{folder_name}/final.wav", final_audio, sample_rate)
    print(f"Success! Final expressive audiobook saved to: {folder_name}/final.wav")

if __name__ == "__main__":
    # Ensure dependencies for indicnlp sentence segmenter are initialized
    from indicnlp import common
    from indicnlp import loader
    # Check if resources are downloaded, if not download them automatically
    # indicnlp requires resources folder to segment sentences
    
    with open("hindi_story_transcript.txt", "r") as f:
        text = f.read()
    inference(text)
