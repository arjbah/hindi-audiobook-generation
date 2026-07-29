import torch
from transformers import AutoModel
from indicnlp.tokenize import sentence_tokenize
import numpy as np
import soundfile as sf
from ruamel.yaml import YAML
import sys
from pathlib import Path
sys.path.append(str((Path(__file__).parent.parent).resolve()))
from mga_clap_training.models.ase_model import ASE
from pydub import AudioSegment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def inference(transcript: str, output_folder: Path, lang_code: str, model: ASE, tts_model, reference_library, all_embeddings: torch.Tensor):
    sentences = sentence_tokenize.sentence_split(transcript, lang=lang_code)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence_idx, sentence in enumerate(sentences):
        with torch.no_grad():
            text_embedding = model.encode_text(sentence).to(device)
            similarities = torch.cosine_similarity(text_embedding, all_embeddings, dim=1)

            best_idx = torch.argmax(similarities).item()
            best_match = reference_library[best_idx]

            audio_path = best_match["audio_path"]
            ref_text = best_match["transcript"]

            output_path = output_folder / f"{sentence_idx}.wav"

            audio = tts_model(sentence, ref_audio_path=audio_path, ref_text=ref_text)

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            sf.write(output_path, np.array(audio, dtype=np.float32), samplerate=24000)

    files = sorted(output_folder.glob("*.wav"), key=lambda p: int(p.stem))

    # Combining all audio outputs to a single file
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
with open("settings/pretrain.yaml", "r") as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f)

for folder in folders:
    model = ASE(config).to(device)
    state_dict = torch.load(f"../mga_clap_training/outputs/{folder.name}/val_rasa_best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    tts_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(device)
    reference_library = torch.load(f"rasa_male_{folder.name}_reference_library.pt")
    for ref in reference_library:
        ref["embedding"] = ref["embedding"].to(device)
    all_embeddings = torch.stack([r["embedding"] for r in reference_library]).to(device)

    files = list(folder.rglob("*.txt"))

    for file in files:
        print(f"Starting {file}")

        with file.open("r", encoding="utf-8") as f:
            text = f.read()

        output_folder = Path("audio") / folder.name / "indicf5" / file.stem
        output_folder.mkdir(parents=True, exist_ok=True)

        inference(text, output_folder, LANG_CODES[folder.name], model, tts_model, reference_library, all_embeddings)

        print(f"Finished {file}")