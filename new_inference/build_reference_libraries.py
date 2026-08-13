import torch
import torchaudio.functional as AF
import torchaudio
from ruamel.yaml import YAML
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MGA_DIR = MODELS_DIR / "mga_clap_training"
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
from data import AUDIO_DURATION, load_rasa
sys.path.insert(0, str(MGA_DIR))
from models.ase_model import ASE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

languages = ["assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi", "tamil", "telugu"]
with (MGA_DIR / "settings" / "pretrain.yaml").open() as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f)
config["device"] = str(device)

max_samples = 32000 * AUDIO_DURATION
for language in languages:
    model = ASE(config).to(device)
    checkpoint = torch.load(MGA_DIR / f"{language}_best_rasa.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    reference_library = []

    count = 0
    with torch.no_grad():
        for split in ["train", "test"]:
            ds = load_rasa(split, "male", language)
            print(f"Current dataset: {language.capitalize()}; Split: {split}; Length: {len(ds)}")

            for item in ds:
                audio = torch.tensor(item["audio"]["array"]).float()
                sr = item["audio"]["sampling_rate"]
                length_seconds = audio.shape[-1] / sr
                if length_seconds < 3:
                    print(f"Skipped {count} because it's too short")
                    count += 1
                    continue

                audio_24k = audio
                if sr != 24000:
                    audio_24k = AF.resample(audio_24k, sr, 24000)
                if audio_24k.ndim > 1:
                    audio_24k = audio_24k[0]

                audio_32k = audio
                if sr != 32000:
                    audio_32k = AF.resample(audio_32k, sr, 32000)
                if audio_32k.ndim > 1:
                    audio_32k = audio_32k[0]

                audio_path = OUTPUT_DIR / "reference_audio" / language / f"rasa_male_{count}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(audio_path, audio_24k.unsqueeze(0).cpu(), 24000)

                if audio_32k.shape[0] > max_samples:
                    audio_32k = audio_32k[:max_samples]
                else:
                    audio_32k = torch.nn.functional.pad(audio_32k, (0, max_samples - audio_32k.shape[0]))

                _, frame_embeddings = model.encode_audio(audio_32k.unsqueeze(0).to(device))
                embedding = model.msc(frame_embeddings, model.codebook)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)

                text = item["text"].strip()

                reference_library.append({
                    "id": f"rasa_male_{language}_{count}",
                    "audio_path": audio_path,
                    "transcript": text,
                    "embedding": embedding.squeeze(0).cpu()
                })
                count += 1

    torch.save(reference_library, OUTPUT_DIR / f"rasa_male_{language}_reference_library.pt")
