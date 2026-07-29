import torch
from datasets import load_dataset
import torchaudio.functional as AF
import torchaudio
from ruamel.yaml import YAML
from mga_clap_training.models.ase_model import ASE
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

languages = ["assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi", "tamil", "telugu"]
with open("settings/pretrain.yaml", "r") as f:
    yaml = YAML(typ='safe', pure=True)
    config = yaml.load(f)

max_samples = 32000 * 10
for language in languages:
    model = ASE(config).to(device)
    state_dict = torch.load(f"../mga_clap_training/outputs/{language}/val_rasa_best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    reference_library = []

    count = 0
    with torch.no_grad():
        for split in ["train", "test"]:
            ds = load_dataset("ai4bharat/Rasa", language.capitalize(), split=split)
            ds = ds.filter(
                lambda item: item["gender"] == "Male",
                desc=f"Filtering {language} Rasa to Male",
            )
            print(f"Current dataset: {language.capitalize()}; Split: {split}; Length: {len(ds)}")

            for item in ds:
                audio = torch.tensor(item["audio"]["array"]).float()
                sr = item["audio"]["sampling_rate"]
                length_seconds = audio.shape[-1] / sr
                if length_seconds < 3:
                    print(f"Skipped {count} becauese it's too short")
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

                audio_path = Path(f"reference_audio/rasa_male_{language}_{count}.wav")
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(audio_path, audio_24k.unsqueeze(0).cpu(), 24000)

                if audio_32k.shape[0] > max_samples:
                    audio_32k = audio_32k[:max_samples]
                else:
                    audio_32k = torch.nn.functional.pad(audio_32k, (0, max_samples - audio_32k.shape[0]))

                embedding = model.encode_audio(audio_32k.unsqueeze(0).to(device))

                text = item["text"].strip()

                reference_library.append({
                    "id": f"rasa_male_{language}_{count}",
                    "audio_path": audio_path,
                    "transcript": text,
                    "embedding": embedding.squeeze(0).cpu()
                })
                count += 1

    torch.save(reference_library, f"rasa_male_{language}_reference_library.pt")