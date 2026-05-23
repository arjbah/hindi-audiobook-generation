from train_clap import HTSATConfig
from clap_model import CLAPModel
import torch
from datasets import load_dataset
import torchaudio.functional as AF
import os
import torchaudio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("reference_audio", exist_ok=True)

config = HTSATConfig()
model = CLAPModel(config).to(device)
state_dict = torch.load("../clap_training/clap_model_epoch_120.pt", map_location=device)
model.load_state_dict(state_dict)
model.train()

target_sr = 32000
max_samples = int(target_sr * 10.24)

reference_library = []

datasets_to_load = [
    ("indicvoices", "ai4bharat/indicvoices_r", "Hindi"),
    ("rasa", "ai4bharat/Rasa", "Hindi")
]

count = 0
with torch.no_grad():
    for ds_id, hf_name, subset in datasets_to_load:
        for split in ["train", "test"]:
            ds = load_dataset(hf_name, subset, split=split)
            print(f"Current dataset: {ds_id}; Length: {len(ds)}")

            for item in ds:
                audio = torch.tensor(item["audio"]["array"]).float()
                sr = item["audio"]["sampling_rate"]
                length_seconds = audio.shape[-1] / sr
                if length_seconds < 3:
                    print(f"Skipped {count}")
                    count += 1
                    continue
                if sr != target_sr:
                    audio = AF.resample(audio, sr, target_sr)
                
                if len(audio.shape) > 1:
                    audio = audio[0]

                original_audio = AF.resample(audio, target_sr, 24000)
                audio_path = f"reference_audio/{ds_id}_{count}.wav"
                torchaudio.save(audio_path, original_audio.unsqueeze(0).cpu(), 24000)
                
                if audio.shape[0] > max_samples:
                    audio = audio[:max_samples]
                else:
                    audio = torch.nn.functional.pad(audio, (0, max_samples - audio.shape[0]))

                embedding = model.encode_audio(audio.unsqueeze(0).to(device))

                text = (item["normalized"].strip() if ds_id == "indicvoices" else item["text"].strip())

                reference_library.append({
                    "id": f"{ds_id}_{count}",
                    "audio_path": audio_path,
                    "transcript": text,
                    "embedding": embedding.squeeze(0).cpu()
                })
                count += 1
                if count % 100 == 0:
                    print(count)

                

torch.save(reference_library, "reference_library.pt")