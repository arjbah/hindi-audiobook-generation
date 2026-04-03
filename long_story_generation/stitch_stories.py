from pathlib import Path
import soundfile as sf
import numpy as np

# Get folder -> Glob all files -> read all files -> concatenate -> write
folder = Path("../data/hindi_story_output_voxtral")
def extract_num(path):
    return int(path.stem.split("_")[2])
files = sorted(folder.glob("*.wav"), key=extract_num)

audio_data = []
sample_rate = None

for file_path in files:
    data, sr = sf.read(file_path)
    if sample_rate is None:
        sample_rate = sr
    elif sr != sample_rate:
        raise ValueError(f"Sample rate mismatch in {file_path.name}")
    audio_data.append(data)

final_audio = np.concatenate(audio_data, axis=0)
sf.write(folder / "full_story.wav", final_audio, sample_rate)