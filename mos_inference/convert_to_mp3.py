from pathlib import Path
import subprocess

input_root = Path("data/mos")
output_root = Path("data/mos_mp3")

for wav_path in input_root.rglob("*.wav"):
    relative_path = wav_path.relative_to(input_root)
    mp3_path = output_root / relative_path.with_suffix(".mp3")

    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-t", "30",
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(mp3_path)
    ], check=True)

    print(f"Converted: {wav_path} -> {mp3_path}")

for wav_path in input_root.rglob("*.mp3"):
    relative_path = wav_path.relative_to(input_root)
    mp3_path = output_root / relative_path.with_suffix(".mp3")

    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-t", "30",
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(mp3_path)
    ], check=True)

    print(f"Converted: {wav_path} -> {mp3_path}")