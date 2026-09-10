from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent
languages = ["assamese", "bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi", "tamil", "telugu"]
models = ["elevenlabs", "indicf5", "indicf5_constant", "indicparler"]

for language in languages:
    language_path = ROOT / "audio" / language

    for model in models:
        model_path = language_path / model

        if model == "elevenlabs":
            files = model_path.glob("*.mp3")
        else:
            files = (
                folder / f"{folder.name}.wav"
                for folder in model_path.glob("*")
                if folder.is_dir()
            )

        for file_path in files:
            if not file_path.is_file():
                continue

            dest_path = ROOT / "mos_audio" / language / model / f"{file_path.stem}.wav"
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            if file_path.suffix == ".mp3":
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(file_path),
                    str(dest_path)
                ], check=True)
            else:
                shutil.copy2(file_path, dest_path)
