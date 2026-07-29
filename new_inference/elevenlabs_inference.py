from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

root = Path("text")
folders = [p for p in root.iterdir() if p.is_dir()]

for folder in folders:
    files = list(folder.rglob("*.txt"))

    for file in files:
        print(f"Starting {file}")
        with file.open("r", encoding="utf-8") as f:
            text = f.read()

        output_path = Path("audio") / folder.name / "elevenlabs" / f"{file.stem}.mp3"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        audio = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_v3"
        )

        with output_path.open("wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        print(f"Finished {file}")