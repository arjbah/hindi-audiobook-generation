from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

text_folder = "../mos_eval/story_transcripts"

for file_name in os.listdir(text_folder):
    if not file_name.endswith(".txt") or file_name in ["story7.txt", "story8.txt", "story4.txt", "story6.txt", "story3.txt"]:
        continue

    print(f"Starting {file_name}")
    with open(os.path.join(text_folder, file_name), "r", encoding="utf-8") as f:
        text = f.read()

    output_path = f"elevenlabs/{os.path.splitext(file_name)[0]}.mp3"

    audio = client.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_v3"
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    print(f"Finished {file_name}")