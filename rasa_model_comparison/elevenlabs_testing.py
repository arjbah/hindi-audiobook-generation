from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api_key)

# Voice: George - Warm, Captivating Storyteller

# Get raw response with headers
audio = client.text_to_speech.convert(
    text="[happy] मम्मी ने सुंदर सा लहंगा चुनरी",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id = "eleven_v3"
)

with open("output.mp3", "wb") as f:
    for chunk in audio:
        if chunk:
            f.write(chunk)