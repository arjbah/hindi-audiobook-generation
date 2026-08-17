import numpy as np
import soundfile as sf
import torch
from indicnlp.tokenize import sentence_tokenize
from pydub import AudioSegment
from transformers import AutoModel
from pathlib import Path


INFERENCE_DIR = Path(__file__).resolve().parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


def inference(transcript: str, output_folder: Path, lang_code: str, tts_model, reference):
    sentences = sentence_tokenize.sentence_split(transcript, lang=lang_code)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    if not sentences:
        return

    for sentence_idx, sentence in enumerate(sentences):
        audio = tts_model(
            sentence,
            ref_audio_path=reference["audio_path"],
            ref_text=reference["transcript"],
        )
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(output_folder / f"{sentence_idx}.wav", np.asarray(audio, dtype=np.float32), 24000)

    combined = AudioSegment.empty()
    for sentence_idx in range(len(sentences)):
        combined += AudioSegment.from_wav(output_folder / f"{sentence_idx}.wav")
    combined.export(output_folder / f"{output_folder.name}.wav", format="wav")


tts_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(device)

for folder in [path for path in (INFERENCE_DIR / "text").iterdir() if path.is_dir()]:
    reference_library = torch.load(
        INFERENCE_DIR / f"rasa_male_{folder.name}_reference_library.pt",
        weights_only=False,
    )
    reference = reference_library[0]
    reference["audio_path"] = (
        INFERENCE_DIR / "reference_audio" / folder.name / Path(reference["audio_path"]).name
    )

    for file in folder.rglob("*.txt"):
        print(f"Starting {file}")
        text = file.read_text(encoding="utf-8")
        output_folder = INFERENCE_DIR / "audio" / folder.name / "indicf5_constant" / file.stem
        output_folder.mkdir(parents=True, exist_ok=True)
        inference(text, output_folder, LANG_CODES[folder.name], tts_model, reference)
        print(f"Finished {file}")
