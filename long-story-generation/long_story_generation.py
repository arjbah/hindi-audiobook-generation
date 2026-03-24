import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from dotenv import load_dotenv
import os
load_dotenv()
from indicnlp.tokenize import sentence_tokenize

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
folder_name = "../data/long_story_output"
os.makedirs(folder_name, exist_ok=True)
max_tokens = 512

with open("../data/long_story_transcript.txt", "r") as f:
    transcript = f.read()
transcript = transcript.replace("\n", " ")

# IndicParler Model + Tokenizers Generation
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Splitting up the transcript
full_tokenized = tokenizer(transcript, return_tensors="pt").to(device)
full_length = full_tokenized.input_ids.shape[1]
print(f"Transcript token length: {full_length}")
sentences = sentence_tokenize.sentence_split(transcript, lang='hi')
chunks = []
current_chunk = ""
current_len = 0

for sentence in sentences:
    tokens = tokenizer(sentence, return_tensors="pt").to(device)
    sentence_length = tokens.input_ids.shape[1]
    if current_len + sentence_length > max_tokens:
        chunks.append(current_chunk)
        current_chunk = sentence
        current_len = sentence_length
    else:
        current_chunk += " " + sentence
        current_len += sentence_length
    
if current_chunk:
    chunks.append(current_chunk.strip())

print(f"# of chunks: {len(chunks)}")

for index, chunk in enumerate(chunks):
    description_input_ids = description_tokenizer("A clear, natural-speaking narrator with a smooth tone. Include subtle emotional variation suitable for storytelling.", return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(chunk, return_tensors="pt").to(device)

    generation = model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
    audio_arr = generation.cpu().numpy().squeeze()
    indicparler_path = f"{folder_name}/indicparler_generation_{index}_{max_tokens}.wav"
    sf.write(indicparler_path, audio_arr, model.config.sampling_rate)
    print(f"Finished IndicParler {index}")

"""# ElevenLabs - Long-form generation must be done on website so this isn't used
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=elevenlabs_api_key)

audio = client.text_to_speech.convert(
    text=transcript,
    voice_id="JBFqnCBsd6RMkjVDRZzb", # Voice: George - Warm, Captivating Storyteller
    model_id = "eleven_v3"
)
elevenlabs_path = f"{folder_name}/elevenlabs_generation.wav"
with open(elevenlabs_path, "wb") as f:
    for chunk in audio:
        if chunk:
            f.write(chunk)

print(f"Finished ElevenLabs")"""