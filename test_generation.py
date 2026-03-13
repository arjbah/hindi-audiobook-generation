"""Test IndicParler-TTS with Hindi text and captions"""

import torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

MODEL_NAME = "ai4bharat/indic-parler-tts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print(f"Model loaded on {DEVICE}")
print(f"Description tokenizer vocab size: {description_tokenizer.vocab_size}")

# Test with Hindi
hindi_text = "हेलो बच्चों, आज जो कहानी मैं सबके लिए लेकर आई हूँ"
hindi_caption = "इस भाषण की शैली उत्साहपूर्ण और ऊर्जावान है"

print(f"\nHindi text: {hindi_text}")
print(f"Hindi caption: {hindi_caption}")

# Tokenize
print("\nTokenizing...")
description_input_ids = description_tokenizer(
    hindi_caption, return_tensors="pt"
).to(DEVICE)
prompt_input_ids = tokenizer(
    hindi_text, return_tensors="pt"
).to(DEVICE)

print(f"Description input IDs shape: {description_input_ids.input_ids.shape}")
print(f"Description input IDs: {description_input_ids.input_ids}")
print(f"Description input IDs max: {description_input_ids.input_ids.max()}")
print(f"Description vocab size: {description_tokenizer.vocab_size}")

print(f"Prompt input IDs shape: {prompt_input_ids.input_ids.shape}")
print(f"Prompt input IDs: {prompt_input_ids.input_ids}")
print(f"Prompt input IDs max: {prompt_input_ids.input_ids.max()}")

# Check if IDs are within vocab
if description_input_ids.input_ids.max() >= description_tokenizer.vocab_size:
    print(f"\n⚠️ WARNING: Description token IDs exceed vocab size!")
    
if prompt_input_ids.input_ids.max() >= tokenizer.vocab_size:
    print(f"\n⚠️ WARNING: Prompt token IDs exceed vocab size!")

# Generate
print("\nGenerating...")
try:
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
        )
    print(f"Generation successful! Shape: {generation.shape}")
except Exception as e:
    print(f"Generation failed: {e}")
    import traceback
    traceback.print_exc()
