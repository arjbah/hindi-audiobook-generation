"""Check the correct tokenizer usage for IndicParler-TTS"""

from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

MODEL_NAME = "ai4bharat/indic-parler-tts"

print("Loading model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

# Check what tokenizers the model expects
print(f"\nModel config text_encoder: {model.config.text_encoder._name_or_path}")
print(f"Model decoder vocab size: {model.config.decoder.vocab_size}")
print(f"Model text encoder vocab size: {model.config.text_encoder.vocab_size}")

# Try loading the tokenizer from the model card
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"\nMain tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Main tokenizer model max ID: {max(tokenizer.get_vocab().values())}")

# Test with both tokenizers being the same
hindi_caption = "इस भाषण की शैली उत्साहपूर्ण और ऊर्जावान है"

# Use the same tokenizer for both
desc_ids = tokenizer(hindi_caption, return_tensors="pt")
print(f"\nUsing main tokenizer for description:")
print(f"  IDs: {desc_ids.input_ids}")
print(f"  Max ID: {desc_ids.input_ids.max()}")
print(f"  Shape: {desc_ids.input_ids.shape}")
