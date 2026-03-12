"""
IndicParler-TTS Inference Script
For long-form expressive Hindi TTS evaluation

Based on: https://huggingface.co/ai4bharat/indic-parler-tts
"""

import torch
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration


def load_model(model_name="ai4bharat/indic-parler-tts", device=None):
    """
    Load IndicParler-TTS model and tokenizers.
    
    Returns:
        model: ParlerTTS model
        tokenizer: Prompt tokenizer (for transcript)
        description_tokenizer: Description tokenizer (for style captions)
        device: Device used for inference
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    description_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path
    )
    
    print("Model loaded successfully!")
    print(f"Sampling rate: {model.config.sampling_rate} Hz")
    return model, tokenizer, description_tokenizer, device


def generate_speech(
    model,
    tokenizer,
    description_tokenizer,
    device,
    text,
    style_caption,
    output_path="output.wav",
):
    """
    Generate speech from text using IndicParler-TTS.
    
    Args:
        model: Loaded ParlerTTS model
        tokenizer: Prompt tokenizer (for transcript)
        description_tokenizer: Description tokenizer (for style captions)
        device: Device for inference
        text: Hindi text to synthesize
        style_caption: Style description (e.g., "A female speaker delivers expressive speech...")
        output_path: Path to save output audio
    
    Returns:
        output_path: Path to generated audio file
    """
    # Tokenize both inputs
    description_input_ids = description_tokenizer(
        style_caption, return_tensors="pt"
    ).to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)
    
    print(f"Text: '{text[:60]}...'")
    print(f"Style: '{style_caption[:80]}...'")
    
    # Generate
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask,
        )
    
    # Convert to audio array
    audio_arr = generation.cpu().numpy().squeeze()
    
    # Save to file
    sf.write(output_path, audio_arr, model.config.sampling_rate)
    print(f"Audio saved to: {output_path}\n")
    
    return output_path


def main():
    """Test inference with sample Hindi texts and style variations."""
    # Load model
    model, tokenizer, description_tokenizer, device = load_model()
    
    # Sample Hindi text for testing (short narrative passage)
    test_text = "राज बहुत खुश था। आज उसका जन्मदिन था और उसके सभी दोस्त उसके घर पर इकट्ठा हुए थे।"
    
    # Style captions for expression evaluation
    # Format inspired by Parler-TTS style descriptors
    style_tests = [
        (
            "neutral narration",
            "A narrator delivers speech in a neutral, clear tone with moderate pace. High quality recording.",
            "output_neutral.wav",
        ),
        (
            "excited, fast pace",
            "A narrator delivers speech with excitement and animated expression, fast pace, high energy. High quality recording.",
            "output_excited.wav",
        ),
        (
            "calm, slow pace",
            "A narrator delivers speech in a calm, relaxed manner with slow pace and gentle tone. High quality recording.",
            "output_calm.wav",
        ),
    ]
    
    print("\n" + "="*60)
    print("IndicParler-TTS Inference Tests - Long-Form Expression Eval")
    print("="*60 + "\n")
    
    for style_name, style_caption, output_path in style_tests:
        print(f"[Test {style_name}]")
        generate_speech(
            model=model,
            tokenizer=tokenizer,
            description_tokenizer=description_tokenizer,
            device=device,
            text=test_text,
            style_caption=style_caption,
            output_path=output_path,
        )
    
    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
