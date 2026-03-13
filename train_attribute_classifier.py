"""
Train Attribute Classifier for L_attr Evaluation

Trains a Wav2Vec2-based classifier to predict speech attributes:
- Emotion (7 categories)
- Rate (3 categories: slow/normal/fast)
- Pitch (3 categories: low/medium/high)
- Energy (3 categories: low/medium/high)

Uses GPT-4o generated style captions as weak supervision labels.

Usage:
    python train_attribute_classifier.py

Requirements:
    - style_captions.json (from generate_captions.py)
    - StoricoTTSDataset/clips/ (audio files)
    - GPU recommended
"""

import os
import json
import csv
import re
import torch
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Paths
    DATA_DIR = "StoricoTTSDataset"
    CLIPS_DIR = os.path.join(DATA_DIR, "clips")
    CAPTIONS_FILE = os.path.join(DATA_DIR, "style_captions.json")
    ATTRIBUTES_LABELED_FILE = os.path.join(DATA_DIR, "attributes_labeled.json")  # NEW: LLM-labeled attributes
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    OUTPUT_DIR = "attribute_classifier"
    
    # Model
    BASE_MODEL = "facebook/wav2vec2-large-xlsr-53"  # Multilingual, good for Hindi
    MAX_AUDIO_LENGTH = 10.0  # seconds
    SAMPLING_RATE = 16000
    
    # Training
    NUM_EPOCHS = 5
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    
    # Attributes to predict
    EMOTION_LABELS = [
        "neutral",
        "excited",
        "calm",
        "sad",
        "angry",
        "frustrated",
        "happy",
        "surprised",
        "fearful",
        "disgusted",
    ]
    
    RATE_LABELS = ["slow", "normal", "fast"]
    PITCH_LABELS = ["low", "medium", "high"]
    ENERGY_LABELS = ["low", "medium", "high"]
    
    # Use LLM-labeled attributes if available, else fall back to keyword mapping
    USE_LLM_LABELS = True
    
    # Train/val split
    VAL_SPLIT = 0.15
    SEED = 42


config = Config()

# ============================================================================
# Caption → Attribute Mapping
# ============================================================================

# Keyword-based mapping from GPT-4o captions to discrete labels
EMOTION_KEYWORDS = {
    "excited": ["excited", "enthusiastic", "energetic", "animated", "lively"],
    "calm": ["calm", "calmly", "relaxed", "gentle", "soothing", "peaceful"],
    "sad": ["sad", "sadly", "sorrow", "melancholy", "somber", "mournful"],
    "angry": ["angry", "anger", "furious", "rage", "hostile"],
    "frustrated": ["frustrated", "frustration", "irritated", "annoyed", "impatient"],
    "happy": ["happy", "happily", "cheerful", "joyful", "bright", "warm", "smile"],
    "neutral": ["neutral", "narrator", "matter-of-fact", "steady", "even"],
}

RATE_KEYWORDS = {
    "slow": ["slow", "slowly", "leisurely", "drawn-out", "measured"],
    "fast": ["fast", "quick", "rapid", "speed", "haste", "hurried"],
    "normal": ["normal", "moderate", "steady", "natural", "conversational"],
}

PITCH_KEYWORDS = {
    "low": ["low", "deep", "bass", "grave"],
    "high": ["high", "raised", "elevated", "shrill"],
    "medium": ["medium", "mid", "natural", "normal"],
}

ENERGY_KEYWORDS = {
    "low": ["low", "quiet", "soft", "gentle", "subdued", "whisper"],
    "high": ["high", "loud", "energetic", "intense", "powerful", "strong"],
    "medium": ["medium", "moderate", "balanced", "normal"],
}


def map_caption_to_attributes(caption: str) -> Dict[str, str]:
    """
    Map a style caption to discrete attribute labels.
    
    Args:
        caption: Style caption string (e.g., "spoken excitedly with high energy")
    
    Returns:
        Dict with attribute labels
    """
    caption_lower = caption.lower()
    
    attributes = {
        'emotion': 'neutral',  # default
        'rate': 'normal',      # default
        'pitch': 'medium',     # default
        'energy': 'medium',    # default
    }
    
    # Emotion
    max_emotion_score = 0
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in caption_lower)
        if score > max_emotion_score:
            max_emotion_score = score
            attributes['emotion'] = emotion
    
    # Rate
    max_rate_score = 0
    for rate, keywords in RATE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in caption_lower)
        if score > max_rate_score:
            max_rate_score = score
            attributes['rate'] = rate
    
    # Pitch
    max_pitch_score = 0
    for pitch, keywords in PITCH_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in caption_lower)
        if score > max_pitch_score:
            max_pitch_score = score
            attributes['pitch'] = pitch
    
    # Energy
    max_energy_score = 0
    for energy, keywords in ENERGY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in caption_lower)
        if score > max_energy_score:
            max_energy_score = score
            attributes['energy'] = energy
    
    return attributes


# ============================================================================
# Dataset
# ============================================================================


class SpeechAttributeDataset(Dataset):
    """Dataset for speech attribute classification."""
    
    def __init__(
        self,
        data_list: List[Dict],
        feature_extractor: Wav2Vec2FeatureExtractor,
        max_length: float = 10.0,
        sampling_rate: int = 16000,
    ):
        self.data_list = data_list
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.sampling_rate = sampling_rate
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data_list[idx]
        audio_path = item['audio_path']
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        except Exception as e:
            # Return dummy data on error (will be filtered)
            audio = np.zeros(int(self.sampling_rate * self.max_length))
        
        # Truncate or pad
        max_samples = int(self.sampling_rate * self.max_length)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Extract features
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="np",
        )
        
        return {
            'input_values': features.input_values[0],
            'emotion_label': item['emotion_label'],
            'rate_label': item['rate_label'],
            'pitch_label': item['pitch_label'],
            'energy_label': item['energy_label'],
            'segment_id': item['segment_id'],
        }


def load_and_prepare_data():
    """Load data and create training samples."""

    print("Loading data...")

    # Try to load LLM-labeled attributes first
    llm_attributes = None
    if config.USE_LLM_LABELS and os.path.exists(config.ATTRIBUTES_LABELED_FILE):
        print(f"Loading LLM-labeled attributes from: {config.ATTRIBUTES_LABELED_FILE}")
        with open(config.ATTRIBUTES_LABELED_FILE, 'r', encoding='utf-8') as f:
            llm_attributes = json.load(f)
        print(f"Loaded {len(llm_attributes)} LLM-labeled samples")
    elif config.USE_LLM_LABELS:
        print(f"WARNING: LLM labels not found at {config.ATTRIBUTES_LABELED_FILE}")
        print("Run label_attributes_llm.py first, or set USE_LLM_LABELS = False")
        print("Falling back to keyword-based mapping...")

    # Load captions
    with open(config.CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        style_captions = json.load(f)

    # Load test.csv for metadata
    segments = []
    with open(config.TEST_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                'segment_id': os.path.basename(row['audio_filepath']).replace('.wav', ''),
                'audio_filepath': row['audio_filepath'],
                'character': row.get('character', '').strip(),
                'emotion': row.get('emotion', '').strip(),
                'text': row.get('text', ''),
                'story': row.get('story', ''),
            })
    
    # Create training samples
    data_list = []
    skipped_no_audio = 0
    skipped_no_caption = 0

    print("Processing segments...")
    for seg in tqdm(segments, desc="Loading audio", unit="seg"):
        segment_id = seg['segment_id']
        audio_path = os.path.join(config.CLIPS_DIR, f"{segment_id}.wav")

        # Check audio exists
        if not os.path.exists(audio_path):
            skipped_no_audio += 1
            continue

        # Get caption
        caption_data = style_captions.get(segment_id, {})
        caption = caption_data.get('style_caption', '')

        # Skip if no caption or error
        if not caption or caption.startswith('[ERROR]'):
            skipped_no_caption += 1
            continue

        # Use LLM-labeled attributes if available
        attributes = None
        if llm_attributes and segment_id in llm_attributes:
            llm_data = llm_attributes[segment_id]
            if 'attributes' in llm_data:
                attributes = llm_data['attributes']
        
        # Fall back to keyword mapping
        if attributes is None:
            # Clean caption
            if caption.startswith('[EXISTING]'):
                caption = caption.replace('[EXISTING]', '').strip()
            attributes = map_caption_to_attributes(caption)

        # Skip if any attribute label is not in our label set
        try:
            emotion_idx = config.EMOTION_LABELS.index(attributes['emotion'])
            rate_idx = config.RATE_LABELS.index(attributes['rate'])
            pitch_idx = config.PITCH_LABELS.index(attributes['pitch'])
            energy_idx = config.ENERGY_LABELS.index(attributes['energy'])
        except ValueError as e:
            print(f"Warning: Unknown label for {segment_id}: {e}")
            skipped_no_caption += 1
            continue

        data_list.append({
            'segment_id': segment_id,
            'audio_path': audio_path,
            'emotion_label': emotion_idx,
            'rate_label': rate_idx,
            'pitch_label': pitch_idx,
            'energy_label': energy_idx,
            'attributes': attributes,  # Keep for reference
        })
    
    print(f"Total samples: {len(data_list)}")
    print(f"Skipped (no audio): {skipped_no_audio}")
    print(f"Skipped (no caption): {skipped_no_caption}")
    
    # Show label distribution
    print("\nLabel distribution:")
    print(f"  Emotion: {[(config.EMOTION_LABELS[i], sum(1 for d in data_list if d['emotion_label']==i)) for i in range(len(config.EMOTION_LABELS))]}")
    print(f"  Rate: {[(config.RATE_LABELS[i], sum(1 for d in data_list if d['rate_label']==i)) for i in range(len(config.RATE_LABELS))]}")
    
    return data_list


# ============================================================================
# Model Training
# ============================================================================


class AttributeClassificationTrainer:
    """Trains separate classifiers for each attribute."""
    
    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.BASE_MODEL
        )
        self.models = {}
        self.trainers = {}
    
    def create_model(self, num_labels: int) -> Wav2Vec2ForSequenceClassification:
        """Create a classification model."""
        config_model = AutoConfig.from_pretrained(
            config.BASE_MODEL,
            num_labels=num_labels,
            label2id={i: i for i in range(num_labels)},
            id2label={i: str(i) for i in range(num_labels)},
            finetuning_task="audio_classification",
        )
        
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            config.BASE_MODEL,
            config=config_model,
            ignore_mismatched_sizes=True,
        )
        
        # Freeze feature extractor for first epoch (optional stability trick)
        model.freeze_feature_encoder()
        
        return model
    
    def train_attribute(
        self,
        attribute_name: str,
        train_data: List[Dict],
        val_data: List[Dict],
        num_labels: int,
    ):
        """Train a classifier for one attribute."""
        
        print(f"\n{'='*60}")
        print(f"Training {attribute_name} classifier")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = SpeechAttributeDataset(
            train_data,
            self.feature_extractor,
            config.MAX_AUDIO_LENGTH,
            config.SAMPLING_RATE,
        )
        val_dataset = SpeechAttributeDataset(
            val_data,
            self.feature_extractor,
            config.MAX_AUDIO_LENGTH,
            config.SAMPLING_RATE,
        )
        
        # Create model
        model = self.create_model(num_labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(config.OUTPUT_DIR, attribute_name),
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE * 2,
            num_train_epochs=config.NUM_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            warmup_steps=config.WARMUP_STEPS,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir=os.path.join(config.OUTPUT_DIR, "logs", attribute_name),
            logging_steps=50,
            save_total_limit=2,
            seed=config.SEED,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
        )
        
        # Compute metrics
        def compute_metrics(eval_pred):
            from sklearn.metrics import accuracy_score, f1_score
            
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            
            # Handle case where labels might be nested
            if labels.ndim > 1:
                labels = labels[:, 0]
            
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
            }
        
        # Custom data collator for our dataset
        def data_collator(features):
            input_values = [f['input_values'] for f in features]
            labels = {
                f'{attribute_name}_label': torch.tensor([f[f'{attribute_name}_label'] for f in features])
            }
            
            # Pad input values
            max_length = max(len(x) for x in input_values)
            input_values = [
                np.pad(x, (0, max_length - len(x)), mode='constant')
                for x in input_values
            ]
            input_values = torch.FloatTensor(np.stack(input_values))
            
            return {
                'input_values': input_values,
                'labels': labels[f'{attribute_name}_label'],
            }
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save model
        save_path = os.path.join(config.OUTPUT_DIR, f"{attribute_name}_final")
        trainer.save_model(save_path)
        self.feature_extractor.save_pretrained(save_path)
        
        print(f"Model saved to: {save_path}")
        
        self.models[attribute_name] = model
        self.trainers[attribute_name] = trainer
        
        return trainer
    
    def train_all(self, data_list: List[Dict]):
        """Train classifiers for all attributes."""
        
        # Train/val split
        np.random.seed(config.SEED)
        indices = np.random.permutation(len(data_list))
        val_size = int(len(data_list) * config.VAL_SPLIT)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        
        print(f"\nTrain/val split: {len(train_data)} / {len(val_data)}")
        
        # Train each attribute
        attributes_to_train = [
            ('emotion', len(config.EMOTION_LABELS)),
            ('rate', len(config.RATE_LABELS)),
            ('pitch', len(config.PITCH_LABELS)),
            ('energy', len(config.ENERGY_LABELS)),
        ]
        
        results = {}
        for attr_name, num_labels in attributes_to_train:
            trainer = self.train_attribute(attr_name, train_data, val_data, num_labels)
            eval_result = trainer.evaluate()
            results[attr_name] = eval_result
        
        # Save all results
        results_path = os.path.join(config.OUTPUT_DIR, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {results_path}")
        
        return results


# ============================================================================
# Inference / Evaluation
# ============================================================================


class AttributeClassifier:
    """Load and use trained attribute classifiers."""
    
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            model_dir = config.OUTPUT_DIR
        
        self.model_dir = model_dir
        self.models = {}
        self.feature_extractor = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained attribute classifiers."""
        
        attributes = ['emotion', 'rate', 'pitch', 'energy']
        
        for attr in attributes:
            model_path = os.path.join(self.model_dir, f"{attr}_final")
            
            if not os.path.exists(model_path):
                print(f"Warning: {attr} model not found at {model_path}")
                continue
            
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()
            
            self.models[attr] = model
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            os.path.join(self.model_dir, "emotion_final")  # Any one works
        )
        
        print(f"Loaded {len(self.models)} attribute classifiers")
    
    def predict(self, audio_path: str) -> Dict[str, str]:
        """
        Predict attributes for an audio file.
        
        Returns:
            Dict with predicted attribute labels
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=config.SAMPLING_RATE)
        
        # Truncate
        max_samples = int(config.SAMPLING_RATE * config.MAX_AUDIO_LENGTH)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Extract features
        features = self.feature_extractor(
            audio,
            sampling_rate=config.SAMPLING_RATE,
            return_tensors="pt",
        )
        
        if torch.cuda.is_available():
            features = {k: v.cuda() for k, v in features.items()}
        
        predictions = {}
        
        with torch.no_grad():
            for attr_name, model in self.models.items():
                outputs = model(**features)
                logits = outputs.logits
                predicted_id = torch.argmax(logits, dim=-1).item()
                
                # Map ID to label
                label_map = {
                    'emotion': config.EMOTION_LABELS,
                    'rate': config.RATE_LABELS,
                    'pitch': config.PITCH_LABELS,
                    'energy': config.ENERGY_LABELS,
                }
                
                predictions[attr_name] = label_map[attr_name][predicted_id]
        
        return predictions


# ============================================================================
# Main
# ============================================================================


def main():
    print("="*60)
    print("Attribute Classifier Training for L_attr Evaluation")
    print("="*60)
    print(f"Base model: {config.BASE_MODEL}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    data_list = load_and_prepare_data()
    
    if not data_list:
        print("No data to train on. Exiting.")
        return
    
    # Save data mapping for reference
    mapping_path = os.path.join(config.OUTPUT_DIR, "data_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False)
    print(f"\nData mapping saved to: {mapping_path}")
    
    # Train classifiers
    trainer = AttributeClassificationTrainer()
    results = trainer.train_all(data_list)
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for attr, metrics in results.items():
        print(f"\n{attr.upper()}:")
        print(f"  Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
        print(f"  F1 (macro): {metrics.get('eval_f1_macro', 'N/A'):.4f}")
        print(f"  F1 (weighted): {metrics.get('eval_f1_weighted', 'N/A'):.4f}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Use AttributeClassifier class for inference.")
    print("="*60)


if __name__ == "__main__":
    main()
