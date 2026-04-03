from datasets import Dataset, Audio, Features, Value
import pandas as pd
import ast

df = pd.read_csv("../data/rasa_clips/rasa_model_comparison.csv")

dataset = Dataset.from_pandas(df)
dataset = dataset.map(
    lambda x: {"rasa_audio": ast.literal_eval(x["rasa_audio"])}
)
dataset = dataset.map(
    lambda x: {"indicparler_audio": ast.literal_eval(x["indicparler_audio"])}
)
dataset = dataset.map(
    lambda x: {"elevenlabs_audio": ast.literal_eval(x["elevenlabs_audio"])}
)
dataset = dataset.map(
    lambda x: {"voxtral_audio": ast.literal_eval(x["voxtral_audio"])}
)
dataset = dataset.cast_column("rasa_audio", Audio())
dataset = dataset.cast_column("indicparler_audio", Audio())
dataset = dataset.cast_column("elevenlabs_audio", Audio())
dataset = dataset.cast_column("voxtral_audio", Audio())

features = Features({
    "transcript": Value("string"),
    "gender": Value("string"),
    "emotion": Value("string"),
    "description": Value("string"),
    "rasa_audio": Audio(),
    "indicparler_audio": Audio(),
    "elevenlabs_description": Value("string"),
    "elevenlabs_audio": Audio(),
    "voxtral_audio": Audio()
})
dataset = dataset.cast(features)

dataset.push_to_hub("williamxing1/rasa_model_comparison")


# transcript,gender,emotion,description,rasa_audio,indicparler_audio,elevenlabs_description,elevenlabs_audio,voxtral_audio