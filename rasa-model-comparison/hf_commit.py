from datasets import Dataset, Audio
import pandas as pd
import ast

df = pd.read_csv("../data/tts_model_out_v3/rasa_model_comparison.csv")

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
dataset = dataset.cast_column("rasa_audio", Audio())
dataset = dataset.cast_column("indicparler_audio", Audio())
dataset = dataset.cast_column("elevenlabs_audio", Audio())

dataset.push_to_hub("williamxing1/rasa_indicparler_comparison")