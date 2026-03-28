import soundfile as sf
import pyarrow.parquet as pq
import io

df = pq.read_table("train-00006-of-00010.parquet")

target_seconds = 1800
current_seconds = 0
file_idx = 0

for batch in df.to_batches(max_chunksize=32):
    batch = batch.to_pandas()

    for _, row in batch.iterrows(): # Note that in the parquet all values are male
        audio_bytes = row["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(audio) / sr
        if duration < 5 or duration > 15:
            continue
        sf.write(f"audio_files/{file_idx}.wav", audio, sr)
        current_seconds += duration
        file_idx += 1

        if current_seconds >= target_seconds:
            print("Achieved target")
            break
    
    print(f"Current seconds: {current_seconds}")
    print(f"File index: {file_idx}")