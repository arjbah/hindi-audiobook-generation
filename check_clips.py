import os

clips = [f for f in os.listdir('StoricoTTSDataset/clips') if f.endswith('.wav')]
print(f'Audio clips: {len(clips)}')

# Check if they match test.csv entries
import csv
with open('StoricoTTSDataset/test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_files = set()
    for row in reader:
        filepath = row.get('audio_filepath', '')
        if filepath:
            filename = os.path.basename(filepath)
            csv_files.add(filename)

print(f'Entries in test.csv: {len(csv_files)}')

missing_in_clips = csv_files - set(clips)
missing_in_csv = set(clips) - csv_files

print(f'Missing in clips folder: {len(missing_in_clips)}')
print(f'Missing in CSV: {len(missing_in_csv)}')
