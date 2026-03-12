import csv

with open('StoricoTTSDataset/test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total = 0
    with_emotion = 0
    emotions = {}
    for row in reader:
        total += 1
        emotion = row.get('emotion', '').strip()
        if emotion:
            with_emotion += 1
            emotions[emotion] = emotions.get(emotion, 0) + 1

print(f'Total segments: {total}')
print(f'With emotion: {with_emotion}')
print(f'Without emotion: {total - with_emotion}')
print(f'Percentage with emotion: {with_emotion/total*100:.2f}%')
print(f'Emotion distribution: {emotions}')
