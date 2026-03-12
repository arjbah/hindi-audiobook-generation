import csv

with open('StoricoTTSDataset/test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    total = 0
    narrator = 0
    character = 0
    character_names = {}
    
    for row in reader:
        total += 1
        char = row.get('character', '').strip()
        if char:
            character += 1
            character_names[char] = character_names.get(char, 0) + 1
        else:
            narrator += 1

print(f'Total segments: {total}')
print(f'Narrator segments: {narrator} ({narrator/total*100:.2f}%)')
print(f'Character segments: {character} ({character/total*100:.2f}%)')
print(f'\nCharacter distribution: {character_names}')
