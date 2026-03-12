import csv

with open('StoricoTTSDataset/test.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    stories = set()
    story_segments = {}
    
    for row in reader:
        story = row.get('story', '').strip()
        stories.add(story)
        story_segments[story] = story_segments.get(story, 0) + 1

print(f'Total unique stories: {len(stories)}')
print(f'\nSegments per story:')
for story in sorted(stories):
    print(f'  Story {story}: {story_segments[story]} segments')
