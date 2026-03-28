with open("../data/long_story_transcript.txt", "r") as f:
    lines = f.read()

lines = lines.replace("\n", " ")
with open("../data/long_story_transcript_clean.txt", "w") as f:
    f.write(lines)