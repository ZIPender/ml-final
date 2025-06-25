import pickle
import json
import random
from pathlib import Path

# Paths
DATA_DIR = Path("../data")
CHAPTERS_PATH = DATA_DIR / "chapters.pkl"
ASR_PATH = DATA_DIR / "asr.pkl"
TRAIN_PATH = Path("train.jsonl")
VAL_PATH = Path("val.jsonl")

# Load data
with open(CHAPTERS_PATH, 'rb') as f:
    chapters = pickle.load(f)
with open(ASR_PATH, 'rb') as f:
    asr = pickle.load(f)

examples = []

for video_id in chapters:
    video_chapters = chapters[video_id]
    video_asr = asr.get(video_id)
    if not video_asr:
        continue
    # video_asr: list of dicts with 'start', 'duration', 'text'
    for chapter in video_chapters:
        # Each chapter: dict with 'start', 'end', 'title'
        start = chapter.get('start')
        end = chapter.get('end')
        title = chapter.get('title')
        if start is None or end is None or not title:
            continue
        # Get transcript segment for this chapter
        segs = [seg['text'] for seg in video_asr if seg['start'] >= start and seg['start'] < end]
        transcript_chunk = ' '.join(segs).strip()
        if len(transcript_chunk.split()) < 10 or len(title.split()) < 2:
            continue
        examples.append({
            'input_text': transcript_chunk,
            'target_text': title.strip()
        })

# Shuffle and split
random.shuffle(examples)
split_idx = int(0.9 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# Write to JSONL
with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
    for ex in train_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')
with open(VAL_PATH, 'w', encoding='utf-8') as f:
    for ex in val_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + '\n')

print(f"Wrote {len(train_examples)} training and {len(val_examples)} validation examples.") 