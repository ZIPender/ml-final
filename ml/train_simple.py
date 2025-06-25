import pickle
import json
import os
import sys
import random
from typing import List, Dict

def load_chapters_data(file_path: str, max_samples: int = 1000):
    """Load a sample of chapters data for training"""
    print(f"Loading chapters data from {file_path}...")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} video entries")
    
    # Convert to training format
    training_samples = []
    count = 0
    
    for video_id, video_data in data.items():
        if count >= max_samples:
            break
            
        if 'chapters' not in video_data or not video_data['chapters']:
            continue
            
        chapters = video_data['chapters']
        if len(chapters) < 2:
            continue
            
        # Create context
        context = ""
        if 'title' in video_data:
            context += f"Video: {video_data['title']}. "
        
        # Create training samples
        for i, chapter in enumerate(chapters):
            if i == 0:  # Skip first chapter
                continue
                
            prev_chapter = chapters[i-1]
            
            # Input: context + previous chapter info
            input_text = f"Context: {context}Previous: {prev_chapter['label']} at {prev_chapter['time']}s. "
            input_text += f"Generate a short YouTube chapter title for the next section at {chapter['time']}s."
            
            # Output: chapter title
            target_text = chapter['label']
            
            if len(target_text.strip()) > 0 and len(target_text) < 50:
                training_samples.append({
                    'input': input_text,
                    'output': target_text,
                    'timestamp': chapter['time']
                })
                count += 1
    
    print(f"Created {len(training_samples)} training samples")
    return training_samples

def save_training_data(samples: List[Dict], output_file: str):
    """Save training data to JSON file"""
    print(f"Saving {len(samples)} samples to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print("Training data saved successfully!")

def analyze_chapter_titles(samples: List[Dict]):
    """Analyze the chapter titles to understand patterns"""
    print("\n=== Chapter Title Analysis ===")
    
    titles = [sample['output'] for sample in samples]
    
    # Length analysis
    lengths = [len(title) for title in titles]
    avg_length = sum(lengths) / len(lengths)
    print(f"Average title length: {avg_length:.1f} characters")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    
    # Word count analysis
    word_counts = [len(title.split()) for title in titles]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"Average word count: {avg_words:.1f} words")
    
    # Show some examples
    print(f"\nSample titles:")
    for i, title in enumerate(random.sample(titles, 10)):
        print(f"  {i+1}. {title}")
    
    # Common patterns
    print(f"\nCommon patterns:")
    common_words = {}
    for title in titles:
        words = title.lower().split()
        for word in words:
            if len(word) > 2:  # Skip short words
                common_words[word] = common_words.get(word, 0) + 1
    
    # Show top 10 common words
    sorted_words = sorted(common_words.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 common words:")
    for word, count in sorted_words[:10]:
        print(f"  '{word}': {count} times")

def main():
    # Configuration
    chapters_file = "../data/chapters.pkl"
    output_file = "training_data.json"
    max_samples = 2000  # Start with reasonable amount
    
    if not os.path.exists(chapters_file):
        print(f"Chapters file not found: {chapters_file}")
        return
    
    # Load and prepare data
    samples = load_chapters_data(chapters_file, max_samples)
    
    if not samples:
        print("No training samples created!")
        return
    
    # Analyze the data
    analyze_chapter_titles(samples)
    
    # Save training data
    save_training_data(samples, output_file)
    
    print(f"\n=== Summary ===")
    print(f"Training data prepared: {len(samples)} samples")
    print(f"Data saved to: {output_file}")
    print(f"Next step: Use this data to fine-tune a model")

if __name__ == "__main__":
    main() 