import pickle
import json
import os
from typing import Dict, List, Any, Iterator
import gc
import sys

def load_pickle_in_chunks(filepath: str, chunk_size: int = 1000) -> Iterator[Any]:
    """
    Load a large pickle file in chunks to avoid memory issues.
    """
    print(f"Loading {filepath} in chunks...")
    
    try:
        with open(filepath, 'rb') as f:
            # Try to load the pickle file
            data = pickle.load(f)
            
            if isinstance(data, dict):
                # If it's a dict, yield items in chunks
                items = list(data.items())
                for i in range(0, len(items), chunk_size):
                    chunk = dict(items[i:i + chunk_size])
                    yield chunk
                    gc.collect()  # Force garbage collection
            else:
                # If it's a list or other iterable
                for i in range(0, len(data), chunk_size):
                    chunk = data[i:i + chunk_size]
                    yield chunk
                    gc.collect()
                    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

def analyze_chapters_structure(filepath: str = "data/chapters.pkl") -> None:
    """
    Safely analyze the structure of chapters.pkl without loading everything into memory.
    """
    print("=== SAFE CHAPTERS ANALYSIS ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Get file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    
    try:
        # Load just a small sample first
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            
            # Show first few keys
            sample_keys = list(data.keys())[:5]
            print(f"Sample keys: {sample_keys}")
            
            # Analyze first item structure
            if data:
                first_key = list(data.keys())[0]
                first_item = data[first_key]
                print(f"First item type: {type(first_item)}")
                print(f"First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                
                # Save a small sample
                sample_data = {k: data[k] for k in list(data.keys())[:10]}
                with open("ml/sample_chapters.json", "w") as f:
                    json.dump(sample_data, f, indent=2, default=str)
                print("Saved sample to ml/sample_chapters.json")
                
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            if data:
                print(f"First item type: {type(data[0])}")
                
                # Save a small sample
                sample_data = data[:10]
                with open("ml/sample_chapters.json", "w") as f:
                    json.dump(sample_data, f, indent=2, default=str)
                print("Saved sample to ml/sample_chapters.json")
                
    except Exception as e:
        print(f"Error analyzing chapters: {e}")

def analyze_asr_structure(filepath: str = "data/asr.pkl") -> None:
    """
    Safely analyze the structure of asr.pkl without loading everything into memory.
    """
    print("\n=== SAFE ASR ANALYSIS ===")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Get file size
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")
    
    if file_size > 1000:  # If larger than 1GB
        print("⚠️  WARNING: This file is very large. Only analyzing structure, not content.")
        print("Consider using a smaller subset for analysis.")
        return
    
    try:
        # Load just a small sample first
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            
            # Show first few keys
            sample_keys = list(data.keys())[:5]
            print(f"Sample keys: {sample_keys}")
            
            # Analyze first item structure
            if data:
                first_key = list(data.keys())[0]
                first_item = data[first_key]
                print(f"First item type: {type(first_item)}")
                print(f"First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
                
                # Save a small sample
                sample_data = {k: data[k] for k in list(data.keys())[:5]}
                with open("ml/sample_asr.json", "w") as f:
                    json.dump(sample_data, f, indent=2, default=str)
                print("Saved sample to ml/sample_asr.json")
                
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
            if data:
                print(f"First item type: {type(data[0])}")
                
                # Save a small sample
                sample_data = data[:5]
                with open("ml/sample_asr.json", "w") as f:
                    json.dump(sample_data, f, indent=2, default=str)
                print("Saved sample to ml/sample_asr.json")
                
    except Exception as e:
        print(f"Error analyzing ASR: {e}")

def create_training_data_safe(output_file: str = "ml/training_data_safe.json", max_samples: int = 1000) -> None:
    """
    Create training data from a small subset of the VidChapters dataset.
    """
    print(f"\n=== CREATING SAFE TRAINING DATA (max {max_samples} samples) ===")
    
    chapters_file = "data/chapters.pkl"
    asr_file = "data/asr.pkl"
    
    if not os.path.exists(chapters_file):
        print(f"Chapters file not found: {chapters_file}")
        return
    
    training_data = []
    sample_count = 0
    
    try:
        # Load chapters data in small chunks
        for chunk in load_pickle_in_chunks(chapters_file, chunk_size=100):
            for video_id, chapter_data in chunk.items():
                if sample_count >= max_samples:
                    break
                    
                try:
                    # Extract chapter information
                    if isinstance(chapter_data, dict):
                        # Look for common chapter data structures
                        if 'chapters' in chapter_data:
                            chapters = chapter_data['chapters']
                        elif 'segments' in chapter_data:
                            chapters = chapter_data['segments']
                        else:
                            # Try to find any list that might contain chapters
                            chapters = None
                            for key, value in chapter_data.items():
                                if isinstance(value, list) and len(value) > 0:
                                    if isinstance(value[0], dict) and any(k in value[0] for k in ['title', 'start', 'end', 'text']):
                                        chapters = value
                                        break
                        
                        if chapters and isinstance(chapters, list):
                            # Process each chapter
                            for chapter in chapters:
                                if sample_count >= max_samples:
                                    break
                                    
                                if isinstance(chapter, dict):
                                    # Extract chapter information
                                    title = chapter.get('title', '')
                                    start_time = chapter.get('start', 0)
                                    end_time = chapter.get('end', 0)
                                    text = chapter.get('text', '')
                                    
                                    if title and text:
                                        training_data.append({
                                            'video_id': video_id,
                                            'title': title,
                                            'start_time': start_time,
                                            'end_time': end_time,
                                            'text': text
                                        })
                                        sample_count += 1
                                        
                                        if sample_count % 100 == 0:
                                            print(f"Processed {sample_count} samples...")
                                            gc.collect()  # Force garbage collection
                
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    continue
            
            if sample_count >= max_samples:
                break
        
        # Save the training data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created training data with {len(training_data)} samples")
        print(f"Saved to: {output_file}")
        
        # Show some examples
        if training_data:
            print("\nSample training data:")
            for i, sample in enumerate(training_data[:3]):
                print(f"\nSample {i+1}:")
                print(f"  Title: {sample['title']}")
                print(f"  Text: {sample['text'][:100]}...")
                print(f"  Time: {sample['start_time']} - {sample['end_time']}")
        
    except Exception as e:
        print(f"Error creating training data: {e}")

def main():
    """Main function to run safe analysis."""
    print("🔒 SAFE VIDCHAPTERS DATA ANALYSIS")
    print("This version processes data in small chunks to avoid memory crashes.")
    print()
    
    # Analyze structure first
    analyze_chapters_structure()
    analyze_asr_structure()
    
    # Create safe training data
    create_training_data_safe()
    
    print("\n✅ Safe analysis complete!")
    print("Check the generated JSON files for sample data structure.")

if __name__ == "__main__":
    main() 