import pickle
import sys
import os

def explore_chapters_sample(file_path, sample_size=5):
    """Explore a small sample of the chapters data"""
    print(f"\n=== Exploring chapters.pkl sample ===")
    
    try:
        print(f"Loading sample from {file_path}...")
        with open(file_path, 'rb') as f:
            # Load just a small portion to understand structure
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())[:10]}...")  # Show first 10 keys
            
            # Get a few sample items
            sample_keys = list(data.keys())[:sample_size]
            for i, key in enumerate(sample_keys):
                print(f"\n--- Sample {i+1} (key: {key}) ---")
                value = data[key]
                if isinstance(value, dict):
                    print(f"Value type: dict with keys: {list(value.keys())}")
                    for k, v in value.items():
                        if isinstance(v, str) and len(v) > 100:
                            print(f"  {k}: {v[:100]}...")
                        else:
                            print(f"  {k}: {v}")
                elif isinstance(value, list):
                    print(f"Value type: list with {len(value)} items")
                    if len(value) > 0:
                        print(f"  First item: {value[0]}")
                else:
                    print(f"Value type: {type(value)}")
                    print(f"  Value: {value}")
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

def main():
    data_dir = "../data"
    
    # Explore chapters.pkl sample
    chapters_path = os.path.join(data_dir, "chapters.pkl")
    if os.path.exists(chapters_path):
        explore_chapters_sample(chapters_path)
    else:
        print("chapters.pkl not found")
    
    # Just show ASR file info
    asr_path = os.path.join(data_dir, "asr.pkl")
    if os.path.exists(asr_path):
        print(f"\n=== ASR file info ===")
        print(f"File size: {os.path.getsize(asr_path) / (1024**3):.2f} GB")
        print("ASR file is very large (19GB), will explore structure separately")

if __name__ == "__main__":
    main() 