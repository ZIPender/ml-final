import pickle
import sys
import os

def explore_pkl_file(file_path, max_samples=3):
    """Explore the structure of a pickle file"""
    print(f"\n=== Exploring {os.path.basename(file_path)} ===")
    
    try:
        print(f"Loading {file_path}...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        
        if isinstance(data, list):
            print(f"First item type: {type(data[0])}")
            if len(data) > 0:
                if isinstance(data[0], dict):
                    print(f"First item keys: {list(data[0].keys())}")
                else:
                    print(f"First item: {data[0]}")
                
            # Show first few samples
            for i, sample in enumerate(data[:max_samples]):
                print(f"\n--- Sample {i+1} ---")
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"{key}: {value[:100]}...")
                        else:
                            print(f"{key}: {value}")
                else:
                    print(f"Value: {sample}")
                    
        else:
            print(f"Data content (first 500 chars): {str(data)[:500]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()

def main():
    data_dir = "../data"
    
    # Explore chapters.pkl
    chapters_path = os.path.join(data_dir, "chapters.pkl")
    if os.path.exists(chapters_path):
        explore_pkl_file(chapters_path)
    else:
        print("chapters.pkl not found")
    
    # Explore asr.pkl (this might be too large, so just check structure)
    asr_path = os.path.join(data_dir, "asr.pkl")
    if os.path.exists(asr_path):
        print(f"\n=== ASR file info ===")
        print(f"File size: {os.path.getsize(asr_path) / (1024**3):.2f} GB")
        # Don't load the full ASR file as it's very large
        print("ASR file is very large (19GB), skipping full exploration for now")
    else:
        print("asr.pkl not found")

if __name__ == "__main__":
    main() 