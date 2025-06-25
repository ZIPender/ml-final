import pickle
import pandas as pd
import sys
import os

def explore_pkl_file(file_path, max_samples=5):
    """Explore the structure of a pickle file"""
    print(f"\n=== Exploring {os.path.basename(file_path)} ===")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        
        if isinstance(data, list):
            print(f"First item type: {type(data[0])}")
            if len(data) > 0:
                print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                
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
                    
        elif isinstance(data, pd.DataFrame):
            print(f"DataFrame shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print("\nFirst few rows:")
            print(data.head())
            
        else:
            print(f"Data content (first 500 chars): {str(data)[:500]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    data_dir = "../data"
    
    # Explore chapters.pkl
    chapters_path = os.path.join(data_dir, "chapters.pkl")
    if os.path.exists(chapters_path):
        explore_pkl_file(chapters_path)
    else:
        print("chapters.pkl not found")
    
    # Explore asr.pkl
    asr_path = os.path.join(data_dir, "asr.pkl")
    if os.path.exists(asr_path):
        explore_pkl_file(asr_path)
    else:
        print("asr.pkl not found")

if __name__ == "__main__":
    main() 