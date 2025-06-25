import sys
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import torch

# Use a much smaller, more reliable model for chapterization
MODEL_NAME = "sshleifer/distilbart-cnn-6-6"  # Only ~50MB, much more reliable

def load_summarizer():
    """Load the summarization pipeline with error handling"""
    try:
        # Check if model is already cached
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, "models--sshleifer--distilbart-cnn-6-6")
        
        if os.path.exists(model_path):
            print("[OK] Using cached model")
            return pipeline("summarization", model=MODEL_NAME, device=-1)  # Use CPU
        
        print("[DOWNLOAD] Downloading model (this may take a few minutes)...")
        print("[INFO] Model size: ~50MB")
        
        # Load with progress tracking
        summarizer = pipeline("summarization", model=MODEL_NAME, device=-1)
        print("[OK] Model downloaded successfully!")
        return summarizer
        
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("[INFO] Trying alternative model...")
        
        # Try an even smaller alternative
        try:
            print("[DOWNLOAD] Trying alternative model: facebook/bart-base-cnn")
            summarizer = pipeline("summarization", model="facebook/bart-base-cnn", device=-1)
            print("[OK] Alternative model loaded successfully!")
            return summarizer
        except Exception as e2:
            print(f"[ERROR] Alternative model also failed: {e2}")
            print("[INFO] This might be due to network issues or insufficient disk space")
            print("[HELP] Try running the script again, or check your internet connection")
            return None

# Initialize the summarizer
summarizer = load_summarizer()

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def to_title_case(s: str) -> str:
    return ' '.join([w.capitalize() for w in s.split()])

def transcript_to_chapters(transcript: List[dict], lang: str = "en", n_chapters: int = 5) -> str:
    if summarizer is None:
        return "[ERROR] Model not loaded. Cannot generate chapters."
    
    total = len(transcript)
    if total < n_chapters:
        n_chapters = max(1, total)
    step = total // n_chapters
    chapters = []
    
    print(f"[PROCESS] Generating {n_chapters} chapters from {total} transcript segments...")
    
    for i in range(n_chapters):
        idx = i * step
        if idx >= total:
            idx = total - 1
        chunk = transcript[idx: idx + step] if i < n_chapters - 1 else transcript[idx:]
        chunk_text = " ".join(seg["text"] for seg in chunk)
        chunk_text = chunk_text[:1000]  # avoid model input limits
        
        prompt = (
            "Give a 2-5 word YouTube chapter title for this transcript section. "
            "Do NOT use quotes. Do NOT use full sentences. Only a short, descriptive title. "
            f"Section: {chunk_text}"
        )
        
        try:
            result = summarizer(prompt, max_length=6, min_length=2, do_sample=False)
            title = result[0]["summary_text"].replace('"', '').replace("'", "").replace(". ", "").replace(".", "").strip()
            title = to_title_case(title)
            timestamp = format_timestamp(transcript[idx]["start"])
            chapters.append(f"{timestamp} - {title}")
            print(f"[OK] Chapter {i+1}: {timestamp} - {title}")
        except Exception as e:
            print(f"[ERROR] Error generating chapter {i+1}: {e}")
            # Fallback to simple title
            timestamp = format_timestamp(transcript[idx]["start"])
            chapters.append(f"{timestamp} - Chapter {i+1}")
    
    return "\n".join(chapters)

if __name__ == "__main__":
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python infer_chapters_debug.py transcript.json [lang]")
        print("[INFO] This is the debug version with full output")
        sys.exit(1)
    
    try:
        # Usage: python infer_chapters_debug.py transcript.json [lang]
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            transcript = json.load(f)
        lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        
        print(f"[INFO] Processing transcript with {len(transcript)} segments...")
        chapters = transcript_to_chapters(transcript, lang)
        print("\n[RESULT] Generated Chapters:")
        print("=" * 50)
        print(chapters)
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {sys.argv[1]}")
        print("[HELP] Make sure the transcript file exists and the path is correct")
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON file: {sys.argv[1]}")
        print("[HELP] Make sure the file contains valid JSON data")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}") 