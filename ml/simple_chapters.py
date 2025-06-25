import sys
import json
import re
from typing import List

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def extract_keywords(text: str) -> str:
    """Extract key phrases from text to create chapter titles"""
    # Remove common words and punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    
    # Filter out common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    # Get meaningful words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    if not keywords:
        return "Introduction"
    
    # Take first few meaningful words
    title_words = keywords[:3]
    title = ' '.join(word.capitalize() for word in title_words)
    
    return title

def transcript_to_chapters(transcript: List[dict], lang: str = "en", n_chapters: int = 5) -> str:
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
        
        # Generate title from keywords
        title = extract_keywords(chunk_text)
        timestamp = format_timestamp(transcript[idx]["start"])
        chapters.append(f"{timestamp} - {title}")
        print(f"[OK] Chapter {i+1}: {timestamp} - {title}")
    
    return "\n".join(chapters)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_chapters.py transcript.json [lang]")
        print("[INFO] This script generates chapters without requiring ML models")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            transcript = json.load(f)
        lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        
        print(f"[INFO] Processing transcript with {len(transcript)} segments...")
        print("[INFO] Using simple keyword extraction (no ML models required)")
        
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