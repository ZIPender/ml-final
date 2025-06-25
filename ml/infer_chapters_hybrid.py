import sys
import os
import json
import random
import re
from typing import List

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def clean_title(title: str) -> str:
    """Clean and improve the generated title"""
    # Remove quotes and extra punctuation
    title = title.replace('"', '').replace("'", "").replace(". ", "").replace(".", "").strip()
    title = re.sub(r'^[-–—]+', '', title)  # Remove leading dashes
    title = re.sub(r'[-–—]+$', '', title)  # Remove trailing dashes
    
    # Capitalize properly
    words = title.split()
    if words:
        # Capitalize first word
        words[0] = words[0].capitalize()
        
        # Capitalize other important words (skip common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        for i in range(1, len(words)):
            if words[i].lower() not in common_words:
                words[i] = words[i].capitalize()
    
    title = ' '.join(words)
    
    # Ensure reasonable length
    if len(title) < 3:
        title = "Chapter"
    elif len(title) > 50:
        title = title[:47] + "..."
    
    return title

def extract_keywords_from_transcript(text: str, max_keywords: int = 3) -> List[str]:
    """Extract meaningful keywords from transcript text"""
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose',
        'just', 'very', 'really', 'quite', 'actually', 'basically', 'literally',
        'like', 'well', 'right', 'okay', 'yeah', 'yes', 'no', 'not', 'so', 'then', 'now',
        'um', 'uh', 'er', 'ah', 'oh', 'wow', 'hey', 'hi', 'hello', 'goodbye', 'bye'
    }
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if (word not in stop_words and 
            len(word) > 2 and 
            len(word) < 15 and
            not word.isdigit() and
            not any(c.isdigit() for c in word)):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word.capitalize() for word, _ in sorted_words[:max_keywords]]

def generate_title_from_transcript(text: str) -> str:
    """Generate a title directly from the transcript content"""
    
    # Look for complete sentences or phrases that could be titles
    sentences = re.split(r'[.!?]', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Look for meaningful sentences (not too short, not too long)
        if 10 <= len(sentence) <= 40:
            # Check if it starts with a question word
            if sentence.lower().startswith(('what', 'how', 'why', 'when', 'where', 'which')):
                return clean_title(sentence)
            
            # Check if it's a complete thought
            if len(sentence.split()) >= 3 and len(sentence.split()) <= 8:
                return clean_title(sentence)
    
    # If no good sentences found, extract keywords and create a title
    keywords = extract_keywords_from_transcript(text, 2)
    
    if len(keywords) >= 2:
        # Create a meaningful title from keywords
        title_patterns = [
            f"{keywords[0]} and {keywords[1]}",
            f"{keywords[0]} for {keywords[1]}",
            f"{keywords[0]} with {keywords[1]}",
            f"The {keywords[0]} {keywords[1]}",
            f"{keywords[0]} {keywords[1]}"
        ]
        return clean_title(random.choice(title_patterns))
    elif len(keywords) == 1:
        return clean_title(keywords[0])
    else:
        return "Chapter"

def transcript_to_chapters(transcript: List[dict], lang: str = "en", n_chapters: int = 5) -> str:
    """Generate chapters using hybrid approach - transcript-based with semantic analysis"""
    
    total = len(transcript)
    if total < n_chapters:
        n_chapters = max(1, total)
    step = total // n_chapters
    chapters = []
    
    for i in range(n_chapters):
        idx = i * step
        if idx >= total:
            idx = total - 1
        chunk = transcript[idx: idx + step] if i < n_chapters - 1 else transcript[idx:]
        chunk_text = " ".join(seg["text"] for seg in chunk)
        
        # Generate title from transcript content
        title = generate_title_from_transcript(chunk_text)
        
        timestamp = format_timestamp(transcript[idx]["start"])
        chapters.append(f"{timestamp} - {title}")
    
    return "\n".join(chapters)

if __name__ == "__main__":
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python infer_chapters_hybrid.py transcript.json [lang]")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            transcript = json.load(f)
        lang = sys.argv[2] if len(sys.argv) > 2 else "en"
        
        chapters = transcript_to_chapters(transcript, lang)
        print(chapters)
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {sys.argv[1]}")
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON file: {sys.argv[1]}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}") 