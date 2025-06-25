import sys
import os
import json
import random
import re
from typing import List

def load_training_patterns(training_file: str = "training_data.json"):
    """Load patterns from training data to improve generation"""
    if not os.path.exists(training_file):
        return None
    
    try:
        with open(training_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract good titles (filter out incomplete ones)
        good_titles = []
        for sample in data:
            title = sample['output']
            # Filter out incomplete or problematic titles
            if (len(title) >= 5 and 
                len(title) <= 40 and 
                not title.islower() and 
                not title.isupper() and
                not title.endswith('...') and
                not any(char.isdigit() for char in title[-3:])):  # Avoid titles ending with numbers
                good_titles.append(title)
        
        return good_titles[:200]  # Use top 200 good examples
    except Exception as e:
        print(f"Warning: Could not load training patterns: {e}")
        return None

def format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def clean_title(title: str) -> str:
    """Clean and improve the generated title"""
    # Remove quotes and extra punctuation
    title = title.replace('"', '').replace("'", "").replace(". ", "").replace(".", "").strip()
    
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

def extract_meaningful_keywords(text: str, max_keywords: int = 3) -> List[str]:
    """Extract meaningful keywords from text, avoiding names and focusing on content"""
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words and problematic words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose',
        'just', 'very', 'really', 'quite', 'actually', 'basically', 'literally',
        'like', 'well', 'right', 'okay', 'yeah', 'yes', 'no', 'not', 'so', 'then', 'now'
    }
    
    # Count word frequencies
    word_counts = {}
    for word in words:
        if (word not in stop_words and 
            len(word) > 2 and 
            not word.isupper() and  # Avoid acronyms
            not re.match(r'^[A-Z][a-z]+$', word)):  # Avoid proper nouns (names)
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Get top keywords, but filter out names and meaningless words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    meaningful_keywords = []
    
    for word, count in sorted_words:
        # Skip if it looks like a name (capitalized in original text)
        if word.capitalize() in text:
            continue
        # Skip if it's too short or too long
        if len(word) < 3 or len(word) > 12:
            continue
        # Skip if it's just a number or contains numbers
        if word.isdigit() or any(c.isdigit() for c in word):
            continue
            
        meaningful_keywords.append(word.capitalize())
        if len(meaningful_keywords) >= max_keywords:
            break
    
    return meaningful_keywords

def generate_semantic_title(text: str) -> str:
    """Generate a title based on the semantic content of the text"""
    
    # Look for specific patterns in the text
    text_lower = text.lower()
    
    # Check for question patterns
    if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'which']):
        # Extract the question
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['what', 'how', 'why', 'when', 'where', 'which']):
                # Clean up the question
                question = sentence.strip()
                if len(question) > 10 and len(question) < 50:
                    return clean_title(question)
    
    # Check for action patterns
    action_words = ['learn', 'understand', 'explore', 'discover', 'build', 'create', 'develop', 'implement', 'analyze', 'review', 'discuss', 'examine', 'investigate', 'study', 'practice', 'train', 'work', 'play', 'use', 'apply']
    
    for action in action_words:
        if action in text_lower:
            # Find the object of the action
            words = text_lower.split()
            try:
                action_idx = words.index(action)
                if action_idx + 1 < len(words):
                    object_word = words[action_idx + 1]
                    if len(object_word) > 2 and object_word not in ['the', 'a', 'an', 'and', 'or']:
                        return f"{action.capitalize()} {object_word.capitalize()}"
            except ValueError:
                continue
    
    # Check for topic patterns
    topic_indicators = ['about', 'regarding', 'concerning', 'related to', 'focus on', 'discuss', 'talk about']
    for indicator in topic_indicators:
        if indicator in text_lower:
            # Extract the topic
            parts = text_lower.split(indicator)
            if len(parts) > 1:
                topic = parts[1].strip().split()[0]
                if len(topic) > 2:
                    return f"About {topic.capitalize()}"
    
    # Fallback: use meaningful keywords
    keywords = extract_meaningful_keywords(text, 2)
    if len(keywords) >= 2:
        return f"{keywords[0]} and {keywords[1]}"
    elif len(keywords) == 1:
        return keywords[0]
    else:
        return "Chapter"

def transcript_to_chapters(transcript: List[dict], lang: str = "en", n_chapters: int = 5) -> str:
    """Generate chapters using semantic content analysis"""
    
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
        
        # Generate title based on semantic content
        title = generate_semantic_title(chunk_text)
        title = clean_title(title)
        
        timestamp = format_timestamp(transcript[idx]["start"])
        chapters.append(f"{timestamp} - {title}")
    
    return "\n".join(chapters)

if __name__ == "__main__":
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python infer_chapters_patterns.py transcript.json [lang]")
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