import sys
import os
import json
import re
import random
from typing import List, Dict, Tuple
from collections import Counter

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def extract_sentences(text: str) -> List[str]:
    """Extract meaningful sentences from text"""
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter sentences
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 200:
            # Remove sentences that are just numbers or single words
            words = sentence.split()
            if len(words) >= 3 and not sentence.isdigit():
                clean_sentences.append(sentence)
    return clean_sentences

def extract_keywords_smart(text: str, top_k: int = 5) -> List[str]:
    """Smart keyword extraction using multiple techniques"""
    # Remove common words and punctuation
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Enhanced stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose',
        'just', 'very', 'really', 'quite', 'actually', 'basically', 'literally',
        'like', 'well', 'right', 'okay', 'yeah', 'yes', 'no', 'not', 'so', 'then', 'now',
        'um', 'uh', 'er', 'ah', 'oh', 'wow', 'hey', 'hi', 'hello', 'goodbye', 'bye',
        'one', 'two', 'three', 'first', 'second', 'third', 'day', 'days', 'time', 'times',
        'get', 'got', 'getting', 'go', 'going', 'gone', 'went', 'come', 'coming', 'came',
        'see', 'saw', 'seen', 'seeing', 'look', 'looking', 'looked', 'say', 'said', 'saying',
        'know', 'knew', 'knowing', 'think', 'thought', 'thinking', 'feel', 'felt', 'feeling',
        'want', 'wanted', 'wanting', 'need', 'needed', 'needing', 'make', 'made', 'making',
        'take', 'took', 'taken', 'taking', 'give', 'gave', 'given', 'giving', 'tell', 'told', 'telling'
    }
    
    # Count word frequencies with weighting
    word_scores = Counter()
    for word in words:
        if word not in stop_words and len(word) > 2:
            # Base score
            score = 1
            
            # Bonus for longer words (likely more specific)
            if len(word) > 5:
                score += 0.5
            
            # Bonus for words that appear in multiple contexts
            if word in text.lower().split():
                score += 0.3
            
            word_scores[word] += score
    
    # Get top keywords
    return [word.title() for word, _ in word_scores.most_common(top_k)]

def find_narrative_breaks_smart(texts: List[str]) -> List[int]:
    """Find natural narrative breaks using intelligent text analysis"""
    if len(texts) < 3:
        return [0]
    
    breaks = [0]  # Always start with first chunk
    
    # Transition words that indicate story changes
    transition_words = [
        'but', 'however', 'meanwhile', 'later', 'then', 'finally', 'suddenly',
        'next', 'after', 'before', 'while', 'during', 'when', 'if', 'because',
        'although', 'despite', 'instead', 'otherwise', 'therefore', 'thus',
        'consequently', 'as a result', 'in conclusion', 'to summarize'
    ]
    
    # Time indicators
    time_indicators = [
        'day', 'days', 'week', 'weeks', 'month', 'months', 'year', 'years',
        'hour', 'hours', 'minute', 'minutes', 'second', 'seconds',
        'morning', 'afternoon', 'evening', 'night', 'today', 'yesterday', 'tomorrow'
    ]
    
    for i in range(1, len(texts)):
        text_lower = texts[i].lower()
        
        # Check for transition words
        has_transition = any(word in text_lower for word in transition_words)
        
        # Check for time indicators
        has_time = any(word in text_lower for word in time_indicators)
        
        # Check for significant topic change (simple heuristic)
        prev_words = set(texts[i-1].lower().split())
        curr_words = set(text_lower.split())
        common_words = prev_words.intersection(curr_words)
        
        # If there's a transition or time indicator, or significant topic change
        if has_transition or has_time or len(common_words) < 3:
            breaks.append(i)
    
    return breaks

def score_sentence_for_title(sentence: str, keywords: List[str]) -> float:
    """Score a sentence for its suitability as a title"""
    score = 0.0
    sentence_lower = sentence.lower()
    
    # Length scoring (prefer medium-length sentences)
    word_count = len(sentence.split())
    if 4 <= word_count <= 8:
        score += 2.0
    elif 3 <= word_count <= 10:
        score += 1.0
    elif word_count > 15:
        score -= 1.0
    
    # Keyword presence scoring
    keyword_matches = 0
    for keyword in keywords:
        if keyword.lower() in sentence_lower:
            keyword_matches += 1
            score += 1.5
    
    # Bonus for sentences with multiple keywords
    if keyword_matches >= 2:
        score += 1.0
    
    # Bonus for question sentences
    if sentence.strip().endswith('?'):
        score += 0.5
    
    # Bonus for sentences starting with question words
    if sentence.lower().startswith(('what', 'how', 'why', 'when', 'where', 'which')):
        score += 1.0
    
    # Penalty for sentences that are too generic
    generic_phrases = ['this is', 'that is', 'it is', 'there is', 'here is']
    if any(phrase in sentence_lower for phrase in generic_phrases):
        score -= 0.5
    
    # Bonus for sentences with action words
    action_words = ['start', 'begin', 'continue', 'finish', 'complete', 'achieve', 'reach', 'get', 'make', 'do']
    if any(word in sentence_lower for word in action_words):
        score += 0.5
    
    return score

def generate_title_smart(text: str, keywords: List[str]) -> str:
    """Generate title using smart analysis"""
    sentences = extract_sentences(text)
    
    if not sentences:
        return generate_title_from_keywords(keywords)
    
    # Score all sentences
    sentence_scores = []
    for sentence in sentences:
        score = score_sentence_for_title(sentence, keywords)
        sentence_scores.append((sentence, score))
    
    # Get the best sentence
    if sentence_scores:
        best_sentence, best_score = max(sentence_scores, key=lambda x: x[1])
        
        # Only use sentence if it has a good score
        if best_score >= 1.0:
            return clean_title(best_sentence)
    
    # Fallback to keyword-based title
    return generate_title_from_keywords(keywords)

def generate_title_from_keywords(keywords: List[str]) -> str:
    """Generate title from keywords"""
    if len(keywords) >= 2:
        title_patterns = [
            f"{keywords[0]} and {keywords[1]}",
            f"{keywords[0]} for {keywords[1]}",
            f"{keywords[0]} with {keywords[1]}",
            f"The {keywords[0]} {keywords[1]}",
            f"{keywords[0]} {keywords[1]}",
            f"{keywords[0]} - {keywords[1]}"
        ]
        return clean_title(random.choice(title_patterns))
    elif len(keywords) == 1:
        return clean_title(keywords[0])
    
    return "Chapter"

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
    elif len(title) > 60:
        title = title[:57] + "..."
    
    return title

def analyze_content_structure(transcript: List[dict]) -> Dict:
    """Analyze the overall structure and content of the transcript"""
    full_text = " ".join(seg["text"] for seg in transcript)
    
    # Extract key themes and topics
    keywords = extract_keywords_smart(full_text, 10)
    
    # Analyze temporal patterns
    total_duration = transcript[-1]["start"] + transcript[-1]["duration"] if transcript else 0
    
    # Find potential story beats
    story_beats = []
    for i, seg in enumerate(transcript):
        text = seg["text"].lower()
        # Look for transition words and phrases
        transition_phrases = [
            'but', 'however', 'meanwhile', 'later', 'then', 'finally', 'suddenly',
            'next', 'after', 'before', 'while', 'during', 'when', 'if', 'because',
            'although', 'despite', 'instead', 'otherwise', 'therefore', 'thus',
            'consequently', 'as a result', 'in conclusion', 'to summarize'
        ]
        if any(phrase in text for phrase in transition_phrases):
            story_beats.append(i)
    
    # Find emotional or dramatic moments
    dramatic_moments = []
    dramatic_words = [
        'amazing', 'incredible', 'unbelievable', 'crazy', 'insane', 'wow', 'oh my god',
        'finally', 'success', 'achieved', 'won', 'lost', 'failed', 'succeeded',
        'challenge', 'difficult', 'hard', 'easy', 'simple', 'complicated'
    ]
    
    for i, seg in enumerate(transcript):
        text = seg["text"].lower()
        if any(word in text for word in dramatic_words):
            dramatic_moments.append(i)
    
    return {
        'keywords': keywords,
        'total_duration': total_duration,
        'story_beats': story_beats,
        'dramatic_moments': dramatic_moments,
        'word_count': len(full_text.split())
    }

def find_optimal_chapter_boundaries(transcript: List[dict], n_chapters: int) -> List[int]:
    """Find optimal chapter boundaries using multiple strategies"""
    total_segments = len(transcript)
    
    # Strategy 1: Use story beats if available
    structure = analyze_content_structure(transcript)
    story_beats = structure['story_beats']
    dramatic_moments = structure['dramatic_moments']
    
    if len(story_beats) >= n_chapters - 1:
        # Use story beats for chapter boundaries
        boundaries = [0] + sorted(story_beats[:n_chapters-1]) + [total_segments]
        return boundaries
    
    # Strategy 2: Use dramatic moments
    if len(dramatic_moments) >= n_chapters - 1:
        boundaries = [0] + sorted(dramatic_moments[:n_chapters-1]) + [total_segments]
        return boundaries
    
    # Strategy 3: Use semantic analysis
    chunk_size = total_segments // n_chapters
    chunks = []
    for i in range(n_chapters):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < n_chapters - 1 else total_segments
        chunk_text = " ".join(seg["text"] for seg in transcript[start_idx:end_idx])
        chunks.append(chunk_text)
    
    # Find natural breaks
    breaks = find_narrative_breaks_smart(chunks)
    
    # Adjust boundaries based on breaks
    boundaries = []
    for i, break_point in enumerate(breaks):
        if i == 0:
            boundaries.append(0)
        else:
            boundaries.append(break_point * chunk_size)
    boundaries.append(total_segments)
    
    return boundaries

def transcript_to_chapters_smart(transcript: List[dict], n_chapters: int = 5) -> str:
    """Generate chapters using smart hybrid approach"""
    
    if not transcript:
        return "00:00 - Chapter"
    
    # Find optimal chapter boundaries
    boundaries = find_optimal_chapter_boundaries(transcript, n_chapters)
    
    # Generate chapters
    chapters = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        # Get chunk text
        chunk = transcript[start_idx:end_idx]
        chunk_text = " ".join(seg["text"] for seg in chunk)
        
        # Extract keywords for this chunk
        chunk_keywords = extract_keywords_smart(chunk_text, 3)
        
        # Generate title
        title = generate_title_smart(chunk_text, chunk_keywords)
        
        # Get timestamp
        timestamp = format_timestamp(transcript[start_idx]["start"])
        
        chapters.append(f"{timestamp} - {title}")
    
    return "\n".join(chapters)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_chapters_smart_hybrid.py transcript.json [n_chapters]")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            transcript = json.load(f)
        
        n_chapters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        print("[INFO] Using smart hybrid approach with intelligent text analysis")
        chapters = transcript_to_chapters_smart(transcript, n_chapters)
        print(chapters)
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {sys.argv[1]}")
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON file: {sys.argv[1]}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}") 