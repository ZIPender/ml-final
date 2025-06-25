import sys
import os
import json
import re
import random
from typing import List, Dict, Tuple
from collections import Counter
import numpy as np

# Try to import ML libraries, fall back gracefully if not available
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[INFO] ML libraries not available, using enhanced rule-based approach")

def format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format"""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\-\'\"]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

def extract_keywords_advanced(text: str, top_k: int = 5) -> List[str]:
    """Advanced keyword extraction using TF-IDF and frequency analysis"""
    if not ML_AVAILABLE:
        return extract_keywords_simple(text, top_k)
    
    try:
        # Use TF-IDF for keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Split text into chunks for better analysis
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        if chunks:
            tfidf_matrix = vectorizer.fit_transform(chunks)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords by TF-IDF scores
            tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
            
            keywords = [feature_names[i] for i in top_indices]
            return [kw.replace('_', ' ').title() for kw in keywords]
    except Exception as e:
        print(f"[WARNING] TF-IDF failed: {e}, falling back to simple extraction")
    
    return extract_keywords_simple(text, top_k)

def extract_keywords_simple(text: str, top_k: int = 5) -> List[str]:
    """Simple keyword extraction using frequency analysis"""
    # Remove common words and punctuation
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom', 'whose',
        'just', 'very', 'really', 'quite', 'actually', 'basically', 'literally',
        'like', 'well', 'right', 'okay', 'yeah', 'yes', 'no', 'not', 'so', 'then', 'now',
        'um', 'uh', 'er', 'ah', 'oh', 'wow', 'hey', 'hi', 'hello', 'goodbye', 'bye',
        'one', 'two', 'three', 'first', 'second', 'third', 'day', 'days', 'time', 'times'
    }
    
    # Count word frequencies
    word_counts = Counter()
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_counts[word] += 1
    
    # Get top keywords
    return [word.title() for word, _ in word_counts.most_common(top_k)]

def analyze_semantic_similarity(texts: List[str]) -> np.ndarray:
    """Analyze semantic similarity between text chunks"""
    if not ML_AVAILABLE or len(texts) < 2:
        return np.eye(len(texts))
    
    try:
        # Use sentence transformers for semantic similarity
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings
        embeddings = model.encode(texts)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    except Exception as e:
        print(f"[WARNING] Semantic analysis failed: {e}, using identity matrix")
        return np.eye(len(texts))

def find_narrative_breaks(texts: List[str], similarity_threshold: float = 0.3) -> List[int]:
    """Find natural narrative breaks based on semantic similarity"""
    if len(texts) < 3:
        return [0]
    
    similarity_matrix = analyze_semantic_similarity(texts)
    
    # Find points where similarity drops significantly
    breaks = [0]  # Always start with first chunk
    
    for i in range(1, len(texts)):
        # Check similarity with previous chunk
        if similarity_matrix[i][i-1] < similarity_threshold:
            breaks.append(i)
    
    return breaks

def generate_title_ml_enhanced(text: str, keywords: List[str]) -> str:
    """Generate title using ML-enhanced approach"""
    if not ML_AVAILABLE:
        return generate_title_rule_based(text, keywords)
    
    try:
        # Use a simple summarization approach
        sentences = extract_sentences(text)
        if not sentences:
            return generate_title_rule_based(text, keywords)
        
        # Score sentences based on keyword presence and length
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Score based on keyword presence
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    score += 2
            
            # Score based on length (prefer medium-length sentences)
            length = len(sentence.split())
            if 5 <= length <= 12:
                score += 1
            elif length > 20:
                score -= 1
            
            # Bonus for sentences that start with question words
            if sentence.lower().startswith(('what', 'how', 'why', 'when', 'where')):
                score += 1
            
            sentence_scores.append((sentence, score))
        
        # Get the best sentence
        if sentence_scores:
            best_sentence, _ = max(sentence_scores, key=lambda x: x[1])
            return clean_title(best_sentence)
    
    except Exception as e:
        print(f"[WARNING] ML title generation failed: {e}, falling back to rule-based")
    
    return generate_title_rule_based(text, keywords)

def generate_title_rule_based(text: str, keywords: List[str]) -> str:
    """Generate title using rule-based approach"""
    sentences = extract_sentences(text)
    
    # Look for good sentence candidates
    for sentence in sentences:
        # Prefer sentences with keywords
        sentence_lower = sentence.lower()
        keyword_count = sum(1 for kw in keywords if kw.lower() in sentence_lower)
        
        if keyword_count > 0 and 5 <= len(sentence.split()) <= 15:
            return clean_title(sentence)
    
    # If no good sentences found, create title from keywords
    if len(keywords) >= 2:
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
    
    # Fallback to first meaningful sentence
    for sentence in sentences:
        if len(sentence.split()) >= 3:
            return clean_title(sentence)
    
    return "Chapter"

def clean_title(title: str) -> str:
    """Clean and improve the generated title, and shorten to max 8-10 words"""
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
    # Shorten to max 8-10 words
    max_words = 10
    if len(words) > max_words:
        title = ' '.join(words[:max_words])
    else:
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
    keywords = extract_keywords_advanced(full_text, 10)
    
    # Analyze temporal patterns
    total_duration = transcript[-1]["start"] + transcript[-1]["duration"] if transcript else 0
    
    # Find potential story beats
    story_beats = []
    for i, seg in enumerate(transcript):
        text = seg["text"].lower()
        # Look for transition words and phrases
        if any(phrase in text for phrase in ['but', 'however', 'meanwhile', 'later', 'then', 'finally', 'suddenly']):
            story_beats.append(i)
    
    return {
        'keywords': keywords,
        'total_duration': total_duration,
        'story_beats': story_beats,
        'word_count': len(full_text.split())
    }

def chunk_transcript_by_time(transcript: List[dict], n_chapters: int) -> List[List[dict]]:
    """Split transcript into n_chapters chunks by time, not by segment count"""
    if not transcript or n_chapters < 1:
        return [transcript]
    total_duration = transcript[-1]["start"] + transcript[-1]["duration"]
    chunk_length = total_duration / n_chapters
    chunks = [[] for _ in range(n_chapters)]
    for seg in transcript:
        seg_time = seg["start"]
        idx = min(int(seg_time // chunk_length), n_chapters - 1)
        chunks[idx].append(seg)
    return chunks

def summarize_text_ml(text: str) -> str:
    """Use ML summarization to generate a short title from text chunk"""
    if not ML_AVAILABLE:
        return ""
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
        # Limit input length for summarizer
        max_input = 512
        sentences = extract_sentences(text)
        input_text = " ".join(sentences)[:max_input]
        summary = summarizer(input_text, max_length=15, min_length=4, do_sample=False)
        if summary and isinstance(summary, list):
            return summary[0]["summary_text"].strip()
    except Exception as e:
        print(f"[WARNING] Summarization failed: {e}")
    return ""

def transcript_to_chapters_advanced(transcript: List[dict], n_chapters: int = 5) -> str:
    """Generate chapters using advanced hybrid approach with time-based chunking and ML summarization"""
    if not transcript:
        return "00:00 - Chapter"
    # Chunk transcript by time
    chunks = chunk_transcript_by_time(transcript, n_chapters)
    chapters = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        chunk_text = " ".join(seg["text"] for seg in chunk)
        chunk_keywords = extract_keywords_advanced(chunk_text, 3)
        # Try ML summarization for title
        title = ""
        if ML_AVAILABLE:
            title = summarize_text_ml(chunk_text)
        if not title:
            # Fallback to ML-enhanced or rule-based
            title = generate_title_ml_enhanced(chunk_text, chunk_keywords)
        # Shorten title
        title = clean_title(title)
        timestamp = format_timestamp(chunk[0]["start"])
        chapters.append(f"{timestamp} - {title}")
    return "\n".join(chapters)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_chapters_advanced_hybrid.py transcript.json [n_chapters]")
        sys.exit(1)
    
    try:
        with open(sys.argv[1], "r", encoding='utf-8') as f:
            transcript = json.load(f)
        
        n_chapters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        
        info = f"Using {'ML-enhanced approach' if ML_AVAILABLE else 'rule-based approach'}"
        chapters = transcript_to_chapters_advanced(transcript, n_chapters)
        # Output as JSON for backend/frontend
        print(json.dumps({"info": info, "chapters": chapters}, ensure_ascii=False))
        
    except FileNotFoundError:
        print(json.dumps({"error": f"File not found: {sys.argv[1]}"}))
    except json.JSONDecodeError:
        print(json.dumps({"error": f"Invalid JSON file: {sys.argv[1]}"}))
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {e}"})) 