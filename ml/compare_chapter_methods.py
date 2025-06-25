import sys
import os
import json
import subprocess
from typing import List, Dict

def run_chapter_generator(script_name: str, transcript_file: str, n_chapters: int = 5) -> str:
    """Run a chapter generation script and return the output"""
    try:
        script_dir = os.path.dirname(script_name)
        cwd = script_dir if script_dir else None
        result = subprocess.run(
            [sys.executable, script_name, transcript_file, str(n_chapters)],
            capture_output=True,
            text=True,
            cwd=cwd
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Error: {str(e)}"

def compare_methods(transcript_file: str, n_chapters: int = 5):
    """Compare different chapter generation methods"""
    
    print("=" * 80)
    print("CHAPTER GENERATION METHODS COMPARISON")
    print("=" * 80)
    print(f"Transcript: {transcript_file}")
    print(f"Number of chapters: {n_chapters}")
    print()
    
    # List of methods to compare
    methods = [
        {
            'name': 'Original Hybrid',
            'script': 'infer_chapters_hybrid.py',
            'description': 'Basic transcript-based with simple text analysis'
        },
        {
            'name': 'Smart Hybrid',
            'script': 'infer_chapters_smart_hybrid.py',
            'description': 'Intelligent text analysis with narrative break detection'
        },
        {
            'name': 'Advanced Hybrid (ML)',
            'script': 'infer_chapters_advanced_hybrid.py',
            'description': 'ML-enhanced with semantic similarity analysis'
        }
    ]
    
    results = {}
    
    for method in methods:
        print(f"Testing: {method['name']}")
        print(f"Description: {method['description']}")
        print("-" * 60)
        
        output = run_chapter_generator(method['script'], transcript_file, n_chapters)
        results[method['name']] = output
        
        print(output)
        print()
        print("=" * 80)
        print()
    
    # Summary comparison
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    for method in methods:
        name = method['name']
        output = results[name]
        
        if output.startswith("Error:"):
            print(f"[FAILED] {name}: {output}")
        else:
            chapters = output.split('\n')
            print(f"[SUCCESS] {name}: {len(chapters)} chapters generated")
            
            # Show first and last chapter for comparison
            if chapters:
                print(f"   First: {chapters[0]}")
                if len(chapters) > 1:
                    print(f"   Last:  {chapters[-1]}")
        print()

def analyze_quality(transcript_file: str):
    """Analyze the quality of generated chapters"""
    print("QUALITY ANALYSIS")
    print("=" * 80)
    
    # Load transcript to analyze content
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        total_duration = transcript[-1]["start"] + transcript[-1]["duration"] if transcript else 0
        total_words = sum(len(seg["text"].split()) for seg in transcript)
        
        print(f"Transcript Statistics:")
        print(f"  - Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"  - Total segments: {len(transcript)}")
        print(f"  - Total words: {total_words}")
        print(f"  - Average words per segment: {total_words/len(transcript):.1f}")
        
        # Analyze content themes
        full_text = " ".join(seg["text"] for seg in transcript)
        words = full_text.lower().split()
        
        # Find most common words (excluding stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"  - Top themes: {', '.join(word.title() for word, _ in top_words)}")
        
    except Exception as e:
        print(f"Error analyzing transcript: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_chapter_methods.py transcript.json [n_chapters]")
        sys.exit(1)
    
    transcript_file = sys.argv[1]
    n_chapters = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Analyze transcript quality
    analyze_quality(transcript_file)
    print()
    
    # Compare methods
    compare_methods(transcript_file, n_chapters) 