from fastapi import FastAPI, HTTPException, Query
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import sys
import os
import json
import subprocess

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_video_id(url: str) -> str:
    import re
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

@app.get("/transcript")
def get_transcript(url: str = Query(..., description="YouTube video URL")):
    try:
        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/chapters")
def get_chapters(url: str = Query(..., description="YouTube video URL"), 
                n_chapters: int = Query(5, description="Number of chapters to generate")):
    """
    Returns a formatted string of timestamped chapters for YouTube description.
    Uses advanced ML-powered approach with semantic understanding and intelligent text analysis.
    """
    try:
        video_id = extract_video_id(url)
        # Try to get transcript in English, fallback to French, else any
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            lang = "en"
        except Exception:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["fr"])
                lang = "fr"
            except Exception:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                lang = transcript[0].get('language', 'unknown')
        
        # Save transcript to temp file
        temp_path = "ml/temp_transcript.json"
        with open(temp_path, "w", encoding='utf-8') as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)
        
        # Use advanced ML-powered approach first (best results)
        try:
            result = subprocess.run([
                sys.executable, "ml/infer_chapters_advanced_hybrid.py", temp_path, str(n_chapters)
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    parsed = json.loads(result.stdout.strip())
                    chapters = parsed["chapters"]
                    info = parsed.get("info", "")
                except Exception:
                    chapters = result.stdout.strip()
                    info = ""
                method_used = "Advanced ML-powered"
            else:
                # Fallback to smart hybrid approach
                result = subprocess.run([
                    sys.executable, "ml/infer_chapters_smart_hybrid.py", temp_path, str(n_chapters)
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and result.stdout.strip():
                    chapters = result.stdout.strip()
                    info = ""
                    method_used = "Smart hybrid"
                else:
                    # Fallback to original hybrid approach
                    result = subprocess.run([
                        sys.executable, "ml/infer_chapters_hybrid.py", temp_path, str(n_chapters)
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0 and result.stdout.strip():
                        chapters = result.stdout.strip()
                        info = ""
                        method_used = "Original hybrid"
                    else:
                        # Final fallback to simple chapters
                        result = subprocess.run([
                            sys.executable, "ml/simple_chapters.py", temp_path, str(n_chapters)
                        ], capture_output=True, text=True, timeout=60)
                        
                        if result.returncode != 0:
                            raise RuntimeError(result.stderr)
                        chapters = result.stdout.strip()
                        info = ""
                        method_used = "Simple pattern-based"
        except Exception as e:
            # Final fallback to simple chapters
            result = subprocess.run([
                sys.executable, "ml/simple_chapters.py", temp_path, str(n_chapters)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise RuntimeError(result.stderr)
            chapters = result.stdout.strip()
            info = ""
            method_used = "Simple pattern-based (fallback)"
        
        return {
            "chapters": chapters, 
            "lang": lang, 
            "method_used": method_used,
            "n_chapters": n_chapters,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
