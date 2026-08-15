"""
===============================================================================
📄 MODULE: Document & Multi-Modal Parsers
===============================================================================
Provides modular extractors for:
- PDF files (pypdf)
- YouTube video transcripts with timestamps (youtube-transcript-api)
- Web URLs (requests + BeautifulSoup4)
- Images (pytesseract OCR)
- Text chunking utility with configurable size and overlap.
===============================================================================
"""

import re
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from PIL import Image

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from config import CHUNK_SIZE, CHUNK_OVERLAP


def parse_pdf(pdf_file) -> str:
    """
    Extracts text from an uploaded PDF file or file path using PyPDF.
    
    Args:
        pdf_file: Uploaded file-like object or filepath.
        
    Returns:
        str: Extracted text content.
    """
    try:
        reader = PdfReader(pdf_file)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                cleaned = re.sub(r'\s+', ' ', text).strip()
                extracted_pages.append(f"[Page {i+1}] {cleaned}")
        return "\n\n".join(extracted_pages)
    except Exception as e:
        return f"Error parsing PDF: {str(e)}"


def extract_youtube_video_id(url: str) -> str:
    """
    Extracts YouTube video ID from standard, shortened, or embedded URLs.
    """
    pattern = r'(?:v=|\/([0-9A-Za-z_-]{11}).*|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})'
    match = re.search(pattern, url)
    if match:
        return match.group(1) or match.group(2)
    return ""


def parse_youtube(url: str) -> str:
    """
    Fetches transcript for a YouTube video URL and formats it with [MM:SS] timestamps.
    Supports both class-level and instance-level YouTubeTranscriptApi methods across API versions.
    """
    if not YouTubeTranscriptApi:
        return "Error: 'youtube-transcript-api' is not installed."
        
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return "Error: Invalid YouTube URL format. Could not extract Video ID."
        
    try:
        transcript_list = []
        
        # Try instance method fetch() first (v0.6.2+)
        try:
            api = YouTubeTranscriptApi()
            if hasattr(api, "fetch"):
                transcript_list = api.fetch(video_id)
            elif hasattr(api, "get_transcript"):
                transcript_list = api.get_transcript(video_id)
        except Exception:
            pass

        # Fall back to class-level methods
        if not transcript_list:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            elif hasattr(YouTubeTranscriptApi, "fetch"):
                transcript_list = YouTubeTranscriptApi.fetch(video_id)
                
        formatted_lines = []
        
        for item in transcript_list:
            if isinstance(item, dict):
                start_seconds = int(item.get("start", 0))
                text = item.get("text", "").strip()
            else:
                start_seconds = int(getattr(item, "start", 0))
                text = getattr(item, "text", "").strip()

            minutes = start_seconds // 60
            seconds = start_seconds % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            if text:
                formatted_lines.append(f"{timestamp} {text}")
                
        if not formatted_lines:
            return "No transcript text found for this video (Subtitles may be disabled)."

        return "\n".join(formatted_lines)
    except Exception as e:
        return f"Error fetching YouTube transcript: {str(e)} (Subtitles may be disabled for this video)."


def parse_url(url: str) -> str:
    """
    Extracts clean article/body text from a Web URL using Requests and BeautifulSoup.
    
    Args:
        url (str): Web page URL.
        
    Returns:
        str: Clean text extracted from HTML.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
            element.decompose()
            
        # Extract body text
        body = soup.body
        if not body:
            return soup.get_text(separator=" ", strip=True)
            
        text = body.get_text(separator="\n", strip=True)
        # Collapse multiple newlines/spaces
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(cleaned_lines)
    except Exception as e:
        return f"Error fetching web URL: {str(e)}"


def parse_image(image_file) -> str:
    """
    Extracts text from an uploaded image using PIL and Tesseract OCR.
    Autodetects common Windows Tesseract installation paths.
    """
    if not pytesseract:
        return "Error: 'pytesseract' module is not installed."
        
    import os
    possible_cmd_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    for p in possible_cmd_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break

    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        cleaned = text.strip()
        if not cleaned:
            return "No readable text detected in the image."
        return cleaned
    except Exception as e:
        err_msg = str(e)
        if "tesseract is not installed" in err_msg.lower() or "not found" in err_msg.lower():
            return "Error: Tesseract OCR binary (tesseract.exe) not found on system path. Please install the Windows Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki."
        return f"Error processing image OCR: {err_msg}"


def chunk_text(text: str, source_name: str = "Document", chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """
    Splits text into overlapping chunks for vector embedding.
    
    Args:
        text (str): Full raw text content.
        source_name (str): Label indicating source (e.g. PDF filename, YouTube, Web URL).
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Overlapping characters between consecutive chunks.
        
    Returns:
        list of dicts: List of chunk items containing 'text', 'source', and 'chunk_index'.
    """
    if not text or not text.strip():
        return []
        
    text_len = len(text)
    chunks = []
    start = 0
    chunk_idx = 1
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_content = text[start:end].strip()
        
        if chunk_content:
            chunks.append({
                "text": chunk_content,
                "source": source_name,
                "chunk_index": chunk_idx
            })
            chunk_idx += 1
            
        start += (chunk_size - chunk_overlap)
        if start >= text_len:
            break
            
    return chunks
