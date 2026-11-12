"""
YouTube video transcript extraction engine.
Parses video IDs, fetches official or auto-generated subtitles, and metadata.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    HAS_YOUTUBE = True
except ImportError:
    YouTubeTranscriptApi = None
    HAS_YOUTUBE = False

from modules.extractors.base import BaseExtractor
from modules.extractors.models import (
    ContentBlock,
    DocumentMetadata,
    ExtractionError,
    ExtractionOptions,
    ExtractionResult,
    ExtractionStatistics,
    ExtractionWarning,
    TranscriptSegment,
)
from modules.extractors.utils import generate_block_id, generate_document_id, normalize_text

logger = logging.getLogger("document_extractor.youtube")

class YouTubeExtractor(BaseExtractor):
    """
    Extracts transcripts with timestamps and public metadata from YouTube URLs.
    """

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract_video_id(self, url: str) -> Optional[str]:
        patterns = [
            r"(?:v=|\/vi\/|youtu\.be\/|\/embed\/|\/shorts\/)([0-9A-Za-z_-]{11})",
            r"^([0-9A-Za-z_-]{11})$",
        ]
        for pat in patterns:
            m = re.search(pat, url)
            if m:
                return m.group(1)
        return None

    def extract(self, url: str) -> ExtractionResult:
        start_time = time.time()
        video_id = self.extract_video_id(url)
        if not video_id:
            res = ExtractionResult(
                document_id=generate_document_id("youtube", url),
                source_id=url,
                source_type="youtube",
                original_source=url,
                canonical_source=url,
            )
            res.errors.append(ExtractionError(code="INVALID_URL", message=f"No video ID found in '{url}'."))
            res.extraction_status = "failed"
            return res

        canonical = f"https://www.youtube.com/watch?v={video_id}"
        doc_id = generate_document_id("youtube", canonical)

        metadata = DocumentMetadata(
            source_url=url,
            canonical_url=canonical,
            mime_type="video/youtube",
        )

        result = ExtractionResult(
            document_id=doc_id,
            source_id=canonical,
            source_type="youtube",
            original_source=url,
            canonical_source=canonical,
            document_metadata=metadata,
        )

        # 1. Title & author via public oEmbed
        if HAS_REQUESTS:
            try:
                resp = requests.get(f"https://www.youtube.com/oembed?url={canonical}&format=json", timeout=5.0)
                if resp.status_code == 200:
                    d = resp.json()
                    result.title = d.get("title")
                    result.document_metadata.title = result.title
                    result.document_metadata.author = d.get("author_name")
                    result.document_metadata.channel_name = d.get("author_name")
            except Exception:
                pass

        # 2. Transcript fetching
        raw_segments = []
        source_label = "captions"

        if YouTubeTranscriptApi is not None:
            try:
                api = YouTubeTranscriptApi() if callable(YouTubeTranscriptApi) else YouTubeTranscriptApi
                fetched = None
                if hasattr(api, "fetch"):
                    fetched = api.fetch(video_id)
                    raw_segments = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
                elif hasattr(api, "get_transcript"):
                    raw_segments = api.get_transcript(video_id)

                if not raw_segments and hasattr(YouTubeTranscriptApi, "get_transcript"):
                    raw_segments = YouTubeTranscriptApi.get_transcript(video_id)
            except Exception as e:
                result.extraction_warnings.append(ExtractionWarning(
                    code="TRANSCRIPT_FAILED", message=f"Subtitles retrieval error: {e}"
                ))

        # 3. Audio transcription fallback (optional)
        if not raw_segments and self.options.enable_audio_transcription_fallback and self.options.audio_transcription_adapter:
            try:
                raw_segments = self.options.audio_transcription_adapter.transcribe_audio(canonical)
                source_label = "speech_recognition"
            except Exception as e:
                result.extraction_warnings.append(ExtractionWarning(
                    code="FALLBACK_FAILED", message=f"Audio transcription fallback failed: {e}"
                ))

        # 4. Assemble segments
        order = 1
        for item in raw_segments:
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            s_sec = float(item.get("start", 0.0))
            dur = float(item.get("duration", 0.0))
            e_sec = s_sec + dur

            seg_id = f"seg_{doc_id[:8]}_{order:04d}"
            result.transcript_segments.append(TranscriptSegment(
                segment_id=seg_id,
                document_id=doc_id,
                text=txt,
                start_time_seconds=round(s_sec, 2),
                end_time_seconds=round(e_sec, 2),
                duration_seconds=round(dur, 2),
                transcript_source=source_label,
            ))

            m, s = int(s_sec) // 60, int(s_sec) % 60
            result.content_blocks.append(ContentBlock(
                block_id=seg_id,
                document_id=doc_id,
                element_type="transcript",
                text=f"[{m:02d}:{s:02d}] {txt}",
                normalized_text=normalize_text(txt),
                reading_order=order,
                extraction_method=f"youtube_{source_label}",
                metadata={"start_time_seconds": s_sec, "end_time_seconds": e_sec},
            ))
            order += 1

        result.extraction_statistics = ExtractionStatistics(
            page_count=0,
            content_block_count=len(result.content_blocks),
            transcript_segment_count=len(result.transcript_segments),
            extraction_duration_seconds=round(time.time() - start_time, 3),
            warning_count=len(result.extraction_warnings),
            error_count=len(result.errors),
        )

        if not result.transcript_segments:
            result.extraction_warnings.append(ExtractionWarning(
                code="NO_TRANSCRIPT_AVAILABLE", message="No subtitles or transcript found for this video."
            ))
            result.extraction_status = "partial_success"
        else:
            result.extraction_status = "success"

        return result
