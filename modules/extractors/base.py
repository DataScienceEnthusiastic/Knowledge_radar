"""
Base class for document extractors.
"""
from __future__ import annotations

from typing import Any, Optional
from modules.extractors.models import ExtractionOptions, ExtractionResult


class BaseExtractor:
    """Abstract base interface for modal document extractors."""

    def __init__(self, options: Optional[ExtractionOptions] = None):
        self.options = options or ExtractionOptions()

    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        raise NotImplementedError
