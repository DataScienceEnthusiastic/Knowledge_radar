# Knowledge Radar 2024 Changelog & Release Notes
Release 2.4.0 (November 2024):
- Deprecated static non-normalized hash embeddings in favor of 256-D normalized vector representations.
- Added support for YouTube video transcript extraction with millisecond timestamps via youtube-transcript-api.
- Added Tesseract OCR pipeline for PNG, JPG, and TIFF image ingestion with Windows path auto-discovery.
- Integrated dual-stream context blending for contradictory evidence resolution.
- Set default top-k retrieval depth to 3 matching chunks per query.
- Bugfix: Resolved issue #104 where PDF extraction crashed on empty white-space pages in PyPDF 3.17.0.
