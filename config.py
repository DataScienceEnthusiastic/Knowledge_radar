"""
Central configuration for Knowledge Radar.

Defines model endpoints, default chunking parameters, and storage paths.
"""

import os

# Model & API Endpoints
LOCAL_MODEL_NAME = "qwen2.5-coder:3b"
CLOUD_MODEL_NAME = "gemini-2.0-flash"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Gemini Model Tiers
GEMINI_MODELS = {
    "low": "gemini-2.0-flash-lite",
    "medium": "gemini-2.0-flash",
    "high": "gemini-1.5-pro",
}

# Text Processing & Chunking Parameters
CHUNK_SIZE = 500       # Target characters per document chunk (Pareto elbow)
CHUNK_OVERLAP = 50     # Context overlap (10%)
DEFAULT_TOP_K = 5      # Number of matching vector chunks to retrieve

# Storage & Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMADB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_radar_v2"
