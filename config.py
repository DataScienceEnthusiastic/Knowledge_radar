"""
===============================================================================
📡 KNOWLEDGE RADAR — Central Configuration
===============================================================================
Contains global configuration settings, model names, API default URLs,
and document chunking parameters for the Knowledge Radar application.
===============================================================================
"""

import os

# --- Model & API Settings ---
LOCAL_MODEL_NAME = "qwen2.5-coder:3b"
CLOUD_MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# --- Text Processing Parameters ---
CHUNK_SIZE = 500       # Target characters per document chunk
CHUNK_OVERLAP = 50     # Context overlap between consecutive chunks
DEFAULT_TOP_K = 3      # Number of matching vector chunks to retrieve

# --- Storage Settings ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMADB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "knowledge_radar_v2"
