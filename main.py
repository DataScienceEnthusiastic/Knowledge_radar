"""
Main application entry point for Knowledge Radar.
Exposes the FastAPI instance for standard ASGI servers and Vercel zero-config.
"""
from app import app

__all__ = ["app"]
