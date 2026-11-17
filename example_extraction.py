"""
Demonstration script for multimodal document extraction and downstream storage.
Forwarding wrapper to modules.extractors.example_extraction.
"""
import os
import sys

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.extractors.example_extraction import main

if __name__ == "__main__":
    main()
