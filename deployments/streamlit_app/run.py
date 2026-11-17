"""
Standalone launcher for Knowledge Radar Streamlit deployment.
Runs the Streamlit Cloud-compatible application locally on http://localhost:8501.
"""
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

if __name__ == "__main__":
    print(f"📡 Launching Knowledge Radar Streamlit App on http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", APP_PATH, "--server.port", "8501"]
    subprocess.run(cmd, cwd=ROOT_DIR)
