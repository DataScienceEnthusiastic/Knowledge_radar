"""
Standalone launcher for Knowledge Radar Streamlit deployment.
Runs the Streamlit Cloud-compatible application locally on http://localhost:8501.
"""
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

def resolve_python_executable() -> str:
    """Find virtualenv python binary if available, otherwise fall back to sys.executable."""
    candidates = [
        os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe"),
        os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe"),
        os.path.join(ROOT_DIR, ".venv", "bin", "python"),
        os.path.join(ROOT_DIR, "venv", "bin", "python"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return sys.executable

if __name__ == "__main__":
    python_bin = resolve_python_executable()
    print(f"📡 Launching Knowledge Radar Streamlit App on http://localhost:8501 using {python_bin} ...")
    cmd = [python_bin, "-m", "streamlit", "run", APP_PATH, "--server.port", "8501"]
    subprocess.run(cmd, cwd=ROOT_DIR)
