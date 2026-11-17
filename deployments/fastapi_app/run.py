"""
Standalone launcher for Knowledge Radar FastAPI deployment.
Runs the local API server and serves the executive dashboard on http://127.0.0.1:8000.
"""
import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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
    print(f"📡 Launching Knowledge Radar FastAPI Console at http://127.0.0.1:8000 using {python_bin} ...")
    cmd = [python_bin, "-m", "uvicorn", "deployments.fastapi_app.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    subprocess.run(cmd, cwd=ROOT_DIR)
