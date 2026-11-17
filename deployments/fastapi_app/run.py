"""
Standalone launcher for Knowledge Radar FastAPI deployment.
Runs the local API server and serves the executive dashboard on http://127.0.0.1:8000.
"""
import os
import sys
import uvicorn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if __name__ == "__main__":
    print("📡 Launching Knowledge Radar FastAPI Console at http://127.0.0.1:8000 ...")
    uvicorn.run("deployments.fastapi_app.app:app", host="127.0.0.1", port=8000, reload=True)
