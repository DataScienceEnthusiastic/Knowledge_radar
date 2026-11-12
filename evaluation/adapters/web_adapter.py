"""
===============================================================================
📦 EVALUATION ADAPTER: Web Verification & Frozen Snapshot Playback
===============================================================================
Provides reproducible web evidence retrieval using frozen snapshots from
`evaluation/dataset/web_snapshots/snapshots.json` to guarantee 100% deterministic
benchmarking insulated from dynamic internet states.
===============================================================================
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from modules.web_search import fetch_live_web_context

SNAPSHOTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "web_snapshots", "snapshots.json"
)


class WebVerificationAdapter:
    """
    Adapter for Web Verification with support for both frozen reproducible
    snapshots and live external search.
    """
    def __init__(self, snapshots_file: str = SNAPSHOTS_PATH):
        self.snapshots_file = snapshots_file
        self.snapshots = {}
        if os.path.exists(snapshots_file):
            with open(snapshots_file, "r", encoding="utf-8") as f:
                self.snapshots = json.load(f)

    def search(
        self,
        query: str,
        mode: str = "snapshot",
        snapshot_id: Optional[str] = None,
        max_results: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieves web evidence according to evaluation mode.
        
        Args:
            query: User prompt.
            mode: "snapshot" (frozen deterministic evidence), "live" (DuckDuckGo), or "none".
            snapshot_id: Specific key in snapshots.json if present.
            max_results: Max snippets to return.
            
        Returns:
            Dict containing retrieved snippets, latency, and mode metadata.
        """
        if mode == "none":
            return {
                "snippets": [],
                "latency_ms": 0.0,
                "mode": "none",
                "source": "none"
            }

        t0 = time.perf_counter()

        if mode == "snapshot":
            # Deterministic snapshot lookup
            matched_snippets = []
            if snapshot_id and snapshot_id in self.snapshots:
                item = self.snapshots[snapshot_id]
                matched_snippets.append({
                    "title": item.get("title", "Web Evidence"),
                    "href": item.get("url", ""),
                    "body": item.get("snippet", ""),
                    "verified_date": item.get("verified_date", ""),
                    "is_conflict_source": item.get("is_conflict_source", False)
                })
            else:
                # Search snapshot corpus for token overlap
                q_words = set(query.lower().split())
                best_match = None
                best_overlap = 0

                for s_id, item in self.snapshots.items():
                    text = (item.get("title", "") + " " + item.get("snippet", "")).lower()
                    overlap = sum(1 for w in q_words if w in text)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = item

                if best_match and best_overlap > 1:
                    matched_snippets.append({
                        "title": best_match.get("title", "Web Evidence"),
                        "href": best_match.get("url", ""),
                        "body": best_match.get("snippet", ""),
                        "verified_date": best_match.get("verified_date", ""),
                        "is_conflict_source": best_match.get("is_conflict_source", False)
                    })

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "snippets": matched_snippets[:max_results],
                "latency_ms": round(elapsed_ms, 3),
                "mode": "snapshot",
                "source": "frozen_dataset"
            }

        elif mode == "live":
            # Live DuckDuckGo search fallback
            try:
                snippets = fetch_live_web_context(query, max_results=max_results)
            except Exception:
                snippets = []
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "snippets": snippets,
                "latency_ms": round(elapsed_ms, 3),
                "mode": "live",
                "source": "duckduckgo_live"
            }

        return {
            "snippets": [],
            "latency_ms": 0.0,
            "mode": mode,
            "source": "unknown"
        }
