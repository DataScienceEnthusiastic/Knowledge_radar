"""
Live Web Search via DuckDuckGo.

Retrieves real-time context snippets for time-sensitive query verification.
"""

import logging

logger = logging.getLogger("web_search")

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None


def fetch_live_web_context(query: str, max_results: int = 2) -> list:
    """
    Executes a DuckDuckGo web search and returns top results.
    Returns an empty list on failure or when the library is unavailable.
    """
    if not query or not query.strip():
        return []

    if not DDGS:
        logger.warning("duckduckgo_search library is not installed; web search skipped.")
        return []

    try:
        results = []
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, max_results=max_results)
            if raw_results:
                for item in raw_results:
                    results.append({
                        "title": item.get("title", "No Title"),
                        "snippet": item.get("body", item.get("snippet", "")),
                        "url": item.get("href", item.get("link", ""))
                    })
        return results
    except Exception as e:
        logger.warning("Live web search failed for query %r: %s", query, e)
        return []
