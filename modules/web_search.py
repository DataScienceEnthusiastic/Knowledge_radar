"""
===============================================================================
🌐 MODULE: Live Web Search (DuckDuckGo)
===============================================================================
Performs real-time web searches using DuckDuckGo (DDGS) to retrieve live, up-to-date
context snippet verification for user queries.
===============================================================================
"""

from duckduckgo_search import DDGS


def fetch_live_web_context(query: str, max_results: int = 2) -> list:
    """
    Executes a DuckDuckGo web search and returns top results.
    
    Args:
        query (str): Search prompt.
        max_results (int): Number of top search results to retrieve (default: 2).
        
    Returns:
        list of dicts: List containing title, snippet, and url for each result.
    """
    if not query or not query.strip():
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
        # Gracefully handle network/API errors without throwing
        return [{
            "title": "Web Search Warning",
            "snippet": f"Could not perform live web search: {str(e)}",
            "url": "#"
        }]
