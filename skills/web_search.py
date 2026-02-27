"""
skills/web_search.py
Search the web via DuckDuckGo (no API key needed).
"""

DESCRIPTION = "Search the web via DuckDuckGo. Args: query (str), max_results (int, default 5)"

import requests
from bs4 import BeautifulSoup
import urllib.parse


def run(query: str, max_results: int = 5) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        results = []
        for block in soup.select(".result__body")[:max_results]:
            title_el = block.select_one(".result__title")
            snippet_el = block.select_one(".result__snippet")
            url_el = block.select_one(".result__url")

            title   = title_el.get_text(strip=True)   if title_el   else ""
            snippet = snippet_el.get_text(strip=True)  if snippet_el else ""
            link    = url_el.get_text(strip=True)      if url_el     else ""

            if snippet:
                results.append(f"**{title}**\n{snippet}\n{link}")

        if not results:
            return f"No results found for: {query}"

        return "\n\n---\n\n".join(results)

    except requests.Timeout:
        return f"Search timed out for: {query}"
    except Exception as e:
        return f"Search failed: {e}"
