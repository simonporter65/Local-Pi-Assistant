"""
skills/web_fetch.py
Fetch and extract text content from a URL.
"""

DESCRIPTION = "Fetch full text content from a URL. Args: url (str), max_chars (int, default 4000)"

import requests
from bs4 import BeautifulSoup
import re


def run(url: str, max_chars: int = 4000) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "")
        if "text/html" in content_type:
            soup = BeautifulSoup(r.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "advertisement", "noscript"]):
                tag.decompose()

            # Try to get main content
            main = (
                soup.find("main") or
                soup.find("article") or
                soup.find(id=re.compile(r"content|main|article", re.I)) or
                soup.find(class_=re.compile(r"content|main|article|post", re.I)) or
                soup.find("body") or
                soup
            )

            text = main.get_text(separator="\n", strip=True)
            # Collapse whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)

        else:
            text = r.text

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[...truncated, total {len(text)} chars]"

        return f"URL: {url}\n\n{text}"

    except requests.Timeout:
        return f"Timeout fetching: {url}"
    except requests.HTTPError as e:
        return f"HTTP error {e.response.status_code} fetching: {url}"
    except Exception as e:
        return f"Failed to fetch {url}: {e}"
