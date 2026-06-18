"""web_article_extractor: web article extraction and YouTube transcript fetch.

Public API:
    extract_article  -- two-stage web article extractor (trafilatura + Playwright).
    fetch_transcript -- YouTube transcript fetcher (manual -> auto subtitle cascade).
    is_youtube_url   -- helper to detect YouTube URLs.
"""

from __future__ import annotations

from .web import extract_article
from .youtube import fetch_transcript, is_youtube_url

__version__ = "0.1.0"

__all__ = [
    "extract_article",
    "fetch_transcript",
    "is_youtube_url",
    "__version__",
]
