#!/usr/bin/env python3
"""Web article extraction.

Two-stage approach: trafilatura first (fast, no browser), Playwright
headless Chromium fallback for JS-heavy pages.

The public entry point is :func:`extract_article`, which returns a dict with
the keys: ``url``, ``title``, ``content``, ``source_method``,
``extracted_at`` and ``word_count``.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Extraction functions (lazy imports to avoid startup cost)
# ---------------------------------------------------------------------------


def _extract_title_from_html(html: str) -> str:
    """Extract <title> tag content from raw HTML via regex."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_with_trafilatura(
    url: str, timeout: float = 15.0
) -> tuple[str | None, str | None, str]:
    """Extract article content using trafilatura.

    Returns:
        (content, title, source_method) -- content/title may be None on failure.
    """
    try:
        import trafilatura
    except ImportError:
        print("ERROR: trafilatura not installed. pip install trafilatura", file=sys.stderr)
        return None, None, "trafilatura"

    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None, "trafilatura"

        content = trafilatura.extract(
            downloaded,
            output_format="markdown",
            with_metadata=True,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
            url=url,
        )

        # Extract title from HTML
        title = _extract_title_from_html(downloaded)

        return content, title, "trafilatura"

    except (OSError, ValueError, RuntimeError) as e:
        print(f"WARNING: trafilatura extraction failed: {e}", file=sys.stderr)
        return None, None, "trafilatura"


def extract_with_playwright(
    url: str, timeout: float = 10.0
) -> tuple[str | None, str | None, str]:
    """Extract article content using Playwright headless Chromium + trafilatura.

    Launches a headless browser, renders JS, then extracts from rendered HTML.

    Returns:
        (content, title, source_method) -- content/title may be None on failure.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "ERROR: playwright not installed. pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return None, None, "playwright+trafilatura"

    try:
        import trafilatura
    except ImportError:
        print("ERROR: trafilatura not installed. pip install trafilatura", file=sys.stderr)
        return None, None, "playwright+trafilatura"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=int(timeout * 1000),
            )
            # Short post-wait for JS to populate DOM
            page.wait_for_timeout(1000)
            title = page.title()
            html = page.content()
            browser.close()

        content = trafilatura.extract(
            html,
            output_format="markdown",
            with_metadata=True,
            include_tables=True,
            favor_precision=True,
            deduplicate=True,
            url=url,
        )

        return content, title, "playwright+trafilatura"

    except (OSError, ValueError, RuntimeError) as e:
        print(f"WARNING: Playwright extraction failed: {e}", file=sys.stderr)
        return None, None, "playwright+trafilatura"


def extract_article(
    url: str,
    force_playwright: bool = False,
    timeout: float = 15.0,
) -> dict | None:
    """Two-stage article extraction: trafilatura first, Playwright fallback.

    Args:
        url: The article URL to extract.
        force_playwright: Skip trafilatura, go straight to Playwright.
        timeout: Per-stage timeout in seconds.

    Returns:
        Dict with url, title, content, source_method, extracted_at, word_count.
        None if extraction failed entirely.
    """
    content = None
    title = None
    method = "unknown"

    # Stage 1: trafilatura (unless forced to Playwright)
    if not force_playwright:
        content, title, method = extract_with_trafilatura(url, timeout=timeout)

        # Check if result is sufficient (>100 chars = meaningful content)
        if content and len(content) > 100:
            return _build_result(url, title, content, method)

    # Stage 2: Playwright fallback (or forced)
    pw_content, pw_title, pw_method = extract_with_playwright(url, timeout=timeout)

    if pw_content and len(pw_content) > 50:
        # Prefer Playwright title if trafilatura didn't provide one
        final_title = pw_title or title or ""
        return _build_result(url, final_title, pw_content, pw_method)

    # If Playwright also failed but trafilatura got *something*, use it
    if content:
        return _build_result(url, title, content, method)

    return None


def _build_result(
    url: str, title: str | None, content: str, method: str
) -> dict:
    """Build the standard result dict."""
    return {
        "url": url,
        "title": title or "",
        "content": content,
        "source_method": method,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "word_count": len(content.split()),
    }
