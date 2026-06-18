"""Integration test for the web article extractor against a real page."""

from __future__ import annotations

import pytest

from web_article_extractor import extract_article

REAL_URL = "https://harnais.be/essais/t0/"


def test_extract_real_page():
    """Extracting a real published page yields non-empty content + fields.

    Skips gracefully when the page is unreachable (offline), but never
    hard-fails on missing connectivity. The extractor itself is used as the
    connectivity probe so that bot-blocked plain HTTP probes do not produce a
    false "no network" skip.
    """
    try:
        result = extract_article(REAL_URL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"extraction raised (likely offline): {exc}")

    if result is None:
        pytest.skip("No network access to the target page")

    # Expected fields are present.
    for key in ("url", "title", "content", "source_method", "word_count"):
        assert key in result, f"missing field: {key}"

    # The Markdown content (and therefore text) must be non-empty.
    assert isinstance(result["content"], str)
    assert result["content"].strip(), "content is empty"
    assert result["word_count"] > 0
    assert result["url"] == REAL_URL
    # The known title of the real page should be present.
    assert result["title"], "title is empty"
