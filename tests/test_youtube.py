"""Tests for the YouTube helpers.

The URL-detection test runs offline. The transcript test probes network and
skips gracefully when there is no connectivity (or no yt-dlp) — it never
hard-fails offline.
"""

from __future__ import annotations

import shutil

import pytest

from web_article_extractor import fetch_transcript, is_youtube_url
from web_article_extractor.youtube import YT_DLP_BIN

# Public-domain video with reliable captions.
YT_URL = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"


def test_is_youtube_url():
    """YouTube URL detection covers the common URL shapes."""
    assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_youtube_url("https://www.youtube.com/shorts/abc123_DEF")
    assert not is_youtube_url("https://example.com/article")
    assert not is_youtube_url("https://harnais.be/essais/t0/")


def _yt_dlp_available() -> bool:
    """Return True when a yt-dlp binary is reachable."""
    import os

    return os.path.isfile(YT_DLP_BIN) or shutil.which("yt-dlp") is not None


def _has_network() -> bool:
    """Probe basic connectivity to YouTube."""
    import urllib.request

    try:
        with urllib.request.urlopen("https://www.youtube.com", timeout=15) as resp:
            return resp.status == 200
    except (OSError, ValueError):
        return False


def test_fetch_transcript_structure():
    """When online with yt-dlp, a transcript fetch returns the expected shape."""
    if not _yt_dlp_available():
        pytest.skip("yt-dlp binary not available")
    if not _has_network():
        pytest.skip("No network access to YouTube")

    try:
        result = fetch_transcript(YT_URL)
    except (OSError, ValueError, RuntimeError) as exc:
        pytest.skip(f"transcript fetch failed (network/throttle): {exc}")

    if result is None:
        pytest.skip("No subtitles available or fetch throttled")

    for key in ("url", "metadata", "text", "source", "language"):
        assert key in result, f"missing field: {key}"
    assert isinstance(result["text"], str)
    assert result["text"].strip(), "transcript text is empty"
