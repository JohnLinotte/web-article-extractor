"""Command-line interface for web_article_extractor.

Usage:
    python -m web_article_extractor <url> [--format json|markdown]

If <url> is a YouTube URL, its transcript is fetched. Otherwise the article
is extracted in the chosen format (default: markdown). The result is printed
to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from .web import extract_article
from .youtube import fetch_transcript, format_human, is_youtube_url


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run extraction or transcript fetch."""
    parser = argparse.ArgumentParser(
        prog="web_article_extractor",
        description=(
            "Extract a web article, or fetch a YouTube transcript when the URL "
            "points to YouTube."
        ),
    )
    parser.add_argument("url", help="Article URL or YouTube video URL.")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        dest="output_format",
        help="Output format (default: markdown).",
    )
    args = parser.parse_args(argv)

    if is_youtube_url(args.url):
        transcript = fetch_transcript(args.url)
        if transcript is None:
            print(
                f"ERROR: Could not fetch a transcript for {args.url}",
                file=sys.stderr,
            )
            return 1
        if args.output_format == "json":
            print(json.dumps(transcript, indent=2, ensure_ascii=False))
        else:
            print(
                format_human(
                    transcript["metadata"],
                    transcript["text"],
                    transcript["source"],
                    transcript["language"],
                )
            )
        return 0

    result = extract_article(args.url)
    if result is None:
        print(f"ERROR: Failed to extract content from {args.url}", file=sys.stderr)
        return 1

    if args.output_format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["content"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
