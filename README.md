# web-article-extractor

A small, dependency-light toolkit for pulling readable content off the web. It
extracts article text from any URL using a two-stage strategy — `trafilatura`
first (fast, no browser), then a headless Playwright Chromium fallback for
JavaScript-heavy pages — and fetches transcripts from YouTube videos through a
manual-then-automatic subtitle cascade.

> **Naming.** The PyPI distribution is `harnais-web-extractor` (this repository
> keeps the name `web-article-extractor`). The Python import package is
> `web_article_extractor`.

## Install

From PyPI:

```bash
pip install harnais-web-extractor
# Playwright also needs a browser binary the first time:
playwright install chromium
# Optional YouTube transcript support:
pip install "harnais-web-extractor[youtube]"
```

Or from source (GitHub):

```bash
pip install git+https://github.com/JohnLinotte/web-article-extractor.git
```

## Usage

### Command line

```bash
# Extract an article as Markdown (default):
python -m web_article_extractor https://example.com/some-article

# As JSON:
python -m web_article_extractor https://example.com/some-article --format json

# A YouTube URL fetches the transcript instead:
python -m web_article_extractor "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

### Python API

```python
from web_article_extractor import extract_article, fetch_transcript, is_youtube_url

result = extract_article("https://example.com/some-article")
if result:
    print(result["title"])
    print(result["content"])      # Markdown
    print(result["word_count"])

if is_youtube_url(url):
    transcript = fetch_transcript(url)
    if transcript:
        print(transcript["text"])
```

`extract_article` returns a dict with `url`, `title`, `content`,
`source_method`, `extracted_at` and `word_count`, or `None` when extraction
fails entirely.

## License

MIT — see [LICENSE](LICENSE).
