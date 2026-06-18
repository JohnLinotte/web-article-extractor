#!/usr/bin/env python3
"""YouTube audio download and transcript extraction via yt-dlp.

Downloads audio and/or subtitles from YouTube URLs using the yt-dlp binary,
and provides a transcript cascade (manual subtitles -> auto-generated
subtitles -> optional faster-whisper fallback). Zero LLM is used for the
subtitle path.

The yt-dlp binary path can be overridden with the ``YT_DLP_BIN`` environment
variable; it defaults to ``~/.local/bin/yt-dlp``.

Download strategy:
  1. Try to fetch existing subtitles (manual or auto-generated) via yt-dlp.
  2. If no subtitles and audio is needed: download the best audio stream.
  3. Return metadata (title, channel, duration, description) in all cases.

Transcript modes:
  - transcript : human-readable transcript.
  - analyze    : JSON with prompt + data (for downstream LLM analysis).
  - whisper    : download audio + transcribe with faster-whisper
                 (fallback when subtitles are unavailable).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger("web_article_extractor.youtube")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def _resolve_ytdlp_bin() -> str:
    """Resolve the yt-dlp binary path.

    Priority: $YT_DLP_BIN env var, then `yt-dlp` found on $PATH (shutil.which),
    then the historical default ~/.local/bin/yt-dlp as a last resort. The
    $PATH lookup is what makes `pip install "harnais-web-extractor[youtube]"`
    work out-of-the-box in a venv (the binary lands in <venv>/bin/yt-dlp).
    """
    env = os.getenv("YT_DLP_BIN")
    if env:
        return env
    on_path = shutil.which("yt-dlp")
    if on_path:
        return on_path
    return str(Path.home() / ".local" / "bin" / "yt-dlp")


YT_DLP_BIN = _resolve_ytdlp_bin()

# YouTube URL patterns
_YT_URL_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"(?:https?://)?youtu\.be/[\w-]+"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/live/[\w-]+"),
    re.compile(r"(?:https?://)?music\.youtube\.com/watch\?v=[\w-]+"),
]

# Preferred subtitle languages (order matters -- first match wins)
_PREFERRED_LANGS = ["fr", "en", "nl", "de"]


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a valid YouTube URL."""
    return any(pattern.search(url) for pattern in _YT_URL_PATTERNS)


def _yt_dlp_extra_args() -> list[str]:
    """Optional yt-dlp arguments from the environment, empty by default.

    - YT_DLP_COOKIES_FROM_BROWSER: e.g. "firefox" → adds --cookies-from-browser firefox
    - YT_DLP_JS_RUNTIME: e.g. "node" → adds --js-runtime node

    Both default to unset (no cookies, standard JS runtime) so the package
    works on any machine. Set them to match your local setup if you need
    cookies or a specific JS runtime.
    """
    extra: list[str] = []
    browser = os.getenv("YT_DLP_COOKIES_FROM_BROWSER")
    if browser:
        extra += ["--cookies-from-browser", browser]
    js = os.getenv("YT_DLP_JS_RUNTIME")
    if js:
        extra += ["--js-runtime", js]
    return extra


def _run_ytdlp(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run yt-dlp with the given arguments.

    Args:
        args: Arguments to pass to yt-dlp (after the binary path).
        timeout: Maximum execution time in seconds.

    Returns:
        CompletedProcess result.

    Raises:
        FileNotFoundError: If yt-dlp binary is not found.
        subprocess.TimeoutExpired: If the command times out.
    """
    cmd = [YT_DLP_BIN, *_yt_dlp_extra_args(), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def fetch_metadata(url: str) -> dict[str, Any]:
    """Fetch video metadata without downloading.

    Uses yt-dlp's JSON dump for reliable parsing (avoids issues with
    multiline descriptions breaking field boundaries).

    Args:
        url: YouTube video URL.

    Returns:
        Dict with title, channel, duration_seconds, description,
        upload_date, view_count, and thumbnail URL.
    """
    try:
        result = _run_ytdlp([
            "--skip-download",
            "--dump-json",
            "--no-warnings",
            url,
        ], timeout=30)
    except subprocess.TimeoutExpired:
        # yt-dlp throttle: the metadata call ran past its internal bound. Return
        # a structured error (NOT a raised traceback) so the caller exits cleanly
        # and can retry instead of crashing.
        return {"error": "yt-dlp metadata fetch timed out (throttle); retryable"}

    if result.returncode != 0:
        return {"error": f"yt-dlp metadata fetch failed: {result.stderr[:500]}"}

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": "Failed to parse yt-dlp JSON output"}

    description = info.get("description", "") or ""
    # Truncate long descriptions
    if len(description) > 500:
        description = description[:500] + "..."

    duration = info.get("duration")
    try:
        duration = int(float(duration)) if duration else 0
    except (ValueError, TypeError):
        duration = 0

    view_count = info.get("view_count")
    try:
        view_count = int(view_count) if view_count is not None else None
    except (ValueError, TypeError):
        view_count = None

    # Extract available subtitle languages for downstream use
    manual_subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    return {
        "title": info.get("title", ""),
        "channel": info.get("channel", "") or info.get("uploader", ""),
        "duration_seconds": duration,
        "description": description,
        "upload_date": info.get("upload_date", ""),
        "view_count": view_count,
        "thumbnail": info.get("thumbnail", ""),
        "subtitle_langs": {
            "manual": list(manual_subs.keys()),
            "auto": list(auto_subs.keys()),
        },
    }


def fetch_subtitles(
    url: str,
    output_dir: str,
    preferred_langs: Optional[list[str]] = None,
    subtitle_langs: Optional[dict[str, list[str]]] = None,
) -> Optional[dict[str, Any]]:
    """Try to download existing subtitles (manual first, then auto-generated).

    Uses a two-step approach to avoid multi-language failures:
    1. Check available subtitles (from subtitle_langs or via --dump-json)
    2. Download the best match with one targeted yt-dlp call

    Args:
        url: YouTube video URL.
        output_dir: Directory to save subtitle files.
        preferred_langs: Ordered list of preferred language codes.
        subtitle_langs: Pre-fetched subtitle availability from fetch_metadata().
            Dict with "manual" and "auto" keys, each a list of language codes.
            If None, fetches availability via --dump-json (extra request).

    Returns:
        Dict with subtitle_path, language, source ("manual" or "auto"),
        and text content. Returns None if no subtitles available.
    """
    langs = preferred_langs or _PREFERRED_LANGS

    # Step 1: Determine what's available
    if subtitle_langs is None:
        try:
            result = _run_ytdlp([
                "--skip-download", "--dump-json", "--no-warnings", url,
            ], timeout=30)
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        try:
            info = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        manual_avail = set((info.get("subtitles") or {}).keys())
        auto_avail = set((info.get("automatic_captions") or {}).keys())
    else:
        manual_avail = set(subtitle_langs.get("manual", []))
        auto_avail = set(subtitle_langs.get("auto", []))

    # Build ordered candidate list: (lang, source, yt-dlp flag)
    # Prefer manual over auto for each language, respect lang priority order.
    candidates = []
    for lang in langs:
        if lang in manual_avail:
            candidates.append((lang, "manual", "--write-subs"))
        if lang in auto_avail:
            candidates.append((lang, "auto", "--write-auto-subs"))

    if not candidates:
        return None

    # Step 2: Try each candidate until one succeeds (handles transient 429s)
    for lang, source, flag in candidates:
        try:
            _run_ytdlp([
                "--skip-download",
                flag,
                "--sub-langs", lang,
                "--sub-format", "srt",
                "--output", os.path.join(output_dir, "%(id)s"),
                "--no-warnings",
                url,
            ], timeout=30)
        except subprocess.TimeoutExpired:
            # This candidate's download was throttled past its bound. Don't
            # crash the whole call -- try the next candidate (different lang /
            # source). If every candidate times out we return None below.
            continue

        sub_file = _find_subtitle_file(output_dir)
        if sub_file:
            return _read_subtitle_result(sub_file, source=source)

    return None


def _find_subtitle_file(directory: str) -> Optional[str]:
    """Find subtitle files in a directory (.srt, .vtt)."""
    for ext in (".srt", ".vtt"):
        for f in Path(directory).glob(f"*{ext}"):
            if f.stat().st_size > 0:
                return str(f)
    return None


def _read_subtitle_result(sub_path: str, source: str) -> dict[str, Any]:
    """Read a subtitle file and return structured result."""
    text = Path(sub_path).read_text(encoding="utf-8", errors="replace")

    # Extract language from filename (e.g., "dQw4w9WgXcQ.fr.srt")
    stem = Path(sub_path).stem
    parts = stem.split(".")
    language = parts[-1] if len(parts) > 1 else "unknown"

    # Clean SRT to plain text (strip timestamps and indices)
    plain_lines = []
    for line in text.splitlines():
        line = line.strip()
        # Skip index lines (pure numbers)
        if re.match(r"^\d+$", line):
            continue
        # Skip timestamp lines
        if re.match(r"^\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->", line):
            continue
        if line:
            plain_lines.append(line)

    plain_text = " ".join(plain_lines)

    return {
        "subtitle_path": sub_path,
        "language": language,
        "source": source,
        "text": plain_text,
        "raw_srt": text,
    }


def download_audio(
    url: str,
    output_dir: str,
) -> dict[str, Any]:
    """Download the best audio stream from a YouTube video.

    Prefers AAC (mp4a) streams; falls back to best available audio.
    faster-whisper supports m4a/webm natively -- no ffmpeg needed.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the audio file.

    Returns:
        Dict with audio_path, format, and file_size_mb.
    """
    result = _run_ytdlp([
        "-f", "bestaudio[acodec=mp4a]/bestaudio",
        "--output", os.path.join(output_dir, "%(id)s.%(ext)s"),
        "--no-playlist",
        "--no-warnings",
        url,
    ], timeout=600)

    if result.returncode != 0:
        return {"error": f"yt-dlp audio download failed: {result.stderr[:500]}"}

    # Find the downloaded audio file
    audio_file = None
    for ext in (".webm", ".m4a", ".mp4", ".ogg", ".opus", ".mp3"):
        for f in Path(output_dir).glob(f"*{ext}"):
            if f.stat().st_size > 0 and not f.name.endswith(".srt"):
                audio_file = str(f)
                break
        if audio_file:
            break

    if not audio_file:
        return {"error": "Audio file not found after download"}

    size_mb = Path(audio_file).stat().st_size / (1024 * 1024)

    return {
        "audio_path": audio_file,
        "format": Path(audio_file).suffix.lstrip("."),
        "file_size_mb": round(size_mb, 2),
    }


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------


def _transcribe_with_whisper(
    audio_path: str,
    model_size: str = "medium",
    language: str | None = None,
) -> dict:
    """Transcribe an audio file using faster-whisper.

    Args:
        audio_path: Path to the audio file (webm, m4a, wav, etc.).
        model_size: Whisper model size (tiny, base, small, medium, large-v3).
            Default 'medium' -- good balance of speed/accuracy for French.
        language: ISO language code for transcription. None = auto-detect.

    Returns:
        Dict with 'text' (full transcript), 'language' (detected/forced),
        'segments' (list of {start, end, text}), and 'model' (model used).

    Raises:
        ImportError: If faster-whisper is not installed.
        RuntimeError: If transcription fails.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. Install with: "
            "pip install faster-whisper"
        ) from exc

    logger.info("Loading whisper model '%s' for transcription...", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    logger.info(
        "Transcribing %s (language=%s)...",
        audio_path, language or "auto-detect",
    )
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    segments = []
    full_text_parts = []
    for segment in segments_iter:
        segments.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
        })
        full_text_parts.append(segment.text.strip())

    detected_language = info.language if info else (language or "unknown")
    language_probability = info.language_probability if info else 0.0

    logger.info(
        "Transcription complete: %d segments, language=%s (%.0f%%), %d chars",
        len(segments), detected_language, language_probability * 100,
        len(" ".join(full_text_parts)),
    )

    return {
        "text": " ".join(full_text_parts),
        "language": detected_language,
        "language_probability": language_probability,
        "segments": segments,
        "model": model_size,
    }


def _format_duration(seconds: int) -> str:
    """Format seconds as Xh YYm ZZs or Ym ZZs."""
    if seconds <= 0:
        return "unknown"
    hours, remainder = divmod(seconds, 3600)
    mins, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    return f"{mins}m{secs:02d}s"


def format_human(
    metadata: dict, transcript_text: str, source: str, language: str
) -> str:
    """Format transcript for human-readable display."""
    title = metadata.get("title", "Unknown")
    channel = metadata.get("channel", "Unknown")
    duration = _format_duration(metadata.get("duration_seconds", 0))
    description = metadata.get("description", "")

    lines = [
        f"# {title}",
        f"**Channel:** {channel}",
        f"**Duration:** {duration}",
        f"**Source:** {source} subtitles ({language})",
    ]
    if description:
        lines.append(f"**Description:** {description}")
    lines.extend(["", "---", "", transcript_text])

    return "\n".join(lines)


def format_llm_context(
    metadata: dict,
    transcript_text: str,
    source: str,
    language: str,
    url: str,
) -> dict:
    """Format transcript as JSON for downstream LLM analysis."""
    return {
        "prompt": (
            "You are an expert video content analyst. Summarize and analyze the "
            "YouTube transcript below. Produce:\n"
            "1. **Summary** (5-10 sentences, the essential points)\n"
            "2. **Key points** (a structured list of the main arguments/ideas)\n"
            "3. **Critical analysis** (relevance, quality of arguments, "
            "what to remember, possible limitations)\n\n"
            "Include the video metadata as a header."
        ),
        "data": {
            "url": url,
            "title": metadata.get("title", ""),
            "channel": metadata.get("channel", ""),
            "duration_seconds": metadata.get("duration_seconds", 0),
            "duration_formatted": _format_duration(
                metadata.get("duration_seconds", 0)
            ),
            "description": metadata.get("description", ""),
            "upload_date": metadata.get("upload_date", ""),
            "transcript_source": f"{source} ({language})",
            "transcript": transcript_text,
        },
    }


def format_whisper_context(
    metadata: dict,
    whisper_result: dict,
    url: str,
) -> dict:
    """Format whisper transcription as JSON for downstream LLM analysis.

    Same structure as format_llm_context but with whisper-specific metadata.
    """
    detected_lang = whisper_result.get("language", "unknown")
    model = whisper_result.get("model", "medium")
    transcript_text = whisper_result.get("text", "")

    return {
        "prompt": (
            "You are an expert video content analyst. Summarize and analyze the "
            "YouTube transcript below. Produce:\n"
            "1. **Summary** (5-10 sentences, the essential points)\n"
            "2. **Key points** (a structured list of the main arguments/ideas)\n"
            "3. **Critical analysis** (relevance, quality of arguments, "
            "what to remember, possible limitations)\n\n"
            "Include the video metadata as a header."
        ),
        "data": {
            "url": url,
            "title": metadata.get("title", ""),
            "channel": metadata.get("channel", ""),
            "duration_seconds": metadata.get("duration_seconds", 0),
            "duration_formatted": _format_duration(
                metadata.get("duration_seconds", 0)
            ),
            "description": metadata.get("description", ""),
            "upload_date": metadata.get("upload_date", ""),
            "transcript_source": f"whisper ({detected_lang}, model={model})",
            "transcript": transcript_text,
        },
    }


def fetch_transcript(
    url: str,
    preferred_langs: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """Fetch a YouTube transcript via the subtitle cascade.

    Tries manual subtitles first, then auto-generated subtitles, using the
    yt-dlp subtitle cascade. No audio download or whisper fallback is
    performed here.

    Args:
        url: YouTube video URL.
        preferred_langs: Ordered list of preferred language codes.

    Returns:
        Dict with metadata, text, source ("manual"/"auto") and language, or
        None when no subtitles are available.
    """
    if not is_youtube_url(url):
        return None

    metadata = fetch_metadata(url)
    if "error" in metadata:
        return None

    with tempfile.TemporaryDirectory(prefix="wae_ytsub_") as tmpdir:
        subs = fetch_subtitles(
            url, tmpdir,
            preferred_langs=preferred_langs,
            subtitle_langs=metadata.get("subtitle_langs"),
        )

    if not subs:
        return None

    return {
        "url": url,
        "metadata": metadata,
        "text": subs["text"],
        "source": subs["source"],
        "language": subs["language"],
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _run_download_cli(args) -> int:
    """CLI for the audio/subtitle downloader."""
    # Validate URL
    if not is_youtube_url(args.url):
        result = {"success": False, "error": f"Not a valid YouTube URL: {args.url}"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    # Validate yt-dlp binary
    if not os.path.isfile(YT_DLP_BIN):
        result = {"success": False, "error": f"yt-dlp not found at {YT_DLP_BIN}"}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="wae_yt_")

    try:
        # Fetch metadata
        metadata = fetch_metadata(args.url)
        if "error" in metadata:
            result = {"success": False, "error": metadata["error"]}
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1

        data: dict[str, Any] = {
            "url": args.url,
            "metadata": metadata,
            "output_dir": output_dir,
        }

        # Subtitle languages
        preferred_langs = (
            args.subs_lang.split(",") if args.subs_lang
            else _PREFERRED_LANGS
        )

        # Fetch subtitles (unless audio-only)
        if not args.audio_only:
            subs_result = fetch_subtitles(
                args.url, output_dir, preferred_langs,
                subtitle_langs=metadata.get("subtitle_langs"),
            )
            if subs_result:
                data["subtitles"] = subs_result
            else:
                data["subtitles"] = None

        # Download audio (unless subs-only)
        if not args.subs_only:
            audio_result = download_audio(args.url, output_dir)
            if "error" in audio_result:
                # If subs were found, this is a partial success
                if data.get("subtitles"):
                    data["audio"] = None
                    data["audio_error"] = audio_result["error"]
                else:
                    result = {"success": False, "error": audio_result["error"]}
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                    return 1
            else:
                data["audio"] = audio_result

        result = {"success": True, "data": data}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    except (OSError, ValueError, RuntimeError) as e:
        result = {"success": False, "error": str(e)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1


def _run_transcript_cli(args) -> int:
    """CLI for the transcript cascade (subtitles + whisper fallback)."""
    # Configure logging for whisper mode (useful for progress)
    if args.mode == "whisper":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    # Validate URL
    if not is_youtube_url(args.url):
        print(
            json.dumps(
                {"success": False, "error": f"Not a YouTube URL: {args.url}"},
                ensure_ascii=False,
            )
        )
        return 1

    # Fetch metadata
    metadata = fetch_metadata(args.url)
    if "error" in metadata:
        print(
            json.dumps(
                {"success": False, "error": metadata["error"]},
                ensure_ascii=False,
            )
        )
        return 1

    # --- Whisper mode: download audio + transcribe ---
    if args.mode == "whisper":
        return _run_whisper_mode(args, metadata)

    # --- Subtitle mode (transcript / analyze) ---
    # Pass subtitle_langs from metadata to avoid a duplicate --dump-json call
    with tempfile.TemporaryDirectory(prefix="wae_ytsub_") as tmpdir:
        subs = fetch_subtitles(
            args.url, tmpdir,
            subtitle_langs=metadata.get("subtitle_langs"),
        )

    if not subs:
        # Differentiate two very different failures that both yield "no subs":
        #   * Subtitles ARE published (metadata listed langs) but the targeted
        #     download failed / timed out -> yt-dlp throttle. RETRYABLE: exit 1
        #   * No subtitles published at all -> exit 2 signals "needs whisper".
        langs = metadata.get("subtitle_langs") or {}
        subs_listed = bool(langs.get("manual") or langs.get("auto"))
        if subs_listed:
            print(
                json.dumps(
                    {
                        "success": False,
                        "retryable": True,
                        "error": (
                            "Subtitles are published but their download failed "
                            "(yt-dlp throttle). Retry the subtitle path."
                        ),
                        "subtitle_langs": langs,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        # No subtitles available -- exit code 2 signals "needs whisper"
        print(
            json.dumps(
                {
                    "needs_whisper": True,
                    "metadata": metadata,
                    "message": (
                        "No subtitles available for this video. "
                        "A whisper fallback is required."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    transcript_text = subs["text"]
    source = subs["source"]
    language = subs["language"]

    if args.mode == "transcript":
        print(format_human(metadata, transcript_text, source, language))
    else:
        print(
            json.dumps(
                format_llm_context(
                    metadata, transcript_text, source, language, args.url
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


def _run_whisper_mode(args, metadata: dict) -> int:
    """Download audio and transcribe with faster-whisper.

    Returns exit code: 0 on success, 3 on whisper failure.
    """
    # Step 1: Download audio via yt-dlp
    with tempfile.TemporaryDirectory(prefix="wae_whisper_") as tmpdir:
        logger.info("Downloading audio for whisper transcription: %s", args.url)
        audio_result = download_audio(args.url, tmpdir)

        if "error" in audio_result:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Audio download failed: {audio_result['error']}",
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 3

        audio_path = audio_result["audio_path"]
        logger.info(
            "Audio downloaded: %s (%.2f MB, %s)",
            audio_path, audio_result.get("file_size_mb", 0),
            audio_result.get("format", "unknown"),
        )

        # Step 2: Transcribe with faster-whisper
        try:
            whisper_result = _transcribe_with_whisper(
                audio_path,
                model_size=args.whisper_model,
                language=args.language,
            )
        except ImportError as exc:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"faster-whisper not available: {exc}",
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 3
        except (RuntimeError, OSError) as exc:
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Whisper transcription failed: {exc}",
                        "metadata": metadata,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 3

    # Step 3: Format output as JSON (same structure as --mode analyze)
    result = format_whisper_context(metadata, whisper_result, args.url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main_download() -> int:
    """CLI entry point for the audio/subtitle downloader."""
    parser = argparse.ArgumentParser(
        description="Download YouTube audio and subtitles via yt-dlp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--url", required=True, type=str,
        help="YouTube video URL",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-created temp dir)",
    )
    parser.add_argument(
        "--audio-only", action="store_true",
        help="Download audio only (skip subtitle check)",
    )
    parser.add_argument(
        "--subs-only", action="store_true",
        help="Download subtitles only (skip audio download)",
    )
    parser.add_argument(
        "--subs-lang", type=str, default=None,
        help="Comma-separated preferred subtitle languages (default: fr,en,nl,de)",
    )
    args = parser.parse_args()
    return _run_download_cli(args)


def main_transcript() -> int:
    """CLI entry point for the transcript cascade."""
    parser = argparse.ArgumentParser(
        description="YouTube transcript extraction (zero LLM for subtitles, "
                    "faster-whisper for whisper mode)",
    )
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument(
        "--mode",
        choices=["transcript", "analyze", "whisper"],
        default="transcript",
        help="Output mode: transcript (human text), analyze (JSON for LLM), "
             "or whisper (audio download + faster-whisper transcription)",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Whisper model size for --mode whisper (default: medium). "
             "Options: tiny, base, small, medium, large-v3",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="ISO language code for whisper transcription (default: auto-detect)",
    )
    args = parser.parse_args()
    return _run_transcript_cli(args)


if __name__ == "__main__":
    sys.exit(main_transcript())
