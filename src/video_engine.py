"""video_engine.py – Video download, clip-building and render helpers for PyNarrator.

Centralises the three core video operations required by the micro-learning
workflow:

* :func:`download_video`    – Downloads a YouTube clip via ``yt-dlp`` using the
                              best available MP4 + M4A audio format.
* :func:`build_source_clip` – Loads a local MP4 file as a MoviePy
                              :class:`~moviepy.editor.VideoFileClip`, strips
                              any alpha channel, and normalises resolution to
                              avoid white/blank frames on social-media players.
* :func:`render_video`      – Concatenates scene clips with
                              ``method="compose"`` (prevents audio-stream
                              cancellation) and writes the final file with
                              ``yuv420p`` pixel format for maximum
                              compatibility.

Pixel-format fix
----------------
When mixing YouTube clips (YUV colour space) with TTS-over-image educational
cards (RGB), some players render the YouTube segment as a white rectangle.
Adding ``-pix_fmt yuv420p`` to ``ffmpeg_params`` forces a uniform colour space
across all frames and eliminates this artefact.

Audio-sync fix
--------------
Using ``method="compose"`` in :func:`~moviepy.editor.concatenate_videoclips`
ensures that each clip's audio track is preserved on its own timeline instead
of being merged and potentially cancelled by out-of-phase waveforms from
different sources (YouTube audio vs. Edge-TTS).

Usage::

    from src.video_engine import download_video, build_source_clip, render_video

    local_path = download_video(
        url="https://www.youtube.com/watch?v=nCKdihvneS0",
        output_dir="assets/video",
    )
    clip = build_source_clip(local_path, start_time=13, end_time=26)
    render_video([clip], output_path="output/lesson.mp4")
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yt-dlp download format – best MP4 video + M4A audio to guarantee a single
# container that FFmpeg can re-mux without quality loss.
# ---------------------------------------------------------------------------
_YTDLP_FORMAT = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"

# Target resolution for source-clip normalisation (height × width in MoviePy
# convention).  720 p portrait normalises YouTube widescreen clips while still
# giving enough vertical resolution for Reels / Shorts.
_TARGET_RESOLUTION = (720, 1280)

# ffmpeg_params injected into every write_videofile call.
# -pix_fmt yuv420p  → universal colour-space; fixes white/blank YouTube frames
# -movflags +faststart → metadata at file start; required for streaming
_FFMPEG_PARAMS = [
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
]


def download_video(
    url: str,
    output_dir: str = "assets/video",
    filename: str = "source.mp4",
) -> str:
    """Download a YouTube video using ``yt-dlp`` and return the local file path.

    Uses the format ``bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]``
    so that the resulting file always contains both a video stream *and* a
    properly muxed audio stream – a prerequisite for :func:`build_source_clip`
    to extract audio without extra post-processing.

    Args:
        url:        YouTube (or other yt-dlp-compatible) URL.
        output_dir: Directory where the downloaded file will be saved.
        filename:   Output filename (must end with ``.mp4``).

    Returns:
        Absolute path to the downloaded MP4 file.

    Raises:
        RuntimeError: If ``yt-dlp`` is not installed or the download fails.
        ValueError:   If *url* is empty.
    """
    if not url:
        raise ValueError("url must not be empty.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    cmd = [
        "yt-dlp",
        "--format", _YTDLP_FORMAT,
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
        url,
    ]

    logger.info("⬇ Downloading video: %s → %s", url, output_path)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.debug("yt-dlp stdout: %s", result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "yt-dlp is not installed or not on PATH. "
            "Install it with: pip install yt-dlp"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"yt-dlp failed (exit {exc.returncode}): {exc.stderr}"
        ) from exc

    logger.info("✅ Download complete: %s", output_path)
    return output_path


def build_source_clip(
    path: str,
    start_time: float = 0,
    end_time: Optional[float] = None,
    target_resolution: tuple[int, int] = _TARGET_RESOLUTION,
):
    """Load a local MP4 file as a normalised MoviePy VideoFileClip.

    Applies the following fixes in order:

    1. **Resolution normalisation** – ``target_resolution`` is passed to
       :class:`~moviepy.editor.VideoFileClip` so that the decoder scales the
       frame during demux.  This avoids mismatched frame sizes when compositing
       clips from different sources.
    2. **Alpha-channel removal** – :meth:`~moviepy.editor.VideoClip.without_mask`
       and ``.set_opacity(1)`` strip any residual alpha channel that could cause
       a white overlay on players that do not support transparency.
    3. **Time trimming** – The clip is subclipped to [*start_time*, *end_time*].

    Args:
        path:              Absolute or relative path to the MP4 file.
        start_time:        Start offset in seconds (default: 0).
        end_time:          End offset in seconds.  ``None`` means use the full
                           clip duration.
        target_resolution: ``(height, width)`` tuple for resolution
                           normalisation (default: ``(720, 1280)``).

    Returns:
        A :class:`~moviepy.editor.VideoFileClip` ready for concatenation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If *start_time* is negative or *end_time* ≤ *start_time*.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}.")
    if end_time is not None and end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})."
        )

    from moviepy.editor import VideoFileClip

    logger.info(
        "🎬 Loading clip: %s [%ss – %ss] @ target_resolution=%s",
        path, start_time, end_time, target_resolution,
    )

    clip = VideoFileClip(path, target_resolution=target_resolution)

    # Strip residual alpha channel to prevent white-frame rendering artefacts.
    clip = clip.without_mask().set_opacity(1)

    # Trim to requested window
    clip_end = min(end_time, clip.duration) if end_time is not None else clip.duration
    clip = clip.subclip(start_time, clip_end)

    logger.info("✅ Clip ready: %.2fs @ %dx%d", clip.duration, clip.w, clip.h)
    return clip


def render_video(
    clips: List,
    output_path: str,
    fps: int = 24,
    codec: str = "libx264",
    audio_codec: str = "aac",
    ffmpeg_params: Optional[List[str]] = None,
    threads: int = 4,
) -> None:
    """Concatenate *clips* and write the final video to *output_path*.

    Audio-sync fix
    ~~~~~~~~~~~~~~
    Passes ``method="compose"`` to
    :func:`~moviepy.editor.concatenate_videoclips` which composites each
    clip on its own timeline.  This prevents audio streams from different
    sources (YouTube audio vs. Edge-TTS) from being merged naïvely and
    potentially cancelling each other.

    Pixel-format fix
    ~~~~~~~~~~~~~~~~
    The default *ffmpeg_params* includes ``-pix_fmt yuv420p`` which forces a
    uniform colour space and eliminates white/blank frames on social-media
    players caused by colour-space incompatibilities between YouTube clips and
    image-card clips.

    Args:
        clips:         List of MoviePy clip objects to concatenate.
        output_path:   Destination file path (e.g. ``"output/lesson.mp4"``).
        fps:           Frames per second for the output (default: 24).
        codec:         Video codec (default: ``"libx264"``).
        audio_codec:   Audio codec (default: ``"aac"``).
        ffmpeg_params: Extra ffmpeg arguments.  Defaults to
                       ``["-pix_fmt", "yuv420p", "-movflags", "+faststart"]``.
        threads:       Number of FFmpeg encoding threads (default: 4).

    Raises:
        ValueError: If *clips* is empty.
    """
    if not clips:
        raise ValueError("clips must not be empty.")

    from moviepy.editor import concatenate_videoclips

    if ffmpeg_params is None:
        ffmpeg_params = list(_FFMPEG_PARAMS)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info("🔗 Concatenating %d clip(s) with method='compose'…", len(clips))
    final = concatenate_videoclips(clips, method="compose")

    logger.info("💾 Rendering → %s", output_path)
    final.write_videofile(
        output_path,
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
        ffmpeg_params=ffmpeg_params,
        threads=threads,
        logger=None,
        verbose=False,
    )
    logger.info("✅ Render complete: %s", output_path)
