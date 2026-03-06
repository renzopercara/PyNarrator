"""video_engine.py – Video download, clip-building and render helpers for PyNarrator.

Centralises the three core video operations required by the micro-learning
workflow:

* :func:`download_video`      – Downloads a YouTube clip via ``yt-dlp`` using
                                the best available MP4 + M4A audio format.
* :func:`_transcode_to_proxy` – Re-encodes a video to ``libx264/baseline/yuv420p``
                                via ``ffmpeg`` before loading.  This is the "proxy
                                trick" that prevents blank/white frames caused by
                                AV1 inter-frame decoding issues in MoviePy.
* :func:`build_source_clip`   – Loads a local MP4 file, optionally transcodes it
                                to a proxy, normalises resolution, strips alpha,
                                fixes FPS, trims the clip, and composites it over
                                a solid black background to guarantee visible frames.
* :func:`render_video`        – Concatenates scene clips with
                                ``method="compose"`` (prevents audio-stream
                                cancellation) and writes the final file with
                                ``yuv420p`` pixel format for maximum compatibility.

Pixel-format fix
----------------
When mixing YouTube clips (YUV colour space) with TTS-over-image educational
cards (RGB), some players render the YouTube segment as a white rectangle.
Adding ``-pix_fmt yuv420p`` to ``ffmpeg_params`` forces a uniform colour space
across all frames and eliminates this artefact.

Proxy-transcoding fix
---------------------
YouTube clips encoded with AV1 (or other inter-frame codecs) may have sparse
I-frames.  When MoviePy subclips such a video it sometimes cannot locate the
nearest keyframe and draws blank/white frames instead.  Re-encoding the whole
file to ``libx264/baseline`` before loading guarantees that every frame can be
decoded independently.

Explicit-background fix
-----------------------
Compositing the clip over a solid black :class:`~moviepy.editor.ColorClip`
with :class:`~moviepy.editor.CompositeVideoClip` forces the render engine to
treat the video as a concrete image layer, eliminating white-transparency
artefacts that appear when MoviePy encounters a clip without a fully opaque
pixel buffer.

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


def _transcode_to_proxy(input_path: str) -> str:
    """Re-encode *input_path* to a ``libx264/baseline/yuv420p`` proxy file.

    YouTube clips are often encoded with AV1 or other inter-frame codecs that
    have sparse I-frames.  When MoviePy subclips such a video it may miss the
    nearest keyframe and render blank/white frames.  Re-encoding the whole file
    to a simple intra-friendly profile before loading eliminates this problem.

    The proxy file is written to the system temporary directory.  It is not
    deleted automatically because :class:`~moviepy.editor.VideoFileClip` reads
    frames lazily via a long-lived ``ffmpeg`` subprocess; deleting the file
    immediately after opening would break frame decoding on some platforms.
    The OS will remove orphaned temp files on reboot (or via standard temp
    directory housekeeping).

    Args:
        input_path: Path to the original video file.

    Returns:
        Path to the temporary proxy MP4 file.

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or transcoding fails.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix="_proxy.mp4", delete=False) as f:
        proxy_path = f.name

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        proxy_path,
    ]

    logger.info("🔄 Transcoding to proxy: %s → %s", input_path, proxy_path)
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug("ffmpeg stdout: %s", result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it with: apt install ffmpeg  (or brew install ffmpeg on macOS)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg proxy transcoding failed (exit {exc.returncode}): {exc.stderr}"
        ) from exc

    logger.info("✅ Proxy ready: %s", proxy_path)
    return proxy_path


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
    fps: int = 24,
    transcode_proxy: bool = True,
):
    """Load a local MP4 file as a normalised MoviePy clip ready for concatenation.

    Applies the following fixes in order:

    1. **Proxy transcoding** – When *transcode_proxy* is ``True`` (default),
       the source file is first re-encoded to a ``libx264/baseline/yuv420p``
       proxy via :func:`_transcode_to_proxy`.  This prevents MoviePy from
       encountering inter-frame codec issues (e.g. AV1) that produce
       blank/white frames after subclipping.
    2. **Resolution normalisation** – ``target_resolution`` is passed to
       :class:`~moviepy.editor.VideoFileClip` so that the decoder scales
       the frame during demux.  This avoids mismatched frame sizes when
       compositing clips from different sources.
    3. **Alpha-channel removal** – :meth:`~moviepy.editor.VideoClip.without_mask`
       and ``.set_opacity(1)`` strip any residual alpha channel.
    4. **FPS normalisation** – ``.set_fps(fps)`` sets a consistent frame rate
       across all clips before compositing (default: 24 fps).
    5. **Time trimming** – The clip is subclipped to [*start_time*, *end_time*].
    6. **Explicit background compositing** – The trimmed clip is placed over a
       solid black :class:`~moviepy.editor.ColorClip` background using
       :class:`~moviepy.editor.CompositeVideoClip`.  This forces the render
       engine to treat the video as a concrete image layer and eliminates
       white-transparency artefacts.

    Args:
        path:              Absolute or relative path to the MP4 file.
        start_time:        Start offset in seconds (default: 0).
        end_time:          End offset in seconds.  ``None`` means use the full
                           clip duration.
        target_resolution: ``(height, width)`` tuple for resolution
                           normalisation (default: ``(720, 1280)``).
        fps:               Target frames per second (default: 24).
        transcode_proxy:   When ``True`` (default), re-encode to a libx264
                           proxy before loading to avoid AV1 blank-frame bugs.
                           The proxy temp file is managed by the OS and will
                           be cleaned up on reboot or by standard temp-dir
                           housekeeping (see :func:`_transcode_to_proxy`).

    Returns:
        A :class:`~moviepy.editor.CompositeVideoClip` ready for concatenation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If *start_time* is negative or *end_time* ≤ *start_time*.
        RuntimeError:      If proxy transcoding fails (only when
                           *transcode_proxy* is ``True``).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    if start_time < 0:
        raise ValueError(f"start_time must be non-negative, got {start_time}.")
    if end_time is not None and end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})."
        )

    from moviepy.editor import ColorClip, CompositeVideoClip, VideoFileClip

    # Proxy trick: re-encode to libx264/baseline/yuv420p so MoviePy can decode
    # all frames correctly, even after subclipping an AV1-encoded YouTube video.
    load_path = path
    if transcode_proxy:
        load_path = _transcode_to_proxy(path)

    logger.info(
        "🎬 Loading clip: %s [%ss – %ss] @ target_resolution=%s fps=%s",
        load_path, start_time, end_time, target_resolution, fps,
    )

    clip = VideoFileClip(load_path, target_resolution=target_resolution)

    # Strip residual alpha channel to prevent white-frame rendering artefacts.
    clip = clip.without_mask().set_opacity(1)

    # Normalise FPS across all clips to avoid sync issues during compositing.
    clip = clip.set_fps(fps)

    # Trim to requested window
    clip_end = min(end_time, clip.duration) if end_time is not None else clip.duration
    clip = clip.subclip(start_time, clip_end)

    # Explicit background compositing: place the clip over a solid black
    # ColorClip.  This forces the render engine to treat the video as a
    # concrete image layer and eliminates white transparency artefacts.
    bg = ColorClip(
        size=(clip.w, clip.h),
        color=(0, 0, 0),
    ).set_duration(clip.duration)
    composite = CompositeVideoClip([bg, clip.set_position("center")])

    logger.info("✅ Clip ready: %.2fs", composite.duration)
    return composite


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
    # Anchor the first clip at t=0 to prevent timing drift when concatenating
    # clips that originated from different sources (YouTube vs. TTS cards).
    anchored = [clips[0].set_start(0), *clips[1:]]
    final = concatenate_videoclips(anchored, method="compose")

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
