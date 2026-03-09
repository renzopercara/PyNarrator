"""video_engine.py – Video download, clip-building and render helpers for PyNarrator.

Centralises the four core video operations required by the micro-learning
workflow:

* :func:`download_video`          – Downloads a YouTube clip via ``yt-dlp`` using
                                    the best available MP4 + M4A audio format.
* :func:`normalize_youtube_clip`  – Trims **and** re-encodes a YouTube clip to
                                    ``libx264/yuv420p/aac`` in a **single FFmpeg
                                    pass**.  This is the recommended pre-processing
                                    step before handing the file to MoviePy.
* :func:`_transcode_to_proxy`     – Re-encodes a video to ``libx264/baseline/yuv420p``
                                    via ``ffmpeg`` before loading.  This is the "proxy
                                    trick" that prevents blank/white frames caused by
                                    AV1 inter-frame decoding issues in MoviePy.
* :func:`build_source_clip`       – Loads a local MP4 file, optionally pre-processes it
                                    via :func:`normalize_youtube_clip` (single-pass trim
                                    + normalise) or :func:`_transcode_to_proxy`
                                    (full-file normalise), then normalises resolution,
                                    strips alpha, fixes FPS, and composites it over a
                                    solid black background to guarantee visible frames.
* :func:`render_video`            – Concatenates scene clips with
                                    ``method="compose"`` (prevents audio-stream
                                    cancellation) and writes the final file with
                                    ``yuv420p`` pixel format for maximum compatibility.

Normalisation pipeline (recommended for YouTube clips)
------------------------------------------------------
The recommended pipeline for YouTube clips is:

1. ``download_video`` – fetch the raw file with ``yt-dlp``.
2. ``normalize_youtube_clip`` – trim to the desired window **and** re-encode
   with ``libx264/yuv420p/aac`` in a single FFmpeg pass.  The normalised file
   is written to ``assets/tmp/youtube_normalized.mp4`` by default.
3. ``build_source_clip`` – load the normalised file, apply resolution/FPS
   normalisation, strip alpha, and composite over a black background.
4. ``render_video`` – concatenate and write the final MP4.

:func:`build_source_clip` automatically invokes :func:`normalize_youtube_clip`
instead of :func:`_transcode_to_proxy` when *transcode_proxy* is ``True`` and
*end_time* is specified (the common YouTube-clip scenario), collapsing trim +
normalise into one FFmpeg pass for efficiency.

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

    from src.video_engine import download_video, normalize_youtube_clip, build_source_clip, render_video

    raw_path = download_video(
        url="https://www.youtube.com/watch?v=nCKdihvneS0",
        output_dir="assets/tmp",
        filename="youtube_raw.mp4",
    )
    normalized_path = normalize_youtube_clip(
        input_path=raw_path,
        output_path="assets/tmp/youtube_normalized.mp4",
        start="13",
        end="26",
    )
    clip = build_source_clip(normalized_path, start_time=0, end_time=13)
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

# Temporary directory used to store intermediate YouTube processing files
# (``youtube_raw.mp4`` and ``youtube_normalized.mp4``).  Created on demand.
_TMP_DIR = os.path.join("assets", "tmp")


def normalize_youtube_clip(
    input_path: str,
    output_path: str,
    start: str,
    end: str,
) -> str:
    """Trim and normalise a YouTube clip using a single FFmpeg pass.

    Runs FFmpeg to cut the source video to the window [*start*, *end*] **and**
    re-encode it with ``libx264/yuv420p/aac`` in one pass.  This is the
    recommended pre-processing step before handing the file to MoviePy because:

    * ``-ss`` is placed **before** ``-i`` (fast input seeking), so FFmpeg jumps
      directly to the nearest keyframe before *start* without decoding every
      preceding frame.  For a 1-hour source this reduces seek time from minutes
      to milliseconds.
    * ``-t`` (clip duration) is used instead of ``-to`` (absolute end time)
      because with input-seeking the output clock starts at 0, making ``-to``
      relative to the *output* rather than the *source*.
    * FFmpeg aligns the output to a key-frame boundary, preventing the
      blank/black-frame artefacts that occur when MoviePy subclips an AV1
      (or other inter-frame codec) source.
    * The ``yuv420p`` pixel format is universally supported by all players and
      by MoviePy's colour-compositing pipeline.
    * A single pass is more efficient than normalising the full file first and
      then trimming (the previous two-step approach).

    The parent directory of *output_path* is created automatically if it does
    not exist (including ``assets/tmp/`` for the default pipeline).

    Args:
        input_path:  Path to the raw downloaded video file (e.g.
                     ``"assets/tmp/youtube_raw.mp4"``).
        output_path: Destination path for the normalised clip (e.g.
                     ``"assets/tmp/youtube_normalized.mp4"``).
        start:       Start time as a numeric string in seconds (e.g. ``"13"``
                     or ``"13.5"``).
        end:         End time as a numeric string in seconds (e.g. ``"26"``).
                     The clip duration is derived as ``float(end) - float(start)``.

    Returns:
        *output_path* – the path to the normalised MP4 file – for convenience
        in call chains.

    Raises:
        RuntimeError: If ``ffmpeg`` is not installed or normalisation fails.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Place -ss BEFORE -i for fast input seeking: FFmpeg jumps to the nearest
    # keyframe before *start* without decoding the preceding video.  This is
    # orders of magnitude faster than output-seeking (placing -ss after -i),
    # which forces FFmpeg to decode every frame from the beginning up to *start*.
    # Use -t (duration) instead of -to (absolute end time) because with input
    # seeking the output clock starts at 0, so -to would be interpreted relative
    # to the output and could silently produce an empty or wrong-length file.
    try:
        clip_duration = float(end) - float(start)
    except (TypeError, ValueError):
        clip_duration = None
        logger.warning(
            "⚠️ normalize_youtube_clip: could not compute clip duration from"
            " start=%r end=%r – omitting -t flag; output length is unbounded.",
            start, end,
        )

    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path]
    if clip_duration is not None:
        cmd += ["-t", str(clip_duration)]
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]

    logger.info(
        "🎬 Normalizing YouTube clip: %s [%s → %s] → %s",
        input_path, start, end, output_path,
    )
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
            f"ffmpeg normalize_youtube_clip failed (exit {exc.returncode}): {exc.stderr}"
        ) from exc

    logger.info("✅ Normalized clip ready: %s", output_path)
    return output_path


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

    After the download completes the function verifies that the file actually
    exists at *output_path*.  When yt-dlp writes to a slightly different path
    (e.g. it appends the video-ID or a different extension) the newest ``.mp4``
    file in *output_dir* is used instead.  This prevents the "Asset not found"
    fallback that occurs when the caller passes the URL directly to the render
    pipeline without a prior download step.

    Args:
        url:        YouTube (or other yt-dlp-compatible) URL.
        output_dir: Directory where the downloaded file will be saved.
                    Resolved to an absolute path automatically.
        filename:   Output filename (must end with ``.mp4``).

    Returns:
        Absolute path to the downloaded MP4 file.

    Raises:
        RuntimeError: If ``yt-dlp`` is not installed or the download fails,
                      or if no MP4 file can be found in *output_dir* after
                      the download.
        ValueError:   If *url* is empty.
    """
    if not url:
        raise ValueError("url must not be empty.")

    output_dir = os.path.abspath(output_dir)
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

    # Verify the file exists at the expected path.  yt-dlp occasionally writes
    # to a slightly different name (e.g. when the URL resolves to a different
    # video ID or when the container extension differs).  Fall back to the
    # newest MP4 in output_dir so the caller always gets a valid local path.
    if not os.path.exists(output_path):
        logger.warning(
            "⚠️ Expected download path not found: %s – searching %s for the newest MP4.",
            output_path, output_dir,
        )
        mp4_files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith(".mp4")
        ]
        if not mp4_files:
            raise RuntimeError(
                f"Downloaded file not found at expected path and no MP4 files "
                f"discovered in {output_dir}. yt-dlp output: {result.stdout}"
            )
        output_path = max(mp4_files, key=os.path.getmtime)
        logger.info("📂 Using discovered download: %s", output_path)

    logger.info("✅ Download complete: %s", output_path)
    return output_path


def build_source_clip(
    path: str,
    start_time: float = 0,
    end_time: Optional[float] = None,
    target_resolution: tuple[int, int] = _TARGET_RESOLUTION,
    fps: int = 24,
    transcode_proxy: bool = True,
    tmp_dir: str = _TMP_DIR,
):
    """Load a local MP4 file as a normalised MoviePy clip ready for concatenation.

    Applies the following fixes in order:

    1. **FFmpeg pre-processing** – When *transcode_proxy* is ``True`` (default):

       * If *end_time* is provided (the common YouTube-clip scenario), calls
         :func:`normalize_youtube_clip` to **trim and re-encode** the clip in
         a single FFmpeg pass.  The normalised file is written to
         ``<tmp_dir>/youtube_normalized.mp4``.  This is the most reliable
         approach because FFmpeg aligns output to a key-frame boundary and
         forces ``yuv420p``, preventing blank/black-frame artefacts caused by
         AV1 inter-frame decoding.
       * If *end_time* is ``None``, falls back to :func:`_transcode_to_proxy`
         (full-file normalise to ``libx264/baseline/yuv420p``).

    2. **Resolution normalisation** – ``target_resolution`` is passed to
       :class:`~moviepy.editor.VideoFileClip` so that the decoder scales
       the frame during demux.  This avoids mismatched frame sizes when
       compositing clips from different sources.
    3. **Alpha-channel removal** – If ``clip.mask`` is not ``None``,
       ``set_mask(None)`` strips the alpha channel.  ``set_opacity(1.0)``
       then forces full opacity to prevent alpha-channel rendering errors.
    4. **FPS normalisation** – ``.set_fps(fps)`` sets a consistent frame rate
       across all clips before compositing (default: 24 fps).
    5. **Explicit duration** – When the clip was pre-trimmed by
       :func:`normalize_youtube_clip`, ``set_duration`` is called with the
       expected duration for reliability.  Otherwise ``subclip`` is used.
    6. **Explicit background compositing** – The trimmed clip is placed over a
       solid black :class:`~moviepy.editor.ColorClip` background using
       :class:`~moviepy.editor.CompositeVideoClip`.  This forces the render
       engine to treat the video as a concrete image layer and eliminates
       white-transparency artefacts.
    7. **Audio assignment** – The audio track from the loaded clip is explicitly
       assigned to the composite to guarantee sync.

    Args:
        path:              Absolute or relative path to the MP4 file.
        start_time:        Start offset in seconds (default: 0).
        end_time:          End offset in seconds.  ``None`` means use the full
                           clip duration.
        target_resolution: ``(height, width)`` tuple for resolution
                           normalisation (default: ``(720, 1280)``).
        fps:               Target frames per second (default: 24).
        transcode_proxy:   When ``True`` (default), pre-process with FFmpeg
                           before loading to avoid AV1 blank-frame bugs.  When
                           *end_time* is set, uses :func:`normalize_youtube_clip`
                           (one-pass trim + normalise); otherwise uses
                           :func:`_transcode_to_proxy` (full-file normalise).
        tmp_dir:           Directory for intermediate normalised files (default:
                           ``assets/tmp``).  Created automatically if absent.

    Returns:
        A :class:`~moviepy.editor.CompositeVideoClip` ready for concatenation.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError:        If *start_time* is negative or *end_time* ≤ *start_time*.
        RuntimeError:      If FFmpeg pre-processing fails (only when
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

    # FFmpeg pre-processing: normalise codec/pixel-format before MoviePy loads
    # the file.  This prevents blank/black-frame artefacts caused by AV1
    # inter-frame decoding issues and colour-space mismatches.
    load_path = path
    already_trimmed = False
    if transcode_proxy and end_time is not None:
        # One-pass approach: trim + normalise with FFmpeg.
        # More efficient and reliable than normalising first, then subclipping.
        os.makedirs(tmp_dir, exist_ok=True)
        normalized_path = os.path.join(tmp_dir, "youtube_normalized.mp4")
        load_path = normalize_youtube_clip(
            path, normalized_path, str(start_time), str(end_time)
        )
        already_trimmed = True
    elif transcode_proxy:
        # Fall back to full-file proxy transcoding when no trim window given.
        load_path = _transcode_to_proxy(path)

    logger.info(
        "🎬 Loading clip: %s [%ss – %ss] @ target_resolution=%s fps=%s",
        load_path, start_time, end_time, target_resolution, fps,
    )

    clip = VideoFileClip(load_path, target_resolution=target_resolution)

    # Strip residual alpha channel to prevent white-frame rendering artefacts.
    # .without_mask() does not exist in all MoviePy versions; use set_mask(None)
    # directly so the call is safe regardless of API version.
    if clip.mask is not None:
        clip = clip.set_mask(None)
    clip = clip.set_opacity(1.0)
    logger.debug(
        "Clip loaded: size=%s duration=%.2fs fps=%s",
        clip.size, clip.duration, clip.fps,
    )

    # Normalise FPS across all clips to avoid sync issues during compositing.
    clip = clip.set_fps(fps)
    logger.debug(
        "After set_fps: size=%s duration=%.2fs fps=%s",
        clip.size, clip.duration, clip.fps,
    )

    if already_trimmed:
        # Clip is pre-trimmed by FFmpeg; set explicit duration for reliability.
        explicit_duration = end_time - start_time
        clip = clip.set_duration(min(explicit_duration, clip.duration))
    else:
        # Trim to requested window via MoviePy.
        clip_end = min(end_time, clip.duration) if end_time is not None else clip.duration
        clip = clip.subclip(start_time, clip_end)
    logger.debug(
        "After subclip: size=%s duration=%.2fs fps=%s",
        clip.size, clip.duration, clip.fps,
    )

    # Explicit background compositing: place the clip over a solid black
    # ColorClip.  This forces the render engine to treat the video as a
    # concrete image layer and eliminates white transparency artefacts.
    # use_bgclip=True designates the black ColorClip as the master reference
    # so the composite inherits its size and duration.
    bg = ColorClip(
        size=(clip.w, clip.h),
        color=(0, 0, 0),
    ).set_duration(clip.duration)
    composite = CompositeVideoClip([bg, clip.set_position("center")], use_bgclip=True)

    # Explicitly assign audio via set_audio() to guarantee correct muxing.
    composite = composite.set_audio(clip.audio)

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
