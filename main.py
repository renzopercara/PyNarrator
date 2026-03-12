import json
import asyncio
import logging
import os
import random
import subprocess
import textwrap
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageEnhance

from src.narrator import ArgentineNarrator
from src.image_manager import get_visual_assets
from src.subtitle_generator import generate_subtitles
from src.sentiment_analyzer import analyze_tone
from src.copy_generator import generate_social_copy
from src.vocabulary_annotator import annotate_story
from src.config import (
    OUTPUT_DIR, AUDIO_DIR,
    MUSIC_DIR, SFX_DIR, VIDEO_RES, LOGO_PATH,
    WATERMARK_PATH, WATERMARK_OPACITY, WATERMARK_WIDTH_PERCENT,
    VOICES, ESL_VOICE_RATE, ESL_VOICE_PITCH,
)
from src.context import (
    detect_context as _detect_context,
    output_path_for_context as _output_path_for_context,
)
from src.video_engine import _transcode_to_proxy, download_video, normalize_youtube_clip

from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, ColorClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, vfx,
)
import moviepy.audio.fx.all as afx

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = VIDEO_RES

# --- PLATFORM RESOLUTIONS ---
# Maps a "target platform" name to its (width, height) render resolution.
_PLATFORM_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "Instagram": (1080, 1920),
    "LinkedIn": (1080, 1080),
}

# Current target resolution – updated at the start of main() based on the
# chosen platform.  Helper functions that need the render size read from
# this variable so the whole pipeline stays consistent.
_TARGET_RESOLUTION: tuple[int, int] = VIDEO_RES

# Smartbuild brand colour used as a fallback background when visual assets
# are unavailable (deep-navy blue).
_BRAND_COLOR: tuple[int, int, int] = (15, 20, 50)

# Source assets longer than this threshold (in seconds) trigger the trim-first
# transcoding path: FFmpeg trims to the scene window before MoviePy loads the
# file.  This prevents encoding an entire 1-hour YouTube video for a 5-second
# micro-learning clip.
_LONG_ASSET_THRESHOLD_SECONDS: float = 60.0

# If the total computed duration of all scene clips exceeds this limit before
# the final export, a warning is logged listing the scenes causing the bloat.
_MAX_EXPORT_DURATION_SECONDS: float = 300.0

# Default scene duration in seconds used when a scene has no explicit end_time.
# JSON timestamps are the single source of truth; this fallback only applies
# when end_time is completely absent from the script.
_DEFAULT_SCENE_DURATION: float = 10.0

# --- CONSTANTES DE ESTILO OPTIMIZADAS ---
_ZOOM_RATE_NORMAL = 0.02
_CONTRAST_BOOST = 1.15   
_SATURATION_BOOST = 1.25 
_GAMMA_CORRECTION = 0.80 

def _enhance_frame(frame):
    """Mejora visual: Brillo, Contraste y Saturación.

    Guarantees that the returned array is ``uint8`` with the **exact same
    shape** as *frame* and contains no ``NaN`` values, so MoviePy's
    ``fl_image`` pipeline never produces corrupted (black/white) frames.

    If any enhancement step raises an exception the original (safe-cast)
    frame is returned unchanged so the pipeline never stalls.
    """
    logger.debug(
        "_enhance_frame input: shape=%s dtype=%s", frame.shape, frame.dtype
    )
    # Ensure uint8 input before any PIL/NumPy operations.
    safe_frame = frame if frame.dtype == np.uint8 else frame.astype(np.uint8)
    try:
        img = PIL.Image.fromarray(safe_frame)
        img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
        img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
        img = PIL.ImageEnhance.Brightness(img).enhance(1.1)

        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.power(arr, _GAMMA_CORRECTION)
        # Guard against NaN / Inf that can arise from degenerate pixel values
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        result = np.clip(arr * 255, 0, 255).astype(np.uint8)

        # Shape must be identical to the input; fall back to the safe cast if not
        if result.shape != safe_frame.shape:
            logger.warning(
                "_enhance_frame shape mismatch: input=%s output=%s – returning original",
                safe_frame.shape, result.shape,
            )
            return safe_frame

        logger.debug(
            "_enhance_frame output: shape=%s dtype=%s", result.shape, result.dtype
        )
        return result
    except Exception as exc:
        logger.warning(
            "_enhance_frame failed (%s) – returning original frame.", exc
        )
        return safe_frame

def _load_hook_font(size: int):
    candidates = ["C:/Windows/Fonts/montserrat-extrabold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    for path in candidates:
        if os.path.exists(path): return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.0) -> ImageClip:
    font = _load_hook_font(75)
    text = topic.upper()
    img = PIL.Image.new("RGBA", (VIDEO_W, 300), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((VIDEO_W - tw) // 2, (300 - th) // 2), text, font=font, 
              fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    return ImageClip(np.array(img)).set_duration(duration).set_position(("center", 200)).crossfadeout(0.5)

def _make_watermark_clip(video_duration: float) -> ImageClip:
    if not os.path.exists(WATERMARK_PATH): return None
    watermark_w = int(VIDEO_W * WATERMARK_WIDTH_PERCENT)
    clip = ImageClip(WATERMARK_PATH).resize(width=watermark_w)
    return (clip.set_opacity(WATERMARK_OPACITY).set_duration(video_duration)
            .set_position(("center", VIDEO_H - clip.size[1] - 50)))


def _resolve_video_source(video_source: str, tmp_dir: str = "assets/tmp") -> str:
    """Return a local file path for *video_source*.

    When *video_source* is a URL (starts with ``http://`` or ``https://``),
    downloads the video via :func:`~src.video_engine.download_video` into
    *tmp_dir* and returns the absolute path to the downloaded file.  For
    local paths the value is returned unchanged (after resolving to an
    absolute path if it already exists on disk).

    This function is the entry-point fix for the "Asset not found for YouTube
    URLs" bug: the micro-learning pipeline stores a raw YouTube URL in
    ``video_source``; without this helper the URL is passed directly to
    :func:`_make_clip_for_scene` which cannot ``os.path.exists()`` a URL
    and silently falls back to a blank :class:`~moviepy.editor.ColorClip`.

    Args:
        video_source: A YouTube URL or a local file path.
        tmp_dir:      Directory used for the downloaded file (created
                      automatically if absent).

    Returns:
        Absolute local path to the video file, or *video_source* unchanged
        if the download fails (the caller's fallback logic will then produce
        a brand-colour placeholder clip and log a clear warning).
    """
    if not video_source:
        return video_source

    if video_source.startswith(("http://", "https://")):
        logger.info("🌐 YouTube URL detected in video_source – downloading: %s", video_source)
        try:
            local_path = download_video(
                url=video_source,
                output_dir=tmp_dir,
                filename="youtube_raw.mp4",
            )
            abs_path = os.path.abspath(local_path)
            logger.info("✅ video_source resolved to local path: %s", abs_path)
            return abs_path
        except Exception as exc:
            logger.error(
                "❌ Failed to download video_source '%s': %s – pipeline will use ColorClip fallback.",
                video_source, exc,
            )
            return video_source

    # Local path: resolve to absolute so downstream checks are consistent.
    abs_path = os.path.abspath(video_source)
    if not os.path.exists(abs_path):
        logger.warning(
            "⚠️ video_source local path does not exist: %s (searched: %s)",
            video_source, abs_path,
        )
    return abs_path


def _make_clip_for_scene(asset_path, duration, zoom_in=True, start_time=0, end_time=None):
    """Build a MoviePy clip for one scene from *asset_path*.

    Applies strict media validation, optional FFmpeg proxy transcoding for
    MP4 assets (to handle AV1 / H.264 inter-frame decoding issues), frame
    enhancement via :func:`_enhance_frame`, and explicit-background
    compositing to guarantee fully visible frames on all players.

    A visible brand-colour :class:`~moviepy.editor.ColorClip` is returned as
    a fallback whenever the asset is missing, has invalid dimensions/duration,
    or cannot be processed.
    """
    if not asset_path or not os.path.exists(asset_path):
        logger.warning(
            "⚠️ Asset not found: '%s' (absolute: '%s') – using brand ColorClip fallback.",
            asset_path,
            os.path.abspath(asset_path) if asset_path else "<None>",
        )
        return ColorClip(size=VIDEO_RES, color=_BRAND_COLOR, duration=duration)

    ext = os.path.splitext(asset_path)[1].lower()

    if ext == ".mp4":
        return _make_video_clip(asset_path, duration, start_time, end_time)

    # --- Lógica para Imágenes ---
    img_clip = ImageClip(asset_path).set_duration(duration)

    # Redimensionamos la imagen para que CUBRA todo el canvas (evita bordes)
    w, h = img_clip.size
    aspect_ratio_target = VIDEO_W / VIDEO_H
    if w / h > aspect_ratio_target:
        img_clip = img_clip.resize(height=VIDEO_H)
    else:
        img_clip = img_clip.resize(width=VIDEO_W)

    # Aplicamos el Zoom
    def _resize_fn(t):
        # Zoom sutil: de 1.0 a 1.1 o de 1.1 a 1.0
        base_scale = 1.0 + 0.1 * t / duration if zoom_in else 1.1 - 0.1 * t / duration
        return base_scale

    img_clip = img_clip.resize(_resize_fn).set_position("center")

    # CLAVE: Forzamos el CompositeVideoClip a VIDEO_RES y le ponemos un fondo negro sólido
    # Esto elimina cualquier transparencia que Instagram pueda detectar como "error de aspecto"
    return (
        CompositeVideoClip(
            [ColorClip(VIDEO_RES, color=(0, 0, 0)).set_duration(duration), img_clip],
            size=VIDEO_RES,
        )
        .set_duration(duration)
        .fl_image(_enhance_frame)
    )


def _make_video_clip(asset_path: str, duration: float, start_time: float, end_time,
                     tmp_dir: str = "assets/tmp"):
    """Load, normalise, and composite an MP4 asset into a scene clip.

    Processing pipeline:

    1. **Trim-first FFmpeg transcoding** – Computes the effective clip window
       ``[start_time, trim_end]`` *before* any transcoding.  Calls
       :func:`~src.video_engine.normalize_youtube_clip` to trim **and**
       re-encode to ``libx264/yuv420p`` in a single FFmpeg pass.  This avoids
       transcoding an entire 1-hour YouTube source when only 5–10 s are needed.
       If FFmpeg is unavailable the original file is loaded directly.
    2. **Strict media validation** – Verifies that the loaded clip has
       non-zero dimensions and a positive duration before continuing.
    3. **Trim** – When the FFmpeg trim succeeded the clip is already cut;
       ``set_duration`` pins the expected length.  Otherwise ``subclip`` is
       used as a fallback.
    4. **FPS normalisation** – Forces 24 fps for consistent compositing.
    5. **Crop-fill** – Resizes and centre-crops to exactly ``VIDEO_RES``.
    6. **Frame enhancement** – Applies :func:`_enhance_frame` via
       ``fl_image`` for contrast/colour/brightness improvement.
    7. **Explicit-background compositing** – Places the enhanced clip at
       pixel coordinates ``(0, 0)`` over a solid black
       :class:`~moviepy.editor.ColorClip` with ``use_bgclip=True`` so the
       render engine treats it as a concrete, fully-opaque image layer.
    8. **Audio assignment** – Explicitly attaches the original audio track to
       the composite to guarantee sync.

    Returns a visible brand-colour :class:`~moviepy.editor.ColorClip` if any
    step fails, logging a warning with the failure details.

    Args:
        asset_path: Path to the MP4 file.
        duration:   Planned scene duration in seconds; used to compute the
                    trim window when *end_time* is ``None`` and as the
                    ColorClip fallback duration.
        start_time: Trim start in seconds.
        end_time:   Trim end in seconds, or ``None`` to derive from
                    ``start_time + duration``.
        tmp_dir:    Directory for the intermediate trimmed file (default:
                    ``assets/tmp``).  Created automatically if absent.

    Returns:
        A :class:`~moviepy.editor.CompositeVideoClip` (or fallback
        :class:`~moviepy.editor.ColorClip`) ready for concatenation.
    """
    raw_clip = None
    try:
        # Step 1: Trim-first FFmpeg transcoding.
        # Compute the effective trim window BEFORE any heavy processing so
        # that FFmpeg only encodes the seconds actually needed by this scene.
        # For a 1-hour YouTube source, this reduces transcoding time from
        # ~minutes to ~seconds.
        trim_end = end_time if end_time is not None else (start_time + duration)

        load_path = asset_path
        already_trimmed = False
        try:
            os.makedirs(tmp_dir, exist_ok=True)
            # Use seconds with 3 decimal places (millisecond precision) in the
            # filename to avoid cache collisions between scenes with the same
            # integer seconds but different fractional parts (e.g. 1.200 vs 1.800).
            # Format: scene_trim_{start:.3f}_{end:.3f}.mp4 (e.g. scene_trim_5.000_7.000.mp4)
            trimmed_path = os.path.join(
                tmp_dir,
                f"scene_trim_{start_time:.3f}_{trim_end:.3f}.mp4",
            )
            if os.path.exists(trimmed_path):
                # Reuse the existing trimmed file to skip a redundant FFmpeg pass.
                load_path = trimmed_path
                already_trimmed = True
                logger.debug(
                    "♻️ Reusing cached trim: %s [%s→%s]",
                    trimmed_path, start_time, trim_end,
                )
            else:
                load_path = normalize_youtube_clip(
                    asset_path, trimmed_path,
                    str(float(start_time)), str(float(trim_end)),
                )
                already_trimmed = True
                logger.debug(
                    "Trim+transcode: %s [%s→%s] → %s",
                    asset_path, start_time, trim_end, load_path,
                )
        except Exception as transcode_exc:
            logger.warning(
                "⚠️ Trim+transcode skipped (%s); loading original file.", transcode_exc
            )

        # Step 2: Load and strict media validation
        raw_clip = VideoFileClip(load_path)
        # Strip residual alpha channel without relying on .without_mask() which
        # does not exist in all MoviePy versions.
        if raw_clip.mask is not None:
            raw_clip = raw_clip.set_mask(None)
        raw_clip = raw_clip.set_opacity(1.0)

        if raw_clip.w == 0 or raw_clip.h == 0 or raw_clip.duration <= 0:
            logger.warning(
                "⚠️ Invalid clip dimensions/duration: size=%s duration=%.2f for %s"
                " – using visible fallback.",
                raw_clip.size, raw_clip.duration, asset_path,
            )
            raw_clip.close()
            return ColorClip(size=VIDEO_RES, color=_BRAND_COLOR, duration=duration)

        logger.debug(
            "Clip loaded: size=%s duration=%.2fs fps=%s",
            raw_clip.size, raw_clip.duration, raw_clip.fps,
        )

        # Step 3: Pin the clip to its expected window.
        if already_trimmed:
            # FFmpeg already cut [start_time, trim_end]; the file starts at 0.
            actual_duration = min(trim_end - start_time, raw_clip.duration)
            clip = raw_clip.set_duration(actual_duration)
        else:
            # FFmpeg trim was skipped; fall back to MoviePy subclip.
            clip_start = start_time
            clip_end = min(trim_end, raw_clip.duration)
            available = clip_end - clip_start
            if available < duration:
                logger.warning(
                    "⚠️ Source asset too short: requested %.2fs but only %.2fs available"
                    " starting at %.2fs in %s – scene will be shorter than planned.",
                    duration, available, start_time, asset_path,
                )
            clip = raw_clip.subclip(clip_start, clip_end)
            actual_duration = clip.duration

        logger.debug(
            "After trim step: size=%s duration=%.2fs fps=%s",
            clip.size, clip.duration, clip.fps,
        )

        # Step 4: FPS normalisation
        clip = clip.set_fps(24)

        # Step 5: Aspect-ratio crop-fill → exact VIDEO_RES
        vid_aspect = clip.w / clip.h
        target_aspect = VIDEO_W / VIDEO_H
        if vid_aspect > target_aspect:
            clip = clip.resize(height=VIDEO_H)
        else:
            clip = clip.resize(width=VIDEO_W)
        clip = clip.crop(x_center=clip.w / 2, y_center=clip.h / 2, width=VIDEO_W, height=VIDEO_H)
        logger.debug(
            "After crop-fill: size=%s duration=%.2fs fps=%s",
            clip.size, clip.duration, clip.fps,
        )

        # Step 6: Frame enhancement
        base = clip.fl_image(_enhance_frame)

        # Validation checkpoint: log slice bounds before compositing so that
        # unexpectedly long clips are caught early.
        logger.info(
            "✅ Scene verified: Target duration %.2fs (Slice: %s–%s)",
            actual_duration, start_time, trim_end,
        )

        # Step 7: Explicit-background compositing with pixel coordinates
        # After crop-fill the clip is exactly VIDEO_RES, so (0, 0) fills the canvas.
        bg = ColorClip(VIDEO_RES, color=(0, 0, 0)).set_duration(actual_duration)
        final = CompositeVideoClip(
            [bg, base.set_position((0, 0))],
            size=VIDEO_RES,
            use_bgclip=True,
        ).set_duration(actual_duration)

        # Step 8: Attach audio to composite to guarantee sync
        return final.set_audio(base.audio)

    except Exception as exc:
        logger.warning(
            "⚠️ Failed to process MP4 %s: %s – using visible fallback.",
            asset_path, exc,
        )
        if raw_clip is not None:
            try:
                raw_clip.close()
            except Exception:
                pass
        return ColorClip(size=VIDEO_RES, color=_BRAND_COLOR, duration=duration)

# ---------------------------------------------------------------------------
# Micro-Learning helpers
# ---------------------------------------------------------------------------

# Card colour palette
_EDU_BG_COLOR = (15, 20, 50)        # Deep navy background
_EDU_TERM_COLOR = (255, 215, 0)     # Gold for the keyword term
_EDU_LABEL_COLOR = (120, 180, 255)  # Light blue for section labels
_EDU_BODY_COLOR = (240, 240, 240)   # Near-white for body text

_EDU_CARD_PADDING = 90              # px from canvas edge
_EDU_LINE_SPACING = 14              # extra px between wrapped lines


def _wrap_text_to_lines(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap *text* into lines that fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _make_educational_card(term: str, definition: str, example: str, duration: float) -> ImageClip:
    """Create a vocabulary educational card as a MoviePy :class:`ImageClip`.

    The card shows:
      - The keyword term (gold, large)
      - A "DEFINITION" section with the plain-English explanation
      - An "EXAMPLE" section with a real IT usage example

    Args:
        term:       The technical keyword.
        definition: Plain-English definition of the term.
        example:    IT usage example sentence.
        duration:   Duration of the resulting clip in seconds.

    Returns:
        A :class:`~moviepy.editor.ImageClip` of size :data:`VIDEO_RES`.
    """
    font_term = _load_hook_font(88)
    font_label = _load_hook_font(48)
    font_body = _load_hook_font(52)

    img = PIL.Image.new("RGB", VIDEO_RES, _EDU_BG_COLOR)
    draw = PIL.ImageDraw.Draw(img)

    max_text_w = VIDEO_W - 2 * _EDU_CARD_PADDING
    y = int(VIDEO_H * 0.12)

    # ── Term ────────────────────────────────────────────────────────────────
    term_upper = term.upper()
    bbox = draw.textbbox((0, 0), term_upper, font=font_term)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((VIDEO_W - tw) // 2, y),
        term_upper, font=font_term, fill=_EDU_TERM_COLOR,
        stroke_width=4, stroke_fill=(0, 0, 0),
    )
    y += th + 60

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line(
        [(_EDU_CARD_PADDING, y), (VIDEO_W - _EDU_CARD_PADDING, y)],
        fill=_EDU_TERM_COLOR, width=4,
    )
    y += 50

    # ── Definition label ────────────────────────────────────────────────────
    draw.text((_EDU_CARD_PADDING, y), "DEFINITION", font=font_label, fill=_EDU_LABEL_COLOR)
    bbox = draw.textbbox((0, 0), "DEFINITION", font=font_label)
    y += (bbox[3] - bbox[1]) + 28

    # ── Definition body ─────────────────────────────────────────────────────
    for line in _wrap_text_to_lines(draw, definition, font_body, max_text_w):
        draw.text((_EDU_CARD_PADDING, y), line, font=font_body, fill=_EDU_BODY_COLOR)
        bbox = draw.textbbox((0, 0), line, font=font_body)
        y += (bbox[3] - bbox[1]) + _EDU_LINE_SPACING
    y += 55

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line(
        [(_EDU_CARD_PADDING, y), (VIDEO_W - _EDU_CARD_PADDING, y)],
        fill=_EDU_LABEL_COLOR, width=2,
    )
    y += 50

    # ── Example label ───────────────────────────────────────────────────────
    draw.text((_EDU_CARD_PADDING, y), "EXAMPLE", font=font_label, fill=_EDU_LABEL_COLOR)
    bbox = draw.textbbox((0, 0), "EXAMPLE", font=font_label)
    y += (bbox[3] - bbox[1]) + 28

    # ── Example body ─────────────────────────────────────────────────────────
    for line in _wrap_text_to_lines(draw, example, font_body, max_text_w):
        draw.text((_EDU_CARD_PADDING, y), line, font=font_body, fill=_EDU_BODY_COLOR)
        bbox = draw.textbbox((0, 0), line, font=font_body)
        y += (bbox[3] - bbox[1]) + _EDU_LINE_SPACING

    return ImageClip(np.array(img)).set_duration(duration)


def _is_micro_learning_script(script) -> bool:
    """Return True if *script* follows the micro-learning JSON format.

    The micro-learning format is a :class:`dict` with both ``"metadata"`` and
    ``"scenes"`` keys, as opposed to the legacy flat list format.
    """
    return isinstance(script, dict) and "metadata" in script and "scenes" in script


def _get_video_duration_ffprobe(path: str, default: float = 5.0) -> float:
    """Return the duration (seconds) of *path* using ``ffprobe``.

    Reads only the container metadata so the entire video file is never
    decoded or buffered into memory – critical for 1-hour YouTube sources
    where ``VideoFileClip`` would hang or exhaust RAM.

    Falls back to *default* if ``ffprobe`` is not available or the call
    fails (e.g. network URL, corrupt file).  When the source is a URL the
    caller should set *default* to a safe cap value (e.g. the long-asset
    threshold) rather than a short placeholder.

    Args:
        path:    Local filesystem path to the video file.
        default: Value returned when ffprobe is unavailable or fails.

    Returns:
        Duration in seconds (float).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(result.stdout.strip())
        logger.debug("ffprobe duration for '%s': %.2fs", path, duration)
        return duration
    except Exception as exc:
        logger.warning(
            "⚠️ ffprobe could not determine duration for '%s' (%s) – using default %.1fs.",
            path, exc, default,
        )
        return default


async def _generate_edu_audio(text: str, voice_key: str, idx: int) -> str:
    """Generate a TTS audio file for an educational scene narration.

    Uses the ESL rate (``-5%``) and pitch (``+0Hz``) defined in
    :mod:`src.config` for maximum educational clarity.

    Args:
        text:      The text to synthesise (definition + example).
        voice_key: ``"H"`` (male ``en-US-AndrewNeural``) or ``"M"`` (female
                   ``en-US-AvaNeural``).
        idx:       Scene index used to derive a unique filename.

    Returns:
        Absolute path to the saved ``.mp3`` file.
    """
    import edge_tts

    os.makedirs(AUDIO_DIR, exist_ok=True)
    voice_name = VOICES.get(voice_key, VOICES["H"])
    path = os.path.join(AUDIO_DIR, f"edu_{idx:04d}.mp3")
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_name,
        rate=ESL_VOICE_RATE,
        pitch=ESL_VOICE_PITCH,
    )
    await communicate.save(path)
    return path


def validate_script_format(script: dict) -> dict:
    """Validate and normalise a micro-learning script dict before rendering.

    Enforces the rule that JSON timestamps are the single source of truth:

    * For ``"original"`` scenes: if ``end_time`` is missing it is set to
      ``start_time + _DEFAULT_SCENE_DURATION`` (10 s safety cap).
    * For ``"highlighted"`` and ``"review"`` scenes: missing timestamps are
      intentionally left absent so the render loop can apply *Temporal
      Inheritance* (copying the window from the nearest ``"original"`` scene).
      The 10 s default is never written for these types.
    * ``"review"`` scenes with ``duration="auto"`` are also skipped (they
      inherit from the first original scene at render time).
    * A warning is logged when ``start_time`` is absent on an ``"original"``
      scene so pipeline operators can detect malformed scripts early.
    * ``ValueError`` is raised when *both* ``start_time`` and ``end_time`` are
      explicitly provided but ``end_time <= start_time``.

    The function mutates *script* in-place **and** returns it for call-chain
    convenience.

    Args:
        script: Micro-learning script dict as produced by
                :func:`src.micro_learning_generator.generate_micro_learning_script`.

    Returns:
        The same *script* dict with any missing ``end_time`` values filled in
        for ``"original"`` scenes.
    """
    video_source = script.get("video_source", "")
    scenes = script.get("scenes", [])

    if not video_source:
        # No video source → nothing to validate (educational-only scripts).
        return script

    video_scene_types = {"original", "highlighted", "review"}
    for i, scene in enumerate(scenes):
        scene_type = scene.get("type", "")
        if scene_type not in video_scene_types:
            continue

        # highlighted and review scenes without explicit timestamps will
        # inherit their window from the nearest original scene at render time
        # (Temporal Inheritance).  Do not fill in a 10 s default here.
        # Only validate end_time > start_time if both are explicitly present.
        if scene_type in ("highlighted", "review"):
            end_time = scene.get("end_time")
            start_time = scene.get("start_time", 0)
            if isinstance(end_time, (int, float)) and end_time <= start_time:
                raise ValueError(
                    f"Scene {i} (type={scene_type}): end_time ({end_time}) must be "
                    f"greater than start_time ({start_time}). Check script.json for "
                    f"incorrect timestamps."
                )
            continue

        # For "original" scenes: apply the 10 s safety cap when end_time is absent.
        start_time = scene.get("start_time")
        if start_time is None:
            logger.warning(
                "⚠️ validate_script_format: Scene %d (type=%s) is missing "
                "'start_time'. Defaulting to 0.",
                i, scene_type,
            )
            start_time = 0
            scene["start_time"] = 0

        end_time = scene.get("end_time")
        if not isinstance(end_time, (int, float)):
            computed_end = start_time + _DEFAULT_SCENE_DURATION
            logger.warning(
                "⚠️ validate_script_format: Scene %d (type=%s) is missing a "
                "numeric 'end_time'. Defaulting to start_time + %.0fs = %.3fs.",
                i, scene_type, _DEFAULT_SCENE_DURATION, computed_end,
            )
            scene["end_time"] = computed_end
        elif end_time <= start_time:
            raise ValueError(
                f"Scene {i} (type={scene_type}): end_time ({end_time}) must be "
                f"greater than start_time ({start_time}). Check script.json for "
                f"incorrect timestamps."
            )

    return script


def _purge_stale_scene_trims(tmp_dir: str, script_path: str) -> int:
    """Delete cached ``scene_trim_*.mp4`` files that are older than *script_path*.

    When ``script.json`` is edited after a previous render the trim cache may
    contain clips produced from the old timestamps (often the 10 s defaults).
    Calling this function before a new render ensures FFmpeg re-trims each
    scene from the updated script rather than serving stale clips.

    Args:
        tmp_dir:     Directory that holds ``scene_trim_*.mp4`` cache files
                     (typically ``assets/tmp``).
        script_path: Path to the script JSON file whose modification time is
                     used as the freshness threshold.

    Returns:
        Number of stale files removed.
    """
    if not os.path.exists(script_path):
        return 0

    script_mtime = os.path.getmtime(script_path)
    removed = 0

    if not os.path.isdir(tmp_dir):
        return 0

    for fname in os.listdir(tmp_dir):
        if not (fname.startswith("scene_trim_") and fname.endswith(".mp4")):
            continue
        fpath = os.path.join(tmp_dir, fname)
        try:
            if os.path.getmtime(fpath) < script_mtime:
                os.remove(fpath)
                removed += 1
                logger.debug("🗑️ Removed stale trim cache: %s", fpath)
        except OSError as exc:
            logger.warning("⚠️ Could not remove stale trim cache %s: %s", fpath, exc)

    if removed:
        logger.info(
            "🧹 Purged %d stale scene_trim_* file(s) (script.json was modified).",
            removed,
        )
    return removed


async def main_micro_learning(script: dict) -> None:
    """Process and render a micro-learning video from a structured script dict.

    Handles the four scene types defined by the micro-learning workflow:

    * ``"original"``    – Loads ``video_source`` as a clip (no new TTS).
    * ``"highlighted"`` – Same as ``"original"``; the ``keywords`` field is
                          stored in the clip's metadata for potential downstream
                          subtitle highlighting.
    * ``"educational"`` – Renders a vocabulary card image with TTS narration
                          voiced by the *opposite* gender from the original
                          narrator (SLA best practice).
    * ``"review"``      – Repeats the ``video_source`` clip without distractions.

    Args:
        script: Micro-learning script dict (as produced by
                :func:`src.micro_learning_generator.generate_micro_learning_script`).
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate and normalise the script: fill in missing end_time values and
    # log warnings for incomplete scenes before any rendering begins.
    script = validate_script_format(script)

    metadata = script.get("metadata", {})
    raw_video_source = script.get("video_source", "")
    scenes = script.get("scenes", [])

    tone = metadata.get("tone", "INFORMATIVE")
    narrator_voice = metadata.get("narrator_voice", "H")

    # Resolve video_source: download from YouTube if it is a URL, otherwise
    # convert to an absolute local path.  This is the core fix for the
    # "Asset not found for YouTube URLs" bug.
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "tmp")
    video_source = _resolve_video_source(raw_video_source, tmp_dir=tmp_dir)

    # Determine duration of the source video via ffprobe – lightweight metadata
    # read that never loads frames.  VideoFileClip on a 1-hour YouTube source
    # would buffer the entire file and cause a hang/OOM.
    # If the source is still a URL (download failed) or the file is absent,
    # default to the long-asset threshold so any slice without an explicit
    # end_time will be capped to a safe length.
    src_duration = 5.0
    if video_source and os.path.exists(video_source):
        src_duration = _get_video_duration_ffprobe(
            video_source, default=_LONG_ASSET_THRESHOLD_SECONDS
        )
        logger.info(
            "📹 video_source resolved | Path: %s | Duration: %.2fs",
            video_source, src_duration,
        )
    else:
        logger.warning(
            "⚠️ video_source not found on disk after resolution attempt: '%s' "
            "(original: '%s') – scenes will use ColorClip fallback.",
            video_source, raw_video_source,
        )

    # Pre-pass: locate the first "original" scene so that "highlighted" and
    # "review" scenes without explicit timestamps can inherit its time window
    # (Temporal Inheritance).  Also used as the ultimate fallback reference
    # when no nearer original scene has been seen yet.
    first_original_scene = next(
        (s for s in scenes if s.get("type") == "original"), None
    )

    output_path = _output_path_for_context("esl")
    logger.info("🎬 Micro-Learning script detected → %s", output_path)

    scene_clips = []
    scene_clip_labels: list[str] = []  # scene type label for each entry in scene_clips
    edu_audio_paths = []

    # Track the most recently processed "original" scene to enable "nearest
    # original" inheritance for highlighted/review scenes that follow it.
    last_original_scene = None

    try:
        for i, scene in enumerate(scenes):
            scene_type = scene.get("type", "")

            if scene_type in ("original", "highlighted", "review"):
                # Load the video source using start_time/end_time when present.
                # Use explicit `is not None` checks so that start_time=0 is never
                # treated as falsy/absent.
                raw_start = scene.get("start_time")
                scene_start = raw_start if raw_start is not None else 0
                raw_end = scene.get("end_time")
                scene_end = raw_end if isinstance(raw_end, (int, float)) else None

                # Auto-duration: review scene with duration="auto" inherits the
                # time window of the first original scene, effectively "looping"
                # the listening context without distractions.
                if (
                    scene_type == "review"
                    and scene.get("duration") == "auto"
                    and first_original_scene is not None
                ):
                    orig_start = first_original_scene.get("start_time")
                    scene_start = orig_start if orig_start is not None else 0
                    raw_orig_end = first_original_scene.get("end_time")
                    scene_end = (
                        raw_orig_end
                        if isinstance(raw_orig_end, (int, float))
                        else None
                    )
                    logger.info(
                        "🔁 Scene %d (review auto): inheriting window from first "
                        "original scene [%ss–%ss].",
                        i, scene_start, scene_end,
                    )

                # Temporal Inheritance: if a highlighted or review scene has no
                # explicit end_time (and therefore no reliable time window), copy
                # the window from the nearest preceding original scene, falling
                # back to the first original scene anywhere in the script.
                # The 10 s Safety Cap is applied only as a last resort when no
                # original scene exists at all.
                if scene_type in ("highlighted", "review") and scene_end is None:
                    reference_scene = last_original_scene or first_original_scene
                    if reference_scene is not None:
                        ref_start = reference_scene.get("start_time")
                        ref_end = reference_scene.get("end_time")
                        # Inherit start only when it was not explicitly provided.
                        if raw_start is None:
                            scene_start = ref_start if ref_start is not None else 0
                        if isinstance(ref_end, (int, float)):
                            scene_end = ref_end
                        logger.info(
                            "Scene %d (%s): No timestamps found. "
                            "Inheriting %s-%s from reference scene.",
                            i, scene_type, scene_start, scene_end,
                        )

                # Fallback duration is used only for non-MP4 assets (e.g. ColorClip).
                # JSON timestamps are the single source of truth: use end_time - start_time
                # when end_time is explicitly provided.  When end_time is still absent
                # after Temporal Inheritance (no original scene exists anywhere), apply
                # the 10 s Safety Cap as a last resort.
                if scene_end is not None:
                    fallback_duration = scene_end - scene_start
                else:
                    fallback_duration = _DEFAULT_SCENE_DURATION
                    scene_end = scene_start + fallback_duration
                    logger.warning(
                        "⚠️ Scene %d (type=%s): no end_time resolved and no "
                        "reference original scene found; applying %.0fs Safety Cap "
                        "= %.3fs.",
                        i, scene_type, _DEFAULT_SCENE_DURATION, scene_end,
                    )

                # Track original scenes for subsequent inheritance lookups.
                if scene_type == "original":
                    last_original_scene = scene
                logger.info(
                    "[VIDEO_ENGINE] Processing %s (ID: %d): %.3fs to %.3fs (Total: %.3fs)",
                    scene_type, i, scene_start, scene_end, fallback_duration,
                )
                logger.info(
                    "[Trim] Scene %d: Source=%s | Start=%.3f | End=%.3f | Duration=%.3fs",
                    i, video_source, scene_start, scene_end, fallback_duration,
                )
                clip = _make_clip_for_scene(
                    video_source, fallback_duration, zoom_in=False,
                    start_time=scene_start, end_time=scene_end,
                )
                if scene_clips:
                    clip = clip.crossfadein(0.4)
                scene_clips.append(clip)
                scene_clip_labels.append(scene_type)
                logger.info(
                    "▶ Scene %d (%s): loaded video_source [%ss–%ss] (%.1fs)",
                    i, scene_type, scene_start,
                    scene_end if scene_end is not None else "end", clip.duration,
                )

            elif scene_type == "educational":
                term = scene.get("term", "")
                definition = scene.get("definition", "")
                # Support both "it_example" (Language Coach) and "example" (basic)
                example = scene.get("it_example") or scene.get("example", "")
                edu_voice = scene.get("narrator_voice", "M" if narrator_voice == "H" else "H")

                # Compose the narration text: "<term>. <definition> For example: <example>"
                narration = f"{term}. {definition} For example: {example}"
                audio_path = await _generate_edu_audio(narration, edu_voice, i)
                edu_audio_paths.append(audio_path)

                audio_clip = AudioFileClip(audio_path)
                edu_duration = audio_clip.duration
                audio_clip.close()

                card = _make_educational_card(term, definition, example, edu_duration)
                card = card.set_audio(AudioFileClip(audio_path))
                if scene_clips:
                    card = card.crossfadein(0.4)
                scene_clips.append(card)
                scene_clip_labels.append("educational")
                logger.info(
                    "📚 Scene %d (educational): '%s' voiced by %s (%.1fs)",
                    i, term, edu_voice, edu_duration,
                )

            else:
                logger.warning("⚠️ Unknown scene type '%s' at index %d – skipping.", scene_type, i)

        if not scene_clips:
            logger.error("❌ No scene clips were generated. Aborting.")
            return

        # Log duration of every clip before concatenation.
        # If any clip exceeds the long-asset threshold, emit a Warning and
        # cap it so the final export never inherits a runaway duration.
        capped_clips = []
        for j, (c, label) in enumerate(zip(scene_clips, scene_clip_labels)):
            clip_dur = c.duration
            logger.info(
                "🎞️ Clip[%d] type=%s duration=%.2fs",
                j, label, clip_dur,
            )
            if clip_dur > _LONG_ASSET_THRESHOLD_SECONDS:
                logger.warning(
                    "⚠️ Clip[%d] (type=%s) duration %.2fs exceeds threshold %.0fs"
                    " – capping to threshold.",
                    j, label, clip_dur, _LONG_ASSET_THRESHOLD_SECONDS,
                )
                c = c.set_duration(_LONG_ASSET_THRESHOLD_SECONDS)
            capped_clips.append(c)
        # Replace scene_clips with capped versions so the finally block
        # closes the correct (possibly updated) clip objects.
        scene_clips = capped_clips

        final_video = concatenate_videoclips(scene_clips, method="compose")

        # Optional watermark
        watermark = _make_watermark_clip(final_video.duration)
        layers = [final_video]
        if watermark:
            layers.append(watermark)
        final_video = CompositeVideoClip(layers, size=VIDEO_RES)

        # Background music (optional)
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = (
            [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))]
            if os.path.isdir(tone_music_dir) else []
        )
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        if m_files:
            bg_music = (
                AudioFileClip(random.choice(m_files))
                .fx(afx.audio_loop, duration=final_video.duration)
                .volumex(0.08)
                .audio_fadeout(2.0)
            )
            existing_audio = final_video.audio
            if existing_audio is not None:
                final_video = final_video.set_audio(
                    CompositeAudioClip([existing_audio, bg_music])
                )
            else:
                final_video = final_video.set_audio(bg_music)

        logger.info("💾 Exporting micro-learning to: %s", output_path)

        # Pre-export duration safety check: warn if total video exceeds the
        # _MAX_EXPORT_DURATION_SECONDS limit so runaway scenes are identified
        # before spending minutes on a render that was never intended.
        total_duration = sum(c.duration for c in scene_clips)
        if total_duration > _MAX_EXPORT_DURATION_SECONDS:
            bloat = [
                f"scene[{j}] type={scenes[j].get('type', '?')} {scene_clips[j].duration:.1f}s"
                for j in range(min(len(scene_clips), len(scenes)))
                if scene_clips[j].duration > _LONG_ASSET_THRESHOLD_SECONDS
            ]
            logger.warning(
                "⚠️ Pre-export duration check: total %.1fs exceeds %.0fs limit."
                " Scenes exceeding %.0fs: %s",
                total_duration,
                _MAX_EXPORT_DURATION_SECONDS,
                _LONG_ASSET_THRESHOLD_SECONDS,
                ", ".join(bloat) if bloat else "none identified",
            )

        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k",
            temp_audiofile=os.path.join(OUTPUT_DIR, "temp-audio.m4a"),
            remove_temp=True,
            threads=os.cpu_count() or 4,
            logger=None,
            verbose=False,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-vf", "setsar=1:1",
                "-movflags", "+faststart",
                "-profile:v", "high",
                "-level", "4.0",
                "-preset", "veryfast",
            ],
        )
        logger.info("✅ Micro-learning complete!")

    finally:
        try:
            final_video.close()
        except Exception:
            pass
        for clip in scene_clips:
            try:
                clip.close()
            except Exception:
                pass
        for path in edu_audio_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


async def main(target_platform: str = "Instagram") -> None:
    """Render the video described by ``script.json``.

    Args:
        target_platform: Output resolution preset.  Supported values are
            ``"Instagram"`` (1080×1920 vertical) and ``"LinkedIn"``
            (1080×1080 square).  Defaults to ``"Instagram"``.
    """
    global _TARGET_RESOLUTION
    _TARGET_RESOLUTION = _PLATFORM_RESOLUTIONS.get(target_platform, VIDEO_RES)
    logger.info("🎯 Target platform: %s → resolution %s", target_platform, _TARGET_RESOLUTION)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # All AudioFileClip / VideoFileClip objects created during this run are
    # collected here so they can be closed in the finally block even if an
    # error occurs mid-render.
    media_clips: list = []
    scene_clips: list = []
    base_video = None
    final_video = None

    try:
        _script_path = "script.json"
        with open(_script_path, "r", encoding="utf-8") as f:
            script = json.load(f)

        # Dispatch: micro-learning format (dict with metadata + scenes) vs. legacy list
        if _is_micro_learning_script(script):
            # Purge any scene_trim_* cache files that are older than script.json
            # to prevent FFmpeg from reusing clips built from outdated timestamps.
            _tmp_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "assets", "tmp"
            )
            _purge_stale_scene_trims(_tmp_dir, _script_path)
            await main_micro_learning(script)
            # Generate social copy from micro-learning scenes.
            scenes_as_list = [
                {"keyword": s.get("term", ""), "texto": s.get("definition", "")}
                for s in script.get("scenes", [])
                if s.get("term") or s.get("definition")
            ]
            _save_social_post(generate_social_copy(scenes_as_list))
            return

        context = _detect_context(script)
        output_path = _output_path_for_context(context)
        logger.info("🎬 Contexto detectado: %s → %s", context, output_path)

        tone = analyze_tone(script)
        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)

        # --- Asset error handling: fall back to brand-colour ColorClips ----
        try:
            visual_assets = get_visual_assets(script, tone=tone)
            if not visual_assets:
                raise ValueError(
                    "get_visual_assets returned an empty list; falling back to brand color clips"
                )
        except Exception as exc:
            logger.warning(
                "⚠️ get_visual_assets failed (%s) – using brand ColorClips as fallback.", exc
            )
            visual_assets = [None] * len(voice_data)

        # 1. Video Base
        zoom_in = True
        for data, asset in zip(voice_data, visual_assets):
            if asset is None:
                clip = ColorClip(
                    size=_TARGET_RESOLUTION,
                    color=_BRAND_COLOR,
                    duration=data["duracion"],
                )
            else:
                clip = _make_clip_for_scene(asset, data["duracion"], zoom_in)
            if scene_clips:
                clip = clip.crossfadein(0.4)
            scene_clips.append(clip)
            zoom_in = not zoom_in

        base_video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")

        # 2. Subtítulos y Branding
        subtitle_clips, _, sentence_start_times = generate_subtitles(
            audio_path, script_data=script, return_segment_times=True, tone=tone
        )
        hook_clip = _make_hook_clip((script[0].get("keyword") or context).strip())
        watermark = _make_watermark_clip(base_video.duration)

        layers = [base_video]
        if watermark:
            layers.append(watermark)

        final_video = CompositeVideoClip(layers + subtitle_clips + [hook_clip], size=_TARGET_RESOLUTION)

        # 3. Mezcla de Audio (CORREGIDO)
        logger.info("🔊 Mezclando capas de audio...")
        voice_audio = AudioFileClip(audio_path).volumex(1.4)
        media_clips.append(voice_audio)
        audio_layers = [voice_audio]

        # Música de Fondo
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))] if os.path.isdir(tone_music_dir) else []
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        if m_files:
            bg_music = (AudioFileClip(random.choice(m_files))
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.08).audio_fadeout(2.0))
            media_clips.append(bg_music)
            audio_layers.append(bg_music)

        # Efectos (Ambiente y Pops)
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            amb_clip = AudioFileClip(a_path).fx(afx.audio_loop, duration=final_video.duration).volumex(0.03)
            media_clips.append(amb_clip)
            audio_layers.append(amb_clip)

        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for t in sentence_start_times:
                pop_clip = AudioFileClip(p_path).volumex(0.15).set_start(t)
                media_clips.append(pop_clip)
                audio_layers.append(pop_clip)

        final_video = final_video.set_audio(CompositeAudioClip(audio_layers))

        # 4. Render Final
        logger.info("💾 Exportando a: %s", output_path)

        # Pre-export duration safety check
        if final_video.duration > _MAX_EXPORT_DURATION_SECONDS:
            logger.warning(
                "⚠️ Pre-export duration check: total %.1fs exceeds %.0fs limit.",
                final_video.duration, _MAX_EXPORT_DURATION_SECONDS,
            )

        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k",
            temp_audiofile=os.path.join(OUTPUT_DIR, "temp-audio.m4a"),
            remove_temp=True,
            threads=os.cpu_count() or 4,
            logger=None,
            verbose=False,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",      # Color compatible con todos los celulares
                "-vf", "setsar=1:1",        # Asegura que cada píxel sea un cuadrado perfecto
                "-movflags", "+faststart",  # Mueve la metadata al principio del archivo (CLAVE para redes sociales)
                "-profile:v", "high",       # Perfil de compresión que prefiere Instagram
                "-level", "4.0",            # Nivel de compatibilidad estándar
                "-preset", "veryfast",      # Fast encode preset for efficient rendering
            ]
        )
        logger.info("✅ ¡Proceso completado!")

        # 5. Social copy
        _save_social_post(generate_social_copy(script))

    finally:
        # Explicit close of every AudioFileClip / VideoFileClip to prevent
        # file-handle leaks even when an error occurs mid-render.
        for obj in media_clips:
            try:
                obj.close()
            except Exception:
                pass
        if final_video is not None:
            try:
                final_video.close()
            except Exception:
                pass
        if base_video is not None:
            try:
                base_video.close()
            except Exception:
                pass
        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass


def _save_social_post(info_file_path: str) -> None:
    """Copy the generated social copy to ``output/social_post.txt``.

    ``generate_social_copy`` already writes to ``output/INFO_POSTEO.txt``.
    This helper additionally saves the same content as ``social_post.txt``
    so users can find it alongside the rendered video without hunting for
    the original file.

    Args:
        info_file_path: Path returned by :func:`~src.copy_generator.generate_social_copy`.
    """
    social_post_path = os.path.join(OUTPUT_DIR, "social_post.txt")
    try:
        with open(info_file_path, "r", encoding="utf-8") as src_fh:
            content = src_fh.read()
        with open(social_post_path, "w", encoding="utf-8") as dst_fh:
            dst_fh.write(content)
        logger.info("📋 Social post saved to: %s", social_post_path)
    except Exception as exc:
        logger.warning("⚠️ No se pudo guardar social_post.txt: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
