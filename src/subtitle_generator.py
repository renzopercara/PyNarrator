import os

import numpy as np
import whisper
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES

_AUDIO_PATH = os.path.join(AUDIO_DIR, "final_voice.mp3")
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES  # width x height (1080 x 1920)

_FONTSIZE = 70
_TEXT_COLOR = (255, 255, 0, 255)    # yellow, fully opaque
_STROKE_COLOR = (0, 0, 0, 255)      # black, fully opaque
_STROKE_WIDTH = 3
_Y_POSITION = int(_VIDEO_HEIGHT * 0.80)

# TrueType font search order: Windows → Linux → macOS → built-in fallback
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Return a TrueType font at *size* pt, falling back to the bitmap default."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# Load font and reusable measurement context once at module level.
_FONT = _load_font(_FONTSIZE)
_MEASURE_DRAW = ImageDraw.Draw(Image.new("RGBA", (1, 1)))


def _make_text_clip(word: str, start: float, duration: float) -> ImageClip:
    """Render *word* as a Pillow RGBA image and wrap it in an ImageClip.

    The image has a transparent background with yellow text and a black
    outline, avoiding any dependency on ImageMagick.
    """
    # Measure the bounding box of the rendered text (including stroke)
    bbox = _MEASURE_DRAW.textbbox((0, 0), word, font=_FONT, stroke_width=_STROKE_WIDTH)
    text_w = bbox[2] - bbox[0] + _STROKE_WIDTH * 2
    text_h = bbox[3] - bbox[1] + _STROKE_WIDTH * 2

    # Draw text onto a transparent canvas
    img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (_STROKE_WIDTH - bbox[0], _STROKE_WIDTH - bbox[1]),
        word,
        font=_FONT,
        fill=_TEXT_COLOR,
        stroke_width=_STROKE_WIDTH,
        stroke_fill=_STROKE_COLOR,
    )

    return (
        ImageClip(np.array(img), ismask=False)
        .set_start(start)
        .set_duration(duration)
        .set_position(("center", _Y_POSITION))
    )


def generate_subtitles(audio_path: str = _AUDIO_PATH) -> list:
    """Generate word-level subtitle clips from *audio_path* using Whisper.

    Loads the Whisper 'base' model, transcribes the audio with
    ``word_timestamps=True``, and converts each word into a Pillow-rendered
    :class:`moviepy.editor.ImageClip` ready to be overlaid on the final
    video composition.  This approach does **not** require ImageMagick,
    avoiding the ``OSError: Invalid Parameter`` that occurs on Windows when
    MoviePy's ``TextClip`` tries to invoke ImageMagick.

    Args:
        audio_path: Path to the MP3 audio file to transcribe.
                    Defaults to ``assets/audio/final_voice.mp3``.

    Returns:
        A list of :class:`moviepy.editor.ImageClip` objects, one per word,
        with start time, duration, and position already set.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    clips: list = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            start = word_info.get("start", 0.0)
            end = word_info.get("end", 0.0)
            duration = max(end - start, 0.05)  # minimum 50 ms so the clip is visible

            if not word:
                continue

            clips.append(_make_text_clip(word, start, duration))

    return clips
