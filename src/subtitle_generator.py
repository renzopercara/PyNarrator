import os
import re

import numpy as np
import whisper
from spellchecker import SpellChecker
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES

_AUDIO_PATH = os.path.join(AUDIO_DIR, "final_voice.mp3")
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES  # width x height (1080 x 1920)

_FONTSIZE = 70
_TEXT_COLOR = (255, 255, 0, 255)        # yellow, fully opaque
_HIGHLIGHT_COLOR = (255, 165, 0, 255)   # orange #FFA500, for the current word
_STROKE_COLOR = (0, 0, 0, 255)          # black, fully opaque
_STROKE_WIDTH = 3
_Y_POSITION = int(_VIDEO_HEIGHT * 0.80)          # default (ENERGICO / RELAJADO)
_Y_POSITION_INFORMATIVO = int(_VIDEO_HEIGHT * 0.88)  # pinned low for INFORMATIVO

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

# ---------------------------------------------------------------------------
# Text post-processing
# ---------------------------------------------------------------------------

# Manual corrections for common Spanish transcription errors produced by
# Whisper / AI voices that run words together.  Keys are the malformed token
# (lower-case); values are the corrected replacement string (may include a
# space to split into two visible words on the same clip).
_CORRECTIONS: dict[str, str] = {
    "reargentinas": "muy argentinas",
    "reargentina": "muy argentina",
    "muiargentinas": "muy argentinas",
    "muiargentina": "muy argentina",
}

# SpellChecker instance loaded once with the Spanish dictionary.
_SPELL = SpellChecker(language="es")


# Pre-compiled regex patterns for punctuation extraction in clean_word().
_LEADING_PUNCT_RE = re.compile(r"^([^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]*)")
_TRAILING_PUNCT_RE = re.compile(r"([^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]*)$")


def clean_word(word: str, is_first: bool = False) -> str:
    """Return a cleaned version of *word* ready for subtitle display.

    Steps applied in order:

    1. Strip surrounding punctuation/whitespace.
    2. Look up the lower-cased token in the manual ``_CORRECTIONS`` dict.
    3. If not found in corrections, use :class:`spellchecker.SpellChecker`
       (Spanish) to check whether the word is known.  If unknown, replace it
       with the best candidate returned by ``correction()``.  If no candidate
       is found the original word is kept unchanged.
    4. Capitalise the first character when *is_first* is ``True`` (start of
       subtitle sequence).

    Args:
        word:     The raw token from Whisper.
        is_first: When ``True`` the result will start with an upper-case letter.

    Returns:
        The cleaned word (or corrected phrase) as a string.
    """
    # 1. Normalise
    cleaned = word.strip()
    if not cleaned:
        return cleaned

    # Preserve leading/trailing punctuation so we can reattach it.
    leading_punct = _LEADING_PUNCT_RE.match(cleaned).group(1)
    trailing_punct = _TRAILING_PUNCT_RE.search(cleaned).group(1)
    core = cleaned[len(leading_punct):len(cleaned) - len(trailing_punct)] if trailing_punct else cleaned[len(leading_punct):]

    if not core:
        return cleaned

    lower_core = core.lower()

    # 2. Manual corrections dict
    if lower_core in _CORRECTIONS:
        corrected_core = _CORRECTIONS[lower_core]
    else:
        # 3. Spell-checker: only attempt correction when the word is unknown.
        # On a successful correction the lower-cased candidate is used; when
        # no candidate is available the original casing of core is preserved.
        unknown = _SPELL.unknown([lower_core])
        if unknown:
            candidate = _SPELL.correction(lower_core)
            corrected_core = candidate if candidate else core
        else:
            corrected_core = core

    # 4. Sentence capitalisation
    if is_first:
        corrected_core = corrected_core[0].upper() + corrected_core[1:]

    return leading_punct + corrected_core + trailing_punct


def _make_text_clip(word: str, start: float, duration: float, y_pos: int = _Y_POSITION) -> "ImageClip":
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
        .set_position(("center", y_pos))
    )


def _make_segment_highlight_clip(
    segment_words: list,
    current_idx: int,
    start: float,
    duration: float,
    y_pos: int = _Y_POSITION,
) -> "ImageClip":
    """Render all *segment_words* on a single subtitle line.

    The word at *current_idx* is drawn in orange (#FFA500); all other words
    are drawn in yellow.  This produces a karaoke-style highlight effect
    where the currently-spoken word stands out from the rest of the line.

    If the total line width exceeds the video width the font is scaled down
    proportionally so everything fits on one row.

    Args:
        segment_words: All cleaned words that belong to the current segment.
        current_idx:   Index of the word that is being spoken right now.
        start:         Clip start time in seconds.
        duration:      Clip duration in seconds.

    Returns:
        A positioned :class:`~moviepy.editor.ImageClip` ready to overlay on
        the video composition.
    """
    if not segment_words:
        return _make_text_clip("", start, duration, y_pos)

    # Add a trailing space to every token except the last so words are
    # separated naturally when rendered side-by-side.
    tokens = [w + " " for w in segment_words[:-1]] + [segment_words[-1]]

    font = _FONT

    def _measure_tokens(f):
        bboxes = [
            _MEASURE_DRAW.textbbox((0, 0), t, font=f, stroke_width=_STROKE_WIDTH)
            for t in tokens
        ]
        total_w = sum(b[2] - b[0] for b in bboxes) + _STROKE_WIDTH * 2
        max_h = max(b[3] - b[1] for b in bboxes) + _STROKE_WIDTH * 2
        return bboxes, total_w, max_h

    bboxes, total_w, max_h = _measure_tokens(font)

    # Scale font down if the line would overflow the video frame
    if total_w > _VIDEO_WIDTH - 40:
        scale_factor = (_VIDEO_WIDTH - 40) / total_w
        scaled_size = max(int(_FONTSIZE * scale_factor), 20)
        font = _load_font(scaled_size)
        bboxes, total_w, max_h = _measure_tokens(font)

    img = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x = _STROKE_WIDTH
    y = _STROKE_WIDTH
    for i, (token, bbox) in enumerate(zip(tokens, bboxes)):
        color = _HIGHLIGHT_COLOR if i == current_idx else _TEXT_COLOR
        draw.text(
            (x - bbox[0], y - bbox[1]),
            token,
            font=font,
            fill=color,
            stroke_width=_STROKE_WIDTH,
            stroke_fill=_STROKE_COLOR,
        )
        x += bbox[2] - bbox[0]

    return (
        ImageClip(np.array(img), ismask=False)
        .set_start(start)
        .set_duration(duration)
        .set_position(("center", y_pos))
    )


def generate_subtitles(audio_path: str = _AUDIO_PATH, tone: str = "INFORMATIVO") -> list:
    """Generate word-level subtitle clips from *audio_path* using Whisper.

    Loads the Whisper 'base' model, transcribes the audio with
    ``word_timestamps=True``, and for each word produces a Pillow-rendered
    :class:`moviepy.editor.ImageClip` that shows **all words in the segment**
    with the currently-spoken word highlighted in orange (#FFA500) and the
    remaining words in yellow – a karaoke-style subtitle effect.

    For ``tone="INFORMATIVO"`` subtitles are pinned closer to the bottom of
    the frame (88 % of frame height) for a clean, consistent look.  For all
    other tones the default position (80 % of frame height) is used.

    This approach does **not** require ImageMagick, avoiding the
    ``OSError: Invalid Parameter`` that occurs on Windows when MoviePy's
    ``TextClip`` tries to invoke ImageMagick.

    Args:
        audio_path: Path to the MP3 audio file to transcribe.
                    Defaults to ``assets/audio/final_voice.mp3``.
        tone:       Overall script tone. ``"INFORMATIVO"`` pins subtitles lower.

    Returns:
        A list of :class:`moviepy.editor.ImageClip` objects, one per word,
        with start time, duration, and position already set.
    """
    y_pos = _Y_POSITION_INFORMATIVO if tone == "INFORMATIVO" else _Y_POSITION
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    clips: list = []
    first_word_overall = True

    for segment in result.get("segments", []):
        # Collect valid word infos for this segment
        seg_word_infos = [
            wi for wi in segment.get("words", [])
            if wi.get("word", "").strip()
        ]
        if not seg_word_infos:
            continue

        # Clean all words in the segment, capitalising only the very first
        # word of the whole transcription.
        seg_clean: list[str] = []
        for j, word_info in enumerate(seg_word_infos):
            is_first = first_word_overall and j == 0
            cleaned = clean_word(word_info["word"].strip(), is_first=is_first)
            seg_clean.append(cleaned)

        if first_word_overall and seg_clean:
            first_word_overall = False

        # Create one clip per word – the clip renders the full segment line
        # with the current word highlighted in orange.
        for idx, (word_info, _) in enumerate(zip(seg_word_infos, seg_clean)):
            start = word_info.get("start", 0.0)
            end = word_info.get("end", 0.0)
            duration = max(end - start, 0.05)  # minimum 50 ms so the clip is visible

            clips.append(
                _make_segment_highlight_clip(seg_clean, idx, start, duration, y_pos)
            )

    return clips
