import difflib
import os
import re
import logging
import numpy as np
import whisper
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES

# Parche de compatibilidad para Pillow 10+
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

logger = logging.getLogger(__name__)

# Configuración Visual Premium
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES
_FONTSIZE = 95  # Un poco más grande para impacto viral
_Y_POS = int(_VIDEO_HEIGHT * 0.70) # Un poco más arriba para no tapar el borde inferior

_ACTIVE_COLOR = (255, 215, 0, 255)    # Oro
_INACTIVE_COLOR = (255, 255, 255, 255) # Blanco
_STROKE_COLOR = (0, 0, 0, 255)        # Borde negro
_STROKE_WIDTH = 10
_SHADOW_OFFSET = (10, 10)
_SHADOW_COLOR = (0, 0, 0, 150)

_MAX_WORDS_PER_LINE = 3  # Fallback cap: never exceed 3 words even mid-sentence

# Punctuation marks that signal the end of a semantic unit
_SENTENCE_PUNCT = frozenset(".!?:;")

# Punctuation marks that specifically signal the end of a sentence (for SFX pop trigger)
_SENTENCE_END_PUNCT = frozenset(".!?")

# Minimum acceptable clip duration (seconds); clips shorter than this are extended
_MIN_DURATION_THRESHOLD = 0.1
_MIN_WORD_DURATION = 0.2

# Anticipation offset: show subtitles slightly before the word is spoken so
# the viewer's eye can land on it before the audio arrives (seconds).
OFFSET_ANTICIPACION = 0.1

# Búsqueda de fuentes
_ASSETS_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CANDIDATES = [
    os.path.join(_ASSETS_FONTS_DIR, "Montserrat-Black.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

def _ends_sentence(word: str) -> bool:
    """Return True if *word* ends with a sentence-breaking punctuation mark."""
    return bool(word) and word[-1] in _SENTENCE_PUNCT


def _normalize_word(w: str) -> str:
    """Strip punctuation and lowercase *w* for fuzzy comparison.

    This allows matching ``"Smartbuild"`` (texto) against ``"Smartbild"``
    (Whisper transcription of the phonetic audio) by normalising both sides
    before running the sequence alignment.
    """
    return re.sub(r"[^\w]", "", w.lower())


def _align_texto_to_whisper(
    texto_words: list[str], whisper_words: list[dict]
) -> list[dict]:
    """Map *texto_words* to Whisper word timestamps via sequence alignment.

    Whisper transcribes the *fonetica* audio (which may contain phonetic
    spellings, e.g. "Smartbild"), so its word list can differ from the
    *texto* display words ("Smartbuild").  We normalise both sequences and
    use a three-pass strategy:

    1. **Sequence alignment** – :class:`difflib.SequenceMatcher` on the full
       word lists to anchor common/identical words quickly.
    2. **Character-level fuzzy matching** – for any *texto* word that was not
       anchored in pass 1, we search the neighbouring unmatched Whisper words
       for the highest character-similarity match (Levenshtein-like ratio via
       :meth:`difflib.SequenceMatcher.ratio`), respecting monotonic ordering.
    3. **Interpolation** – timestamps for words still unmatched after both
       passes are linearly interpolated between the nearest anchored
       neighbours.

    Args:
        texto_words:   Display words from the ``"texto"`` fields of the script.
        whisper_words: Word-level dicts from Whisper (keys: ``word``,
                       ``start``, ``end``).

    Returns:
        List of dicts with keys ``word`` (from *texto_words*), ``start``,
        and ``end`` (in seconds, sourced from or interpolated between
        Whisper timestamps).
    """
    texto_norm = [_normalize_word(w) for w in texto_words]
    whisper_norm = [_normalize_word(w["word"]) for w in whisper_words]

    # --- Pass 1: sequence-level alignment (exact / near-exact word matches) ---
    matcher = difflib.SequenceMatcher(None, texto_norm, whisper_norm, autojunk=False)
    mapping: dict[int, int] = {}
    whisper_used: set[int] = set()
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        for offset in range(block.size):
            ti = block.a + offset
            wi = block.b + offset
            mapping[ti] = wi
            whisper_used.add(wi)

    # --- Pass 2: character-level fuzzy matching for still-unmatched words ---
    # Only consider Whisper words that (a) haven't been used yet and (b) lie
    # between the nearest already-anchored neighbours to preserve time order.
    _FUZZY_THRESHOLD = 0.6

    for ti in range(len(texto_words)):
        if ti in mapping:
            continue

        # Determine the Whisper index bounds from anchored neighbours
        prev_wi = next((mapping[j] for j in range(ti - 1, -1, -1) if j in mapping), -1)
        next_wi = next(
            (mapping[j] for j in range(ti + 1, len(texto_words)) if j in mapping),
            len(whisper_words),
        )
        candidates = [
            wi
            for wi in range(prev_wi + 1, next_wi)
            if wi not in whisper_used
        ]

        best_score, best_wi = 0.0, None
        for wi in candidates:
            score = difflib.SequenceMatcher(
                None, texto_norm[ti], whisper_norm[wi]
            ).ratio()
            if score > best_score:
                best_score, best_wi = score, wi

        if best_wi is not None and best_score >= _FUZZY_THRESHOLD:
            mapping[ti] = best_wi
            whisper_used.add(best_wi)

    # --- Pass 3: interpolate remaining unmatched words ---
    result: list[dict] = []
    n = len(texto_words)

    for i, word in enumerate(texto_words):
        if i in mapping:
            w = whisper_words[mapping[i]]
            result.append({"word": word, "start": w["start"], "end": w["end"]})
        else:
            # Interpolate between the nearest anchored neighbours
            prev_i = next((j for j in range(i - 1, -1, -1) if j in mapping), None)
            next_i = next((j for j in range(i + 1, n) if j in mapping), None)

            if prev_i is not None and next_i is not None:
                prev_end = whisper_words[mapping[prev_i]]["end"]
                next_start = whisper_words[mapping[next_i]]["start"]
                gap = max(next_start - prev_end, 0.0)
                steps = max(next_i - prev_i, 1)  # guard against degenerate case
                step_size = gap / steps
                start = prev_end + (i - prev_i) * step_size
                end = start + step_size
            elif prev_i is not None:
                prev_end = whisper_words[mapping[prev_i]]["end"]
                start = prev_end
                end = prev_end + _MIN_WORD_DURATION
            elif next_i is not None:
                next_start = whisper_words[mapping[next_i]]["start"]
                end = next_start
                start = max(0.0, next_start - _MIN_WORD_DURATION)
            else:
                start, end = 0.0, _MIN_WORD_DURATION

            result.append({"word": word, "start": start, "end": end})

    return result


def _group_words_by_punctuation(word_timings: list[dict]) -> list[list[dict]]:
    """Group word timings into subtitle clips using punctuation breaks.

    Rules:
    - A group ends immediately after a word whose last character is in
      ``_SENTENCE_PUNCT`` (., !, ?, :, ;).
    - Groups are also capped at ``_MAX_WORDS_PER_LINE`` words so no line is
      too long even inside a run-on sentence.
    - Groups of any length (even 1 word) are kept as their own clip.
    """
    groups: list[list[dict]] = []
    current: list[dict] = []

    for word_info in word_timings:
        current.append(word_info)
        if _ends_sentence(word_info["word"]) or len(current) >= _MAX_WORDS_PER_LINE:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _cubic_bezier_pop(t, pop_dur):
    """Efecto rebote suave para la palabra activa."""
    if t >= pop_dur: return 1.0
    x = t / pop_dur
    return 1.25 - 0.25 * (3 * x**2 - 2 * x**3) # Escala de 1.25 a 1.0

def _make_segment_highlight_clip(segment_words, current_idx, start, duration, y_pos):
    """Renderiza la línea con la palabra activa resaltada y con 'Pop'."""
    font_normal = _load_font(_FONTSIZE)
    font_active = _load_font(int(_FONTSIZE * 1.15))
    
    tokens = [w + " " for w in segment_words[:-1]] + [segment_words[-1]]
    
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    fonts = [font_active if i == current_idx else font_normal for i in range(len(tokens))]
    
    # Calcular anchos
    bboxes = [temp_draw.textbbox((0, 0), t, font=f, stroke_width=_STROKE_WIDTH) for t, f in zip(tokens, fonts)]
    total_w = sum(b[2]-b[0] for b in bboxes) + 40
    max_h = max(b[3]-b[1] for b in bboxes) + 40

    img = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_ptr = 20
    for i, (token, bbox, font) in enumerate(zip(tokens, bboxes, fonts)):
        color = _ACTIVE_COLOR if i == current_idx else _INACTIVE_COLOR
        # Sombra profunda
        draw.text((x_ptr - bbox[0] + _SHADOW_OFFSET[0], 10 - bbox[1] + _SHADOW_OFFSET[1]),
                  token, font=font, fill=_SHADOW_COLOR, stroke_width=_STROKE_WIDTH, stroke_fill=_SHADOW_COLOR)
        # Texto con borde
        draw.text((x_ptr - bbox[0], 10 - bbox[1]),
                  token, font=font, fill=color, stroke_width=_STROKE_WIDTH, stroke_fill=_STROKE_COLOR)
        x_ptr += bbox[2] - bbox[0]

    pop_dur = min(0.12, duration)
    return (ImageClip(np.array(img))
            .set_start(start)
            .set_duration(duration)
            .resize(lambda t: _cubic_bezier_pop(t, pop_dur))
            .set_position(("center", y_pos)))

def generate_subtitles(audio_path, script_data=None, return_segment_times=False, tone="ENERGICO"):
    logger.info("🎙️ Whisper: Sincronizando timestamps con texto original...")
    model = whisper.load_model("base")
    # Usamos verbose=False y word_timestamps para precisión quirúrgica
    result = model.transcribe(audio_path, language="es", word_timestamps=True, fp16=False)

    clips = []
    segment_starts = []

    # 1. Obtener todos los timestamps de Whisper
    whisper_words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            whisper_words.append(w)

    # 2. Obtener lista de palabras del SCRIPT (La verdad absoluta para subtítulos)
    script_words = []
    if script_data:
        for item in script_data:
            script_words.extend(item.get("texto", "").split())

    if not script_words or not whisper_words:
        return ([], [], []) if return_segment_times else []

    # 3. ALINEACIÓN POR SECUENCIA (Fuzzy Matching)
    # Usamos SequenceMatcher para mapear las palabras del texto de visualización
    # a los timestamps reales de Whisper, ignorando diferencias fonéticas
    # (ej. "Smartbild" en fonetica → "Smartbuild" en texto).
    word_timings = _align_texto_to_whisper(script_words, whisper_words)

    # 4. Aplicar OFFSET_ANTICIPACION para que el subtítulo aparezca
    # ligeramente antes de que el locutor pronuncie cada palabra.
    for wt in word_timings:
        wt["start"] = max(0.0, wt["start"] - OFFSET_ANTICIPACION)

    # 5. Agrupar por puntuación (y fallback a máximo de palabras) y crear clips
    groups = _group_words_by_punctuation(word_timings)

    for group in groups:
        group_text_list = [w["word"] for w in group]

        segment_starts.append(group[0]["start"])

        for idx, word_info in enumerate(group):
            start = word_info["start"]
            # Flicker prevention: each word lasts until the next word starts;
            # the last word of the group uses its own Whisper end time.
            if idx < len(group) - 1:
                duration = group[idx + 1]["start"] - start
            else:
                duration = word_info["end"] - start
            if duration < _MIN_DURATION_THRESHOLD:
                duration = _MIN_WORD_DURATION

            clips.append(_make_segment_highlight_clip(group_text_list, idx, start, duration, _Y_POS))

    # Collect start times of groups that follow a sentence-ending punctuation mark (., !, ?)
    sentence_start_times = [
        groups[i][0]["start"]
        for i in range(1, len(groups))
        if groups[i - 1][-1]["word"] and groups[i - 1][-1]["word"][-1] in _SENTENCE_END_PUNCT
    ]

    return (clips, segment_starts, sentence_start_times) if return_segment_times else clips