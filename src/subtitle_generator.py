import os
import re
import logging
import numpy as np
import whisper
from spellchecker import SpellChecker
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES
from src.constants import CUSTOM_CORRECTIONS

# Parche de compatibilidad para Pillow 10+
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

logger = logging.getLogger(__name__)

# Configuración Visual
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES
_FONTSIZE = 75
_Y_NORMAL = int(_VIDEO_HEIGHT * 0.75) 
_Y_INFORMATIVO = int(_VIDEO_HEIGHT * 0.85) 

_TEXT_COLOR = (255, 215, 0, 255)    # Amarillo vibrante
_HIGHLIGHT_COLOR = (255, 255, 255, 255) # Blanco para palabra actual
_STROKE_COLOR = (0, 0, 0, 255)      # Borde negro
_STROKE_WIDTH = 6
_SHADOW_OFFSET = (8, 8)
_SHADOW_COLOR = (0, 0, 0, 180)      # Sombra suave

# Búsqueda de fuentes
_ASSETS_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CANDIDATES = [
    os.path.join(_ASSETS_FONTS_DIR, "Montserrat-Black.ttf"),
    os.path.join(_ASSETS_FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

_MAX_WORDS_PER_LINE = 3  # Hard limit for subtitle line length

def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# Inicialización de herramientas de texto
_SPELL = SpellChecker(language="es")
_LEADING_PUNCT_RE = re.compile(r"^([^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]*)")
_TRAILING_PUNCT_RE = re.compile(r"([^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]*)$")

def clean_word(word: str, is_first: bool = False) -> str:
    """Limpia, corrige ortografía y aplica correcciones regionales."""
    cleaned = word.strip()
    if not cleaned: return cleaned

    # Separar puntuación
    lead = _LEADING_PUNCT_RE.match(cleaned).group(1)
    trail = _TRAILING_PUNCT_RE.search(cleaned).group(1)
    core = cleaned[len(lead):len(cleaned)-len(trail)] if trail else cleaned[len(lead):]
    
    if not core: return cleaned

    # 1. Correcciones manuales (prioridad)
    lower_core = core.lower()
    if lower_core in CUSTOM_CORRECTIONS:
        core = CUSTOM_CORRECTIONS[lower_core]
    else:
        # 2. Corrector ortográfico
        if _SPELL.unknown([lower_core]):
            candidate = _SPELL.correction(lower_core)
            if candidate: core = candidate

    if is_first:
        core = core[0].upper() + core[1:]
    
    return lead + core + trail

def _cubic_bezier_pop(t, pop_dur):
    """Smooth-step (cubic Hermite) ease-in-out scale from 0.9 → 1.0 over *pop_dur* seconds."""
    if t >= pop_dur:
        return 1.0
    x = t / pop_dur
    ease = 3 * x * x - 2 * x * x * x  # smoothstep: S-curve ease-in-out
    return 0.9 + 0.1 * ease


def _make_segment_highlight_clip(segment_words, current_idx, start, duration, y_pos):
    """Renderiza una línea de subtítulo estilo Karaoke con palabra activa a 1.15×."""
    tokens = [w + " " for w in segment_words[:-1]] + [segment_words[-1]]
    font_normal = _load_font(_FONTSIZE)
    font_active = _load_font(int(_FONTSIZE * 1.15))

    # Medir tamaño total (active word uses larger font)
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    fonts = [font_active if i == current_idx else font_normal for i in range(len(tokens))]
    bboxes = [temp_draw.textbbox((0, 0), t, font=f, stroke_width=_STROKE_WIDTH)
              for t, f in zip(tokens, fonts)]
    total_w = sum(b[2]-b[0] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[0]
    max_h = max(b[3]-b[1] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[1]

    # Ajustar si es muy largo
    if total_w > _VIDEO_WIDTH - 60:
        scale = (_VIDEO_WIDTH - 60) / total_w
        base_size = max(int(_FONTSIZE * scale), 30)
        font_normal = _load_font(base_size)
        font_active = _load_font(int(base_size * 1.15))
        fonts = [font_active if i == current_idx else font_normal for i in range(len(tokens))]
        bboxes = [temp_draw.textbbox((0, 0), t, font=f, stroke_width=_STROKE_WIDTH)
                  for t, f in zip(tokens, fonts)]
        total_w = sum(b[2]-b[0] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[0]
        max_h = max(b[3]-b[1] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[1]

    img = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_offset = _STROKE_WIDTH
    for i, (token, bbox, font) in enumerate(zip(tokens, bboxes, fonts)):
        color = _HIGHLIGHT_COLOR if i == current_idx else _TEXT_COLOR
        # Sombra
        draw.text((x_offset - bbox[0] + _SHADOW_OFFSET[0], _STROKE_WIDTH - bbox[1] + _SHADOW_OFFSET[1]),
                  token, font=font, fill=_SHADOW_COLOR, stroke_width=_STROKE_WIDTH, stroke_fill=_SHADOW_COLOR)
        # Texto Principal
        draw.text((x_offset - bbox[0], _STROKE_WIDTH - bbox[1]),
                  token, font=font, fill=color, stroke_width=_STROKE_WIDTH, stroke_fill=_STROKE_COLOR)
        x_offset += bbox[2] - bbox[0]

    # Cubic-bezier ease-in-out "Pop" animation over first 0.1 s
    pop_dur = min(0.1, duration * 0.3)
    return (ImageClip(np.array(img))
            .set_start(start)
            .set_duration(duration)
            .resize(lambda t: _cubic_bezier_pop(t, pop_dur))
            .set_position(("center", y_pos)))

def generate_subtitles(audio_path, return_segment_times=False, tone="ENERGICO"):
    """Genera clips de subtítulos detectando tono y corrigiendo texto."""
    logger.info(f"🎙️ Transcribiendo audio para subtítulos (Tono: {tone})...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="es", word_timestamps=True, fp16=False)
    
    y_pos = _Y_INFORMATIVO if tone == "INFORMATIVO" else _Y_NORMAL
    clips = []
    segment_starts = []
    is_very_first = True

    for segment in result.get("segments", []):
        words_info = [w for w in segment.get("words", []) if w.get("word", "").strip()]
        if not words_info: continue

        # Limpiar palabras del segmento
        clean_words = []
        for i, w_info in enumerate(words_info):
            capital = is_very_first and i == 0
            clean_words.append(clean_word(w_info["word"], is_first=capital))

        if is_very_first: is_very_first = False

        # Agrupar en líneas de máximo _MAX_WORDS_PER_LINE palabras
        for group_start in range(0, len(words_info), _MAX_WORDS_PER_LINE):
            group_words_info = words_info[group_start:group_start + _MAX_WORDS_PER_LINE]
            group_clean_words = clean_words[group_start:group_start + _MAX_WORDS_PER_LINE]

            segment_starts.append(group_words_info[0]["start"])

            # Crear un clip por cada palabra resaltada dentro del grupo
            for idx, w_info in enumerate(group_words_info):
                start = w_info["start"]
                duration = max(w_info["end"] - start, 0.08)
                clips.append(_make_segment_highlight_clip(group_clean_words, idx, start, duration, y_pos))

    return (clips, segment_starts) if return_segment_times else clips