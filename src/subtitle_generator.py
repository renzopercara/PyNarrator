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
_STROKE_WIDTH = 4
_SHADOW_OFFSET = (5, 5)
_SHADOW_COLOR = (0, 0, 0, 180)      # Sombra suave

# Búsqueda de fuentes
_ASSETS_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CANDIDATES = [
    os.path.join(_ASSETS_FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
]

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

def _make_segment_highlight_clip(segment_words, current_idx, start, duration, y_pos):
    """Renderiza una línea de subtítulo estilo Karaoke."""
    tokens = [w + " " for w in segment_words[:-1]] + [segment_words[-1]]
    font = _load_font(_FONTSIZE)
    
    # Medir tamaño total
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bboxes = [temp_draw.textbbox((0, 0), t, font=font, stroke_width=_STROKE_WIDTH) for t in tokens]
    total_w = sum(b[2]-b[0] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[0]
    max_h = max(b[3]-b[1] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[1]

    # Ajustar si es muy largo
    if total_w > _VIDEO_WIDTH - 60:
        scale = (_VIDEO_WIDTH - 60) / total_w
        font = _load_font(max(int(_FONTSIZE * scale), 30))
        bboxes = [temp_draw.textbbox((0, 0), t, font=font, stroke_width=_STROKE_WIDTH) for t in tokens]
        total_w = sum(b[2]-b[0] for b in bboxes) + _STROKE_WIDTH * 2 + _SHADOW_OFFSET[0]

    img = Image.new("RGBA", (total_w, max_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    x_offset = _STROKE_WIDTH
    for i, (token, bbox) in enumerate(zip(tokens, bboxes)):
        color = _HIGHLIGHT_COLOR if i == current_idx else _TEXT_COLOR
        # Sombra
        draw.text((x_offset - bbox[0] + _SHADOW_OFFSET[0], _STROKE_WIDTH - bbox[1] + _SHADOW_OFFSET[1]), 
                  token, font=font, fill=_SHADOW_COLOR, stroke_width=_STROKE_WIDTH, stroke_fill=_SHADOW_COLOR)
        # Texto Principal
        draw.text((x_offset - bbox[0], _STROKE_WIDTH - bbox[1]), 
                  token, font=font, fill=color, stroke_width=_STROKE_WIDTH, stroke_fill=_STROKE_COLOR)
        x_offset += bbox[2] - bbox[0]

    return (ImageClip(np.array(img))
            .set_start(start)
            .set_duration(duration)
            .resize(lambda t: min(0.9 + 2*t, 1.0) if t < 0.05 else 1.0) # Animación "Pop" rápida
            .set_position(("center", y_pos)))

def generate_subtitles(audio_path, return_segment_times=False, tone="ENERGICO"):
    """Genera clips de subtítulos detectando tono y corrigiendo texto."""
    logger.info(f"🎙️ Transcribiendo audio para subtítulos (Tono: {tone})...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="es", word_timestamps=True)
    
    y_pos = _Y_INFORMATIVO if tone == "INFORMATIVO" else _Y_NORMAL
    clips = []
    segment_starts = []
    is_very_first = True

    for segment in result.get("segments", []):
        words_info = [w for w in segment.get("words", []) if w.get("word", "").strip()]
        if not words_info: continue

        segment_starts.append(words_info[0]["start"])
        
        # Limpiar palabras del segmento
        clean_words = []
        for i, w_info in enumerate(words_info):
            capital = is_very_first and i == 0
            clean_words.append(clean_word(w_info["word"], is_first=capital))
        
        if is_very_first: is_very_first = False

        # Crear un clip por cada palabra resaltada
        for idx, w_info in enumerate(words_info):
            start = w_info["start"]
            duration = max(w_info["end"] - start, 0.08)
            clips.append(_make_segment_highlight_clip(clean_words, idx, start, duration, y_pos))

    return (clips, segment_starts) if return_segment_times else clips