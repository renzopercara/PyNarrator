import os
import re
import numpy as np
import whisper
from spellchecker import SpellChecker
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES
from src.constants import CUSTOM_CORRECTIONS

_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES
_FONTSIZE = 75
_Y_NORMAL = int(_VIDEO_HEIGHT * 0.75) 
_Y_INFORMATIVO = int(_VIDEO_HEIGHT * 0.85) # Más abajo para videos serios

_HIGHLIGHT_COLOR = (255, 255, 255, 255) # Blanco para la palabra actual
_NORMAL_COLOR = (255, 215, 0, 255)    # Amarillo para el resto

def _load_font(size):
    # Intenta cargar fuentes modernas, si no usa default
    paths = ["assets/fonts/Montserrat-ExtraBold.ttf", "C:/Windows/Fonts/impact.ttf"]
    for p in paths:
        if os.path.exists(p): return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def generate_subtitles(audio_path, return_segment_times=False, tone="ENERGICO"):
    """Genera subtítulos con 'Highlight' y corrección de palabras"""
    logger = logging.getLogger(__name__)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="es", word_timestamps=True)
    
    y_pos = _Y_INFORMATIVO if tone == "INFORMATIVO" else _Y_NORMAL
    font = _load_font(_FONTSIZE)
    
    subtitle_clips = []
    segment_starts = []

    for segment in result["segments"]:
        segment_starts.append(segment["start"])
        words = segment["words"]
        
        for i, word_info in enumerate(words):
            text = word_info["word"].strip().lower()
            # Aplicar correcciones manuales (reargentinas -> re argentinas)
            text = CUSTOM_CORRECTIONS.get(text, text)
            
            # Crear frame del subtítulo
            img = Image.new("RGBA", (_VIDEO_WIDTH, 200), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Dibujar texto con borde negro
            w, h = draw.textbbox((0, 0), text, font=font)[2:]
            x = (_VIDEO_WIDTH - w) // 2
            draw.text((x, 50), text, font=font, fill=_HIGHLIGHT_COLOR, stroke_width=4, stroke_fill=(0,0,0))
            
            clip = (ImageClip(np.array(img))
                    .set_start(word_info["start"])
                    .set_duration(word_info["end"] - word_info["start"])
                    .set_position(("center", y_pos)))
            subtitle_clips.append(clip)

    if return_segment_times:
        return subtitle_clips, segment_starts
    return subtitle_clips