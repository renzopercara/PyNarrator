import difflib
import os
import re
import logging
import numpy as np
import whisper
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from src.config import AUDIO_DIR, VIDEO_RES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN VISUAL PREMIUM ---
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES
_BASE_FONTSIZE = 95  
_Y_POS = int(_VIDEO_HEIGHT * 0.65) # Posición optimizada para 2 líneas

_ACTIVE_COLOR = (255, 215, 0, 255)     # Oro
_INACTIVE_COLOR = (255, 255, 255, 255)  # Blanco
_STROKE_COLOR = (0, 0, 0, 255)          # Borde negro
_STROKE_WIDTH = 10
_SHADOW_OFFSET = (10, 10)
_SHADOW_COLOR = (0, 0, 0, 150)

_MAX_SUBTITLE_WIDTH = int(_VIDEO_WIDTH * 0.85)
_MAX_WORDS_PER_LINE = 2  # Recomendado para Reels/TikTok
_SENTENCE_PUNCT = frozenset(".!?:;")
_SENTENCE_END_PUNCT = frozenset(".!?")

_MIN_DURATION_THRESHOLD = 0.1
_MIN_WORD_DURATION = 0.2
OFFSET_ANTICIPACION = 0.1

# Búsqueda de fuentes
_ASSETS_FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")
_FONT_CANDIDATES = [
    os.path.join(_ASSETS_FONTS_DIR, "Montserrat-Black.ttf"),
    "C:/Windows/Fonts/impact.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

# --- FUNCIONES DE UTILIDAD ---

def _ends_sentence(word: str) -> bool:
    return bool(word) and word[-1] in _SENTENCE_PUNCT

def _normalize_word(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower())

def _load_font(size: int):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _cubic_bezier_pop(t, pop_dur):
    if t >= pop_dur: return 1.0
    x = t / pop_dur
    return 1.25 - 0.25 * (3 * x**2 - 2 * x**3)

# --- LÓGICA DE ALINEACIÓN ---

def _align_texto_to_whisper(texto_words: list[str], whisper_words: list[dict]) -> list[dict]:
    texto_norm = [_normalize_word(w) for w in texto_words]
    whisper_norm = [_normalize_word(w["word"]) for w in whisper_words]

    matcher = difflib.SequenceMatcher(None, texto_norm, whisper_norm, autojunk=False)
    mapping = {}
    whisper_used = set()

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            ti, wi = block.a + offset, block.b + offset
            mapping[ti] = wi
            whisper_used.add(wi)

    # Fuzzy matching para palabras restantes
    for ti in range(len(texto_words)):
        if ti in mapping: continue
        prev_wi = next((mapping[j] for j in range(ti - 1, -1, -1) if j in mapping), -1)
        next_wi = next((mapping[j] for j in range(ti + 1, len(texto_words)) if j in mapping), len(whisper_words))
        
        candidates = [wi for wi in range(prev_wi + 1, next_wi) if wi not in whisper_used]
        best_score, best_wi = 0.0, None
        for wi in candidates:
            score = difflib.SequenceMatcher(None, texto_norm[ti], whisper_norm[wi]).ratio()
            if score > best_score:
                best_score, best_wi = score, wi
        
        if best_wi is not None and best_score >= 0.6:
            mapping[ti] = best_wi
            whisper_used.add(best_wi)

    # Interpolación de tiempos
    result_timings = []
    for i, word in enumerate(texto_words):
        if i in mapping:
            w = whisper_words[mapping[i]]
            result_timings.append({"word": word, "start": w["start"], "end": w["end"]})
        else:
            prev_end = result_timings[-1]["end"] if result_timings else 0.0
            result_timings.append({"word": word, "start": prev_end, "end": prev_end + _MIN_WORD_DURATION})
    return result_timings

def _group_words_by_punctuation(word_timings: list[dict]) -> list[list[dict]]:
    groups, current = [], []
    for word_info in word_timings:
        current.append(word_info)
        if _ends_sentence(word_info["word"]) or len(current) >= _MAX_WORDS_PER_LINE:
            groups.append(current)
            current = []
    if current: groups.append(current)
    return groups

# --- RENDERIZADO DE VIDEO ---

def _make_segment_highlight_clip(segment_words, current_idx, start, duration, y_pos):
    font_size = _BASE_FONTSIZE
    
    def get_layout(f_size):
        f_normal = _load_font(f_size)
        f_active = _load_font(int(f_size * 1.15))
        temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tokens = [w + " " for w in segment_words[:-1]] + [segment_words[-1]]
        fonts = [f_active if i == current_idx else f_normal for i in range(len(tokens))]
        
        word_metrics = []
        for t, f in zip(tokens, fonts):
            bbox = temp_draw.textbbox((0, 0), t, font=f, stroke_width=_STROKE_WIDTH)
            word_metrics.append({'w': bbox[2]-bbox[0], 'h': bbox[3]-bbox[1], 'bbox': bbox, 'token': t, 'font': f})
        
        lines, current_line, current_w = [], [], 0
        for m in word_metrics:
            if current_w + m['w'] > _MAX_SUBTITLE_WIDTH and current_line:
                lines.append(current_line)
                current_line, current_w = [m], m['w']
            else:
                current_line.append(m); current_w += m['w']
        if current_line: lines.append(current_line)
            
        final_w = max([sum(m['w'] for m in line) for line in lines]) + 60
        l_h = max(m['h'] for m in word_metrics) + 20
        return final_w, (l_h * len(lines)) + 40, lines, l_h

    tw, th, lines, l_height = get_layout(font_size)
    while tw > _VIDEO_WIDTH and font_size > 40:
        font_size -= 5
        tw, th, lines, l_height = get_layout(font_size)

    img = Image.new("RGBA", (int(tw), int(th)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    curr_y, global_idx = 20, 0
    
    for line in lines:
        line_w = sum(m['w'] for m in line)
        x_ptr = (tw - line_w) // 2
        for m in line:
            color = _ACTIVE_COLOR if global_idx == current_idx else _INACTIVE_COLOR
            draw.text((x_ptr - m['bbox'][0] + _SHADOW_OFFSET[0], curr_y - m['bbox'][1] + _SHADOW_OFFSET[1]),
                      m['token'], font=m['font'], fill=_SHADOW_COLOR, stroke_width=_STROKE_WIDTH, stroke_fill=_SHADOW_COLOR)
            draw.text((x_ptr - m['bbox'][0], curr_y - m['bbox'][1]),
                      m['token'], font=m['font'], fill=color, stroke_width=_STROKE_WIDTH, stroke_fill=_STROKE_COLOR)
            x_ptr += m['w']
            global_idx += 1
        curr_y += l_height

    pop_dur = min(0.12, duration)
    return (ImageClip(np.array(img))
            .set_start(start).set_duration(duration)
            .resize(lambda t: _cubic_bezier_pop(t, pop_dur))
            .set_position(("center", y_pos)))

def generate_subtitles(audio_path, script_data=None, return_segment_times=False, tone="ENERGICO"):
    logger.info("🎙️ Whisper: Sincronizando timestamps con texto original...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language="es", word_timestamps=True, fp16=False)

    whisper_words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    script_words = []
    if script_data:
        for item in script_data:
            script_words.extend(item.get("texto", "").split())

    if not script_words or not whisper_words:
        return ([], [], []) if return_segment_times else []

    word_timings = _align_texto_to_whisper(script_words, whisper_words)
    for wt in word_timings:
        wt["start"] = max(0.0, wt["start"] - OFFSET_ANTICIPACION)

    clips, segment_starts = [], []
    groups = _group_words_by_punctuation(word_timings)

    for group in groups:
        group_text_list = [w["word"] for w in group]
        segment_starts.append(group[0]["start"])
        for idx, word_info in enumerate(group):
            start = word_info["start"]
            duration = (group[idx + 1]["start"] - start) if idx < len(group) - 1 else (word_info["end"] - start)
            duration = max(duration, _MIN_WORD_DURATION)
            clips.append(_make_segment_highlight_clip(group_text_list, idx, start, duration, _Y_POS))

    sentence_start_times = [groups[i][0]["start"] for i in range(1, len(groups)) 
                            if groups[i-1][-1]["word"][-1] in _SENTENCE_END_PUNCT]

    return (clips, segment_starts, sentence_start_times) if return_segment_times else clips