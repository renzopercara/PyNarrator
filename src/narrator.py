import asyncio
import logging
import re
import edge_tts
from moviepy.editor import concatenate_audioclips, AudioFileClip
import os
from src.config import VOICES, AUDIO_DIR

logger = logging.getLogger(__name__)

# Voice rate adjustments per tone (edge-tts rate string format)
_TONE_VOICE_RATE: dict[str, str] = {
    "ENERGICO": "+20%",      # faster, higher energy
    "INFORMATIVO": "+0%",    # neutral default
    "INFORMATIVE": "-5%",    # ESL/educational: slower for maximum clarity
    "RELAJADO": "-15%",      # slower, calmer
}

# Voice pitch adjustments per tone
_TONE_VOICE_PITCH: dict[str, str] = {
    "ENERGICO": "+5Hz",
    "INFORMATIVO": "+0Hz",
    "INFORMATIVE": "+0Hz",   # ESL/educational: natural pitch for clarity
    "RELAJADO": "-5Hz",
}

# Per-voice-key prosody deltas applied on top of the tone-level values.
# "H" (male) → Protective/Expert: slightly deeper voice, marginally slower.
# "M" (female) → Helpful/Professional: warmer pitch, neutral rate.
_VOICE_KEY_RATE_DELTA: dict[str, int] = {
    "H": -10,  # Antes era -5, ahora -10 para que sea más pausado.
    "M": 0,
}

_VOICE_KEY_PITCH_DELTA: dict[str, int] = {
    "H": -5,   # Antes era -2, ahora -5 Hz para una voz más profunda y real.
    "M": +2,
}

# Patterns that signal a "serious / confessional" sentence.
_SERIOUS_RE = re.compile(
    r"(honestamente|la verdad|seamos claros|confieso|te digo algo"
    r"|sabés qué|sabes que|lo que nadie te cuenta|hay algo que)",
    re.IGNORECASE,
)

# Argentine fillers / openers that should be followed by a short pause.
_FILLER_RE = re.compile(
    r"\b(Che|Mirá|Mira|O sea|Dale|Bueno|Igual)\b(?![,.\?!\u2026])",
    re.IGNORECASE | re.UNICODE,
)

# Prosody adjustment constants
_QUESTION_PITCH_BOOST_HZ = 3    # +Hz for interrogative sentences
_SERIOUS_RATE_REDUCTION_PCT = -7  # % rate reduction for serious/confessional sentences
_NORMALIZED_EXCLAMATION = "!!"  # canonical form for high-energy emphasis


def _is_question(text: str) -> bool:
    """Return True if *text* is a question."""
    return "?" in text


def _is_serious(text: str) -> bool:
    """Return True if *text* reads as a serious or confessional moment."""
    return bool(_SERIOUS_RE.search(text))


def _combine_rate(base: str, delta: int) -> str:
    """Return a new edge-tts rate string after adding *delta* percentage points to *base*.

    >>> _combine_rate("+0%", -7)
    '-7%'
    >>> _combine_rate("+20%", -5)
    '+15%'
    """
    m = re.match(r"([+-]?)(\d+)%", base)
    if not m:
        return base
    sign = -1 if m.group(1) == "-" else 1
    value = sign * int(m.group(2)) + delta
    return f"{value:+d}%"


def _combine_pitch(base: str, delta: int) -> str:
    """Return a new edge-tts pitch string after adding *delta* Hz to *base*.

    >>> _combine_pitch("+0Hz", 3)
    '+3Hz'
    >>> _combine_pitch("+5Hz", -2)
    '+3Hz'
    """
    m = re.match(r"([+-]?)(\d+)Hz", base)
    if not m:
        return base
    sign = -1 if m.group(1) == "-" else 1
    value = sign * int(m.group(2)) + delta
    return f"{value:+d}Hz"


def preprocess_text(text: str) -> str:
    """
    Versión optimizada para acento argentino (voseo) y cadencia natural.
    
    Transformaciones:
    - Normaliza exclamaciones a '!!' para disparar el modo enfático de la IA.
    - Cambia rellenos (Che, Mirá) por puntos suspensivos para pausas más largas.
    - Inserta micro-pausas tras palabras agudas (voseo) como 'Comentá' o 'Mirá'.
    - Convierte puntos suspensivos a carácter Unicode para una respiración uniforme.
    """
    
    # 1. Normalizar exclamaciones (mantiene el modo 'Neural' con energía)
    text = re.sub(r"!{3,}", _NORMALIZED_EXCLAMATION, text)

    # 2. Reemplazar fillers por puntos suspensivos para una pausa más "pensada"
    # En lugar de una coma, usamos '...' para que no atropelle la frase siguiente.
    text = _FILLER_RE.sub(r"\1...", text)

    # 3. TRUCO DE ORO PARA EL VOSEO:
    # Las IAs suelen ignorar la tilde final en 'Comentá' o 'Buscá'. 
    # Al agregar una coma justo después de la vocal con tilde, obligamos a la IA 
    # a enfatizar esa sílaba y hacer una pausa respiratoria natural.
    text = re.sub(r"([áéó])\b", r"\1,", text)

    # 4. Normalizar elipsis a Unicode para una interpretación de pausa estándar
    text = re.sub(r"\.{3,}", "\u2026", text)

    # 5. Limpieza de espacios dobles que puedan haber quedado
    text = re.sub(r"\s+", " ", text).strip()

    return text


class ArgentineNarrator:
    def __init__(self):
        os.makedirs(AUDIO_DIR, exist_ok=True)

    async def generate_voice_overs(self, script_data, tone: str = "INFORMATIVO"):
        if not script_data:
            return []

        temp_files = []
        try:
            # --- 1. Synthesise one MP3 per line ---------------------------------
            base_rate = _TONE_VOICE_RATE.get(tone, "+0%")
            base_pitch = _TONE_VOICE_PITCH.get(tone, "+0Hz")
            
            for idx, item in enumerate(script_data):
                # AQUÍ ESTABA EL ERROR: Faltaba definir raw_text
                # Extraemos primero el texto sucio (priorizando fonética)
                raw_text = (item.get("fonetica") or item.get("texto", "")).strip()
                
                if not raw_text:
                    logger.warning(f"Item {idx} vacío, saltando...")
                    continue

                # Ahora pasamos ese raw_text por tu nueva función optimizada
                text = preprocess_text(raw_text)

                voice_key = item.get("voz", "H")
                voice_name = VOICES.get(voice_key, VOICES["H"])

                # --- El resto del código sigue igual ---
                rate = _combine_rate(base_rate, _VOICE_KEY_RATE_DELTA.get(voice_key, 0))
                pitch = _combine_pitch(base_pitch, _VOICE_KEY_PITCH_DELTA.get(voice_key, 0))

                if _is_question(text):
                    pitch = _combine_pitch(pitch, _QUESTION_PITCH_BOOST_HZ)
                if _is_serious(text):
                    rate = _combine_rate(rate, _SERIOUS_RATE_REDUCTION_PCT)

                temp_path = os.path.join(AUDIO_DIR, f"fragment_{idx:04d}.mp3")

                communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate, pitch=pitch)
                await communicate.save(temp_path)
                temp_files.append(temp_path)

            # --- 2. Load clips and compute start times --------------------------
            clips = []
            for path in temp_files:
                clips.append(AudioFileClip(path))

            # --- 3. Concatenate into a single file ------------------------------
            final_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
            final_clip = concatenate_audioclips(clips)
            final_clip.write_audiofile(final_path)

            results = []
            current_time = 0.0
            for idx, (item, clip) in enumerate(zip(script_data, clips)):
                results.append(
                    {
                        "texto": item.get("texto", ""),
                        "archivo_audio": final_path,
                        "duracion": clip.duration,
                        "start_time": current_time,
                    }
                )
                current_time += clip.duration

            # Close individual clips to release file handles
            for clip in clips:
                clip.close()
            final_clip.close()

            return results

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Unexpected error in generate_voice_overs: {exc}") from exc
        finally:
            # --- 4. Clean up temporary fragment files ---------------------------
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass