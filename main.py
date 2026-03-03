import json
import asyncio
import logging
import os
import random
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageEnhance

# Parche para compatibilidad de Pillow con MoviePy
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from src.narrator import ArgentineNarrator
from src.image_manager import get_visual_assets
from src.subtitle_generator import generate_subtitles
from src.sentiment_analyzer import analyze_tone
from src.copy_generator import generate_social_copy
from src.config import (
    OUTPUT_DIR, AUDIO_DIR,
    MUSIC_DIR, SFX_DIR, VIDEO_RES, LOGO_PATH,
)

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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "video_final_smartbuild.mp4")

# --- CONSTANTES DE ESTILO OPTIMIZADAS (MÁS BRILLO Y COLOR) ---
_MIN_ZOOM_SCALE = 1.0
_MAX_ZOOM_SCALE = 1.3
_ZOOM_RATE_NORMAL = 0.02
_ZOOM_RATE_ENERGICO = 0.04

# Enhancement: Eliminamos el oscurecimiento. Ahora aclaramos y saturamos.
_CONTRAST_BOOST = 1.15   # +15% contraste para mayor definición
_SATURATION_BOOST = 1.25 # +25% saturación para que la obra se vea "viva"
_GAMMA_CORRECTION = 0.80 # Menos de 1.0 ACLARA los tonos medios (Exposure boost)

def _enhance_frame(frame):
    """Aclara la imagen y hace que los colores resalten sin filtros oscuros."""
    img = PIL.Image.fromarray(frame.astype(np.uint8))
    # Aplicar mejoras de brillo y contraste
    img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
    img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
    img = PIL.ImageEnhance.Brightness(img).enhance(1.1) # Boost extra de brillo
    
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, _GAMMA_CORRECTION) # Aplica el lift de exposición
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

# --- CARGA DE FUENTES ---
_HOOK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/montserrat-extrabold.ttf", # Recomendada para Smartbuild
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def _load_hook_font(size: int):
    for path in _HOOK_FONT_CANDIDATES:
        if os.path.exists(path):
            return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.0) -> ImageClip:
    """Diseño moderno: Texto con borde, sin bloques negros que tapen la imagen."""
    font = _load_hook_font(75)
    text = topic.upper()
    
    # Crear canvas transparente
    img = PIL.Image.new("RGBA", (VIDEO_W, 300), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    
    # Medir texto para centrar
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    
    # Dibujar texto con stroke (borde) para legibilidad sobre fondo claro
    x, y = (VIDEO_W - tw) // 2, (300 - th) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255), 
              stroke_width=6, stroke_fill=(0, 0, 0))
    
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(("center", 200)) # Arriba para no tapar el centro
            .crossfadeout(0.5))

def _make_clip_for_scene(asset_path, duration, zoom_in=True, zoom_rate=_ZOOM_RATE_NORMAL):
    if not asset_path or not os.path.exists(asset_path):
        return ColorClip(size=VIDEO_RES, color=(240, 240, 240), duration=duration)

    ext = os.path.splitext(asset_path)[1].lower()
    if ext == ".mp4":
        clip = VideoFileClip(asset_path)
        base = clip.subclip(0, duration) if clip.duration >= duration else clip.fx(vfx.loop, duration=duration)
        return base.fl_image(_enhance_frame)

    _TRANSITION_DUR = 0.4
    def _resize_fn(t):
        base = min(1.0 + zoom_rate * t, 1.25) if zoom_in else max(1.25 - zoom_rate * t, 1.0)
        # Efecto de zoom dinámico al final para transiciones suaves
        if (duration - t) < _TRANSITION_DUR:
            prog = 1.0 - ((duration - t) / _TRANSITION_DUR)
            base *= (1.0 + 0.05 * (prog**2))
        return base

    img_clip = (ImageClip(asset_path)
                .set_duration(duration)
                .resize(_resize_fn)
                .set_position("center"))
    
    return (CompositeVideoClip([img_clip], size=VIDEO_RES)
            .set_duration(duration)
            .fl_image(_enhance_frame))

async def main():
    logger.info("🏗️ Generando video premium para Smartbuild...")
    
    # --- AGREGÁ ESTO PARA SOLUCIONAR EL ERROR ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"📁 Carpeta creada: {OUTPUT_DIR}")

    scene_clips, subtitle_clips, sfx_audio_clips = [], [], []
    video = final_video = None

    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)
        
        tone = analyze_tone(script)
        narrator = ArgentineNarrator()
        
        # EL SECRETO: Usar 'fonetica' para el audio y 'texto' para la pantalla
        voice_data = await narrator.generate_voice_overs(script, tone=tone)
        visual_assets = get_visual_assets(script, tone=tone)

        zoom_in = True
        for i, (data, asset) in enumerate(zip(voice_data, visual_assets)):
            clip = _make_clip_for_scene(asset, data["duracion"], zoom_in, _ZOOM_RATE_NORMAL)
            # Crossfade solo si no es el primer clip
            if scene_clips:
                clip = clip.crossfadein(0.4)
            scene_clips.append(clip)
            zoom_in = not zoom_in

        # Unir clips de video
        video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
        
        # Generar subtítulos EXACTOS al campo "texto"
        subtitle_clips, _, sentence_start_times = generate_subtitles(audio_path, script_data=script, return_segment_times=True, tone=tone)

        # Hook y Composición Final
        hook_topic = (script[0].get("keyword") or "Smartbuild").strip()
        hook_clip = _make_hook_clip(hook_topic)
        
        final_video = CompositeVideoClip([video] + subtitle_clips + [hook_clip], size=VIDEO_RES)

        # Marca de Agua sutil
        if os.path.exists(LOGO_PATH):
            logo = (ImageClip(LOGO_PATH).resize(width=180).set_opacity(0.4)
                    .set_duration(final_video.duration).set_position(("right", "top")))
            final_video = CompositeVideoClip([final_video, logo], size=VIDEO_RES)

        # Audio final
        voice_audio = AudioFileClip(audio_path)

        # --- Música de fondo: Tone-based selection + Safe Mixing ---
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = []
        if os.path.isdir(tone_music_dir):
            try:
                m_files = [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir)
                           if f.endswith((".mp3", ".wav"))]
            except OSError as e:
                logger.warning("⚠️ No se pudo leer la carpeta de música '%s': %s", tone_music_dir, e)
        if not m_files:
            try:
                m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR)
                           if f.endswith((".mp3", ".wav"))]
            except OSError as e:
                logger.warning("⚠️ No se pudo leer la carpeta de música raíz '%s': %s", MUSIC_DIR, e)

        music_layers = []
        if m_files:
            music_looped = (AudioFileClip(random.choice(m_files))
                            .fx(afx.audio_loop, duration=final_video.duration)
                            .volumex(0.06)
                            .audio_fadeout(2.0))
            music_layers.append(music_looped)

        # --- Ambience de obra: loop a volumen 0.02 con fadeout ---
        ambience_layers = []
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            ambience_looped = (AudioFileClip(a_path)
                               .fx(afx.audio_loop, duration=final_video.duration)
                               .volumex(0.02)
                               .audio_fadeout(2.0))
            ambience_layers.append(ambience_looped)

        # --- SFX de Transición: 0.15s antes de cada cambio de escena ---
        transition_sfx = []
        t_path = os.path.join(SFX_DIR, "transition.mp3")
        if os.path.exists(t_path):
            cumulative = 0.0
            for data in voice_data[:-1]:  # no transition after the last scene
                cumulative += data["duracion"]
                sfx_start = max(0.0, cumulative - 0.15)
                transition_sfx.append(AudioFileClip(t_path).set_start(sfx_start))

        # --- SFX Pop: sólo al inicio de clips que siguen puntuación de oración ---
        pop_sfx = []
        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for t in sentence_start_times:
                pop_sfx.append(AudioFileClip(p_path).volumex(0.15).set_start(t))

        # --- Mezcla final normalizada ---
        all_audio_layers = [voice_audio] + music_layers + ambience_layers + transition_sfx + pop_sfx
        final_audio = CompositeAudioClip(all_audio_layers)
        final_audio.fps = 44100
        if len(all_audio_layers) > 1:
            final_audio = final_audio.fx(afx.audio_normalize)

        final_video = final_video.set_audio(final_audio)

        # Renderizar
        final_video.write_videofile(OUTPUT_PATH, fps=24, codec="libx264", audio_codec="aac", 
                                    threads=4, logger=None, verbose=False)
        
        logger.info("✅ ¡Video Smartbuild generado con éxito!")

    finally:
        # Limpieza de recursos
        for c in scene_clips + subtitle_clips + sfx_audio_clips:
            try: c.close()
            except: pass
        if video: video.close()
        if final_video: final_video.close()

if __name__ == "__main__":
    asyncio.run(main())