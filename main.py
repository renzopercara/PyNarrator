import json
import asyncio
import logging
import os
import random
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

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
    MUSIC_DIR, MUSIC_FAST_DIR, MUSIC_SLOW_DIR, MUSIC_CORPORATE_DIR,
    SFX_DIR, VIDEO_RES, LOGO_PATH,
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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "video_final_py_narrator.mp4")

# Constantes de Estilo y Zoom
_MIN_ZOOM_SCALE = 1.0
_MAX_ZOOM_SCALE = 1.3
_ZOOM_RATE_NORMAL = 0.02
_ZOOM_RATE_ENERGICO = 0.05
_WARM_FILTER_RGB = np.array([1.10, 1.00, 0.85], dtype=np.float32)

_HOOK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

def _load_hook_font(size: int):
    for path in _HOOK_FONT_CANDIDATES:
        if os.path.exists(path):
            return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.5) -> ImageClip:
    """Crea el título gancho inicial (Hook)"""
    strip_h = 160
    font = _load_hook_font(55)
    text = topic.upper()[:45]
    img = PIL.Image.new("RGBA", (VIDEO_W, strip_h), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.rectangle([0, 0, VIDEO_W, strip_h], fill=(200, 0, 0, 220)) # Rojo vibrante
    draw.text((50, 45), text, font=font, fill=(255, 255, 255, 255))
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(("center", (VIDEO_H - strip_h) // 2))
            .crossfadeout(0.5))

def _make_cta_clip(duration: float = 2.0) -> CompositeVideoClip:
    """Placa de cierre con llamado a la acción"""
    bg = ColorClip(size=VIDEO_RES, color=(0, 0, 0), duration=duration)
    cta_text = "¡SEGUINOS PARA MÁS!"
    font = _load_hook_font(70)
    img = PIL.Image.new("RGBA", (800, 200), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.text((10, 10), cta_text, font=font, fill=(255, 255, 255, 255))
    text_clip = (ImageClip(np.array(img))
                 .set_duration(duration)
                 .resize(lambda t: 1.0 + 0.15 * (t / duration))
                 .set_position(("center", int(VIDEO_H * 0.40))))
    layers = [bg, text_clip]
    if os.path.exists(LOGO_PATH):
        logo = (ImageClip(LOGO_PATH)
                .resize(width=250)
                .set_duration(duration)
                .set_position(("center", int(VIDEO_H * 0.58))))
        layers.append(logo)
    return CompositeVideoClip(layers, size=VIDEO_RES)

def _build_sfx_audio_clips(sfx_dir, scene_times, subtitle_times):
    """Mezcla de efectos de sonido POP y TRANSITION"""
    clips = []
    t_path = os.path.join(sfx_dir, "transition.mp3")
    if os.path.exists(t_path):
        for t in scene_times:
            clips.append(AudioFileClip(t_path).volumex(0.2).set_start(t))
    p_path = os.path.join(sfx_dir, "pop.mp3")
    if os.path.exists(p_path):
        for t in subtitle_times:
            clips.append(AudioFileClip(p_path).volumex(0.2).set_start(t))
    return clips

def _make_clip_for_scene(asset_path, duration, zoom_in=True, zoom_rate=_ZOOM_RATE_NORMAL):
    if not asset_path or not os.path.exists(asset_path):
        return ColorClip(size=VIDEO_RES, color=(50, 50, 50), duration=duration)
    
    ext = os.path.splitext(asset_path)[1].lower()
    if ext == ".mp4":
        clip = VideoFileClip(asset_path)
        return clip.subclip(0, duration) if clip.duration >= duration else clip.fx(vfx.loop, duration=duration)

    resize_fn = (lambda t: min(1.0 + zoom_rate * t, 1.3)) if zoom_in else (lambda t: max(1.3 - zoom_rate * t, 1.0))
    img_clip = ImageClip(asset_path).set_duration(duration).resize(resize_fn).set_position("center")
    return CompositeVideoClip([img_clip], size=VIDEO_RES).set_duration(duration)

async def main():
    logger.info("🚀 Iniciando PyNarrator Pro...")
    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)
        
        tone = analyze_tone(script)
        logger.info("🎭 Tono detectado por el Director IA: %s", tone)

        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)
        visual_assets = get_visual_assets(script, tone=tone)

        scene_clips = []
        zoom_in = True
        z_rate = _ZOOM_RATE_ENERGICO if tone == "ENERGICO" else _ZOOM_RATE_NORMAL
        x_fade = {"ENERGICO": 0.0, "INFORMATIVO": 0.5, "RELAJADO": 1.0}.get(tone, 0.5)

        for i, (data, asset) in enumerate(zip(voice_data, visual_assets)):
            logger.info("🎬 Procesando escena %d...", i+1)
            clip = _make_clip_for_scene(asset, data["duracion"], zoom_in, z_rate)
            if scene_clips and x_fade > 0:
                clip = clip.crossfadein(x_fade)
            scene_clips.append(clip)
            zoom_in = not zoom_in

        video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
        
        # Generar subtítulos adaptativos
        sub_clips, sub_times = generate_subtitles(audio_path, return_segment_times=True, tone=tone)

        # Agregar Hook inicial
        hook_topic = (script[0].get("keyword") or "Contenido").strip()[:45]
        hook_clip = _make_hook_clip(hook_topic)
        
        final_video = CompositeVideoClip([video] + sub_clips + [hook_clip], size=VIDEO_RES)

        # Marca de Agua
        if os.path.exists(LOGO_PATH):
            wm = (ImageClip(LOGO_PATH).resize(width=200).set_opacity(0.5)
                  .set_duration(final_video.duration).set_position(("right", "top")))
            final_video = CompositeVideoClip([final_video, wm], size=VIDEO_RES)

        # Color Grading según Tono
        if tone == "ENERGICO":
            final_video = final_video.fx(vfx.lum_contrast, contrast=30)
        elif tone == "RELAJADO":
            final_video = final_video.fl_image(lambda f: np.clip(f * _WARM_FILTER_RGB, 0, 255).astype(np.uint8))

        # Añadir CTA final
        cta = _make_cta_clip()
        final_video = concatenate_videoclips([final_video, cta], method="compose")

        # Audio final con Música y SFX
        audio_clip = AudioFileClip(audio_path)
        m_dir = {"ENERGICO": MUSIC_FAST_DIR, "INFORMATIVO": MUSIC_CORPORATE_DIR, "RELAJADO": MUSIC_SLOW_DIR}.get(tone, MUSIC_DIR)
        
        m_files = [os.path.join(m_dir, f) for f in os.listdir(m_dir) if f.endswith(".mp3")] if os.path.exists(m_dir) else []
        if m_files:
            music_path = random.choice(m_files)
            logger.info("🎵 Música elegida: %s", os.path.basename(music_path))
            bg_music = (AudioFileClip(music_path)
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.1))
            final_audio = CompositeAudioClip([audio_clip, bg_music])
        else:
            final_audio = audio_clip

        # SFX
        scene_trans = []
        curr = 0.0
        for d in voice_data[:-1]:
            curr += d["duracion"]
            scene_trans.append(curr)
        
        sfx_clips = _build_sfx_audio_clips(SFX_DIR, scene_trans, sub_times)
        final_audio = CompositeAudioClip([final_audio] + sfx_clips)
        
        final_video = final_video.set_audio(final_audio)

        # Exportación
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_video.write_videofile(OUTPUT_PATH, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        # Generar Copy para el CM
        generate_social_copy(script)
        logger.info("✅ ¡Video e info de posteo listos en la carpeta output!")

    finally:
        logger.info("🧹 Limpieza de recursos finalizada.")

if __name__ == "__main__":
    asyncio.run(main())