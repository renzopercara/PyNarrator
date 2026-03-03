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

# Enhancement constants
_CONTRAST_BOOST = 1.10   # +10% contrast
_SATURATION_BOOST = 1.10 # +10% saturation (vibrancy)
_GAMMA_CORRECTION = 0.95   # gamma < 1 lifts midtones (~5% exposure boost)


def _enhance_frame(frame):
    """Boost contrast (+10%) and saturation (+10%) with a subtle gamma correction.

    Replaces the old vignette/darkening approach: images stay bright and
    vibrant while colors are made to pop.
    """
    img = PIL.Image.fromarray(frame.astype(np.uint8))
    img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
    img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, _GAMMA_CORRECTION)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

# Font candidates shared by hook and CTA renderers
_HOOK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

def _load_hook_font(size: int):
    """Return a TrueType font at *size* pt, falling back to the bitmap default."""
    for path in _HOOK_FONT_CANDIDATES:
        if os.path.exists(path):
            return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.5) -> ImageClip:
    """Crea el título gancho inicial (Hook) con diseño mejorado"""
    strip_h = 160
    font = _load_hook_font(55)
    text = topic.upper()[:45]
    
    measure_draw = PIL.ImageDraw.Draw(PIL.Image.new("RGBA", (1, 1)))
    bbox = measure_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = PIL.Image.new("RGBA", (VIDEO_W, strip_h), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.rectangle([0, 0, VIDEO_W, strip_h], fill=(200, 0, 0, 220)) # Rojo vibrante
    
    x = max(0, (VIDEO_W - text_w) // 2)
    y = max(0, (strip_h - text_h) // 2)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(("center", (VIDEO_H - strip_h) // 2))
            .crossfadeout(0.5))

def _make_cta_clip(duration: float = 2.0) -> CompositeVideoClip:
    """Placa de cierre con llamado a la acción y zoom suave"""
    bg = ColorClip(size=VIDEO_RES, color=(0, 0, 0), duration=duration)
    cta_text = "¡SEGUINOS PARA MÁS!"
    font = _load_hook_font(70)
    
    img = PIL.Image.new("RGBA", (800, 250), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.text((10, 10), cta_text, font=font, fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0,0,0))
    
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
        base = clip.subclip(0, duration) if clip.duration >= duration else clip.fx(vfx.loop, duration=duration)
        return base.fl_image(_enhance_frame)

    # Ken Burns + smooth zoom-in transition over last 0.5 s
    _TRANSITION_DUR = 0.5

    def _resize_fn(t):
        base = min(1.0 + zoom_rate * t, 1.3) if zoom_in else max(1.3 - zoom_rate * t, 1.0)
        remaining = duration - t
        if remaining < _TRANSITION_DUR and _TRANSITION_DUR > 0:
            progress = 1.0 - (remaining / _TRANSITION_DUR)
            # cubic ease-in zoom push (x³) for outgoing transition
            ease = progress * progress * progress
            base = base * (1.0 + 0.1 * ease)
        return base

    img_clip = (ImageClip(asset_path)
                .set_duration(duration)
                .resize(_resize_fn)
                .set_position("center"))
    return (CompositeVideoClip([img_clip], size=VIDEO_RES)
            .set_duration(duration)
            .fl_image(_enhance_frame))

async def main():
    logger.info("🚀 Iniciando PyNarrator Pro...")
    
    # Inicialización para limpieza posterior
    scene_clips, subtitle_clips, sfx_audio_clips = [], [], []
    video = final_video = audio = music_audio = watermark = None

    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)
        
        tone = analyze_tone(script)
        logger.info("🎭 Tono detectado por el Director IA: %s", tone)

        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)
        visual_assets = get_visual_assets(script, tone=tone)

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
        
        # Subtítulos y tiempos para SFX
        subtitle_clips, subtitle_segment_times = generate_subtitles(audio_path, script_data=script, return_segment_times=True, tone=tone)

        # Hook inicial
        hook_topic = (script[0].get("keyword") or "Contenido").strip()[:45]
        hook_clip = _make_hook_clip(hook_topic)
        
        final_video = CompositeVideoClip([video] + subtitle_clips + [hook_clip], size=VIDEO_RES)

        # Marca de Agua
        if os.path.exists(LOGO_PATH):
            watermark = (ImageClip(LOGO_PATH).resize(width=200).set_opacity(0.5)
                        .set_duration(final_video.duration).set_position(("right", "top")))
            final_video = CompositeVideoClip([final_video, watermark], size=VIDEO_RES)

        # Color Grading
        if tone == "ENERGICO":
            final_video = final_video.fx(vfx.lum_contrast, contrast=30)
        elif tone == "RELAJADO":
            final_video = final_video.fl_image(lambda f: np.clip(f * _WARM_FILTER_RGB, 0, 255).astype(np.uint8))

        # CTA Final
        cta_clip = _make_cta_clip()
        final_video = concatenate_videoclips([final_video, cta_clip], method="compose")

        # Audio, Música y SFX
        audio = AudioFileClip(audio_path)
        m_dir = {"ENERGICO": MUSIC_FAST_DIR, "INFORMATIVO": MUSIC_CORPORATE_DIR, "RELAJADO": MUSIC_SLOW_DIR}.get(tone, MUSIC_DIR)
        
        m_files = [os.path.join(m_dir, f) for f in os.listdir(m_dir) if f.endswith(".mp3")] if os.path.exists(m_dir) else []
        if m_files:
            music_path = random.choice(m_files)
            music_audio = (AudioFileClip(music_path)
                          .fx(afx.audio_loop, duration=final_video.duration)
                          .volumex(0.1))
            base_audio = CompositeAudioClip([audio, music_audio])
        else:
            base_audio = audio

        scene_trans = []
        curr = 0.0
        for d in voice_data[:-1]:
            curr += d["duracion"]
            scene_trans.append(curr)
        
        sfx_audio_clips = _build_sfx_audio_clips(SFX_DIR, scene_trans, subtitle_segment_times)
        final_video = final_video.set_audio(CompositeAudioClip([base_audio] + sfx_audio_clips))

        # Exportación
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        final_video.write_videofile(OUTPUT_PATH, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        
        generate_social_copy(script)
        logger.info("✅ ¡Todo listo en la carpeta output!")

    finally:
        logger.info("🧹 Liberando recursos...")
        for c in scene_clips + subtitle_clips + sfx_audio_clips:
            try: c.close()
            except: pass
        if video: video.close()
        if final_video: final_video.close()
        if audio: audio.close()
        if music_audio: music_audio.close()

if __name__ == "__main__":
    asyncio.run(main())