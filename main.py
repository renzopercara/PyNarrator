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

# --- CONSTANTES DE ESTILO OPTIMIZADAS ---
_MIN_ZOOM_SCALE = 1.0
_MAX_ZOOM_SCALE = 1.3
_ZOOM_RATE_NORMAL = 0.02

_CONTRAST_BOOST = 1.15   
_SATURATION_BOOST = 1.25 
_GAMMA_CORRECTION = 0.80 

def _enhance_frame(frame):
    """Aclara la imagen y hace que los colores resalten."""
    img = PIL.Image.fromarray(frame.astype(np.uint8))
    img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
    img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
    img = PIL.ImageEnhance.Brightness(img).enhance(1.1) 
    
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, _GAMMA_CORRECTION) 
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

# --- CARGA DE FUENTES ---
_HOOK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/montserrat-extrabold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

def _load_hook_font(size: int):
    for path in _HOOK_FONT_CANDIDATES:
        if os.path.exists(path):
            return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.0) -> ImageClip:
    font = _load_hook_font(75)
    text = topic.upper()
    img = PIL.Image.new("RGBA", (VIDEO_W, 300), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y = (VIDEO_W - tw) // 2, (300 - th) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255), 
              stroke_width=6, stroke_fill=(0, 0, 0))
    
    return (ImageClip(np.array(img))
            .set_duration(duration)
            .set_position(("center", 200)) 
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
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_clips, subtitle_clips = [], []
    video = final_video = None

    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)
        
        tone = analyze_tone(script)
        narrator = ArgentineNarrator()
        
        voice_data = await narrator.generate_voice_overs(script, tone=tone)
        visual_assets = get_visual_assets(script, tone=tone)

        zoom_in = True
        for i, (data, asset) in enumerate(zip(voice_data, visual_assets)):
            clip = _make_clip_for_scene(asset, data["duracion"], zoom_in, _ZOOM_RATE_NORMAL)
            if scene_clips:
                clip = clip.crossfadein(0.4)
            scene_clips.append(clip)
            zoom_in = not zoom_in

        video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
        
        # Subtítulos y tiempos de oraciones
        subtitle_clips, _, sentence_start_times = generate_subtitles(audio_path, script_data=script, return_segment_times=True, tone=tone)

        hook_topic = (script[0].get("keyword") or "Smartbuild").strip()
        hook_clip = _make_hook_clip(hook_topic)
        
        final_video = CompositeVideoClip([video] + subtitle_clips + [hook_clip], size=VIDEO_RES)

        # --- MEZCLA DE AUDIO PROFESIONAL ---
        # 1. Voz Narrador (Boost de volumen para claridad)
        voice_audio = AudioFileClip(audio_path).volumex(1.2)

        # 2. Música de fondo según el tono
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = []
        if os.path.isdir(tone_music_dir):
            m_files = [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))]
        
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        audio_layers = [voice_audio]

        if m_files:
            bg_music = (AudioFileClip(random.choice(m_files))
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.06)
                        .audio_fadeout(2.0))
            audio_layers.append(bg_music)

        # 3. Ambiente de Obra
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            ambience = (AudioFileClip(a_path)
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.015)
                        .audio_fadeout(2.0))
            audio_layers.append(ambience)

        # 4. SFX Transición (Volumen bajado drásticamente para no tapar voz)
        t_path = os.path.join(SFX_DIR, "transition.mp3")
        if os.path.exists(t_path):
            cumulative = 0.0
            for data in voice_data[:-1]:
                cumulative += data["duracion"]
                sfx_start = max(0.0, cumulative - 0.15)
                audio_layers.append(AudioFileClip(t_path).volumex(0.08).set_start(sfx_start))

        # 5. SFX Pop (Volumen sutil)
        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for t in sentence_start_times:
                audio_layers.append(AudioFileClip(p_path).volumex(0.10).set_start(t))

        # Mezcla final
        final_audio = CompositeAudioClip(audio_layers)
        final_audio.fps = 44100
        if len(audio_layers) > 1:
            final_audio = final_audio.fx(afx.audio_normalize)

        final_video = final_video.set_audio(final_audio)

        # Renderizar
        final_video.write_videofile(OUTPUT_PATH, fps=24, codec="libx264", audio_codec="aac", 
                                    threads=4, logger=None, verbose=False)
        
        logger.info("✅ ¡Video Smartbuild generado con éxito!")

    finally:
        for c in scene_clips + subtitle_clips:
            try: c.close()
            except: pass
        if video: video.close()
        if final_video: final_video.close()

if __name__ == "__main__":
    asyncio.run(main())