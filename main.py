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
    WATERMARK_PATH, WATERMARK_OPACITY, WATERMARK_WIDTH_PERCENT,
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
_ZOOM_RATE_NORMAL = 0.02
_CONTRAST_BOOST = 1.15   
_SATURATION_BOOST = 1.25 
_GAMMA_CORRECTION = 0.80 

def _enhance_frame(frame):
    """Mejora visual: Brillo, Contraste y Saturación."""
    img = PIL.Image.fromarray(frame.astype(np.uint8))
    img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
    img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
    img = PIL.ImageEnhance.Brightness(img).enhance(1.1) 
    
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, _GAMMA_CORRECTION) 
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

def _load_hook_font(size: int):
    candidates = ["C:/Windows/Fonts/montserrat-extrabold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    for path in candidates:
        if os.path.exists(path): return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.0) -> ImageClip:
    font = _load_hook_font(75)
    text = topic.upper()
    img = PIL.Image.new("RGBA", (VIDEO_W, 300), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((VIDEO_W - tw) // 2, (300 - th) // 2), text, font=font, 
              fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    return ImageClip(np.array(img)).set_duration(duration).set_position(("center", 200)).crossfadeout(0.5)

def _make_watermark_clip(video_duration: float) -> ImageClip:
    if not os.path.exists(WATERMARK_PATH): return None
    watermark_w = int(VIDEO_W * WATERMARK_WIDTH_PERCENT)
    clip = ImageClip(WATERMARK_PATH).resize(width=watermark_w)
    return (clip.set_opacity(WATERMARK_OPACITY).set_duration(video_duration)
            .set_position(("center", VIDEO_H - clip.size[1] - 50)))

def _make_clip_for_scene(asset_path, duration, zoom_in=True):
    if not asset_path or not os.path.exists(asset_path):
        return ColorClip(size=VIDEO_RES, color=(240, 240, 240), duration=duration)

    ext = os.path.splitext(asset_path)[1].lower()
    
    if ext == ".mp4":
        # Forzamos que el video ocupe todo el alto y recortamos el ancho sobrante
        clip = VideoFileClip(asset_path).resize(height=VIDEO_H)
        # El crop es fundamental para asegurar que no queden bordes negros
        clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=VIDEO_W, height=VIDEO_H)
        base = clip.subclip(0, duration) if clip.duration >= duration else clip.fx(vfx.loop, duration=duration)
        return base.fl_image(_enhance_frame)

    # --- Lógica para Imágenes ---
    img_clip = ImageClip(asset_path).set_duration(duration)
    
    # Redimensionamos la imagen para que CUBRA todo el canvas (evita bordes)
    w, h = img_clip.size
    aspect_ratio_target = VIDEO_W / VIDEO_H
    if w / h > aspect_ratio_target:
        img_clip = img_clip.resize(height=VIDEO_H)
    else:
        img_clip = img_clip.resize(width=VIDEO_W)

    # Aplicamos el Zoom
    def _resize_fn(t):
        # Zoom sutil: de 1.0 a 1.1 o de 1.1 a 1.0
        base = 1.0 + 0.1 * t/duration if zoom_in else 1.1 - 0.1 * t/duration
        return base

    img_clip = img_clip.resize(_resize_fn).set_position("center")
    
    # CLAVE: Forzamos el CompositeVideoClip a VIDEO_RES y le ponemos un fondo negro sólido
    # Esto elimina cualquier transparencia que Instagram pueda detectar como "error de aspecto"
    return (CompositeVideoClip([ColorClip(VIDEO_RES, color=(0,0,0)), img_clip], size=VIDEO_RES)
            .set_duration(duration)
            .fl_image(_enhance_frame))

async def main():
    logger.info("🏗️ Iniciando proceso de renderizado Smartbuild...")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)
        
        tone = analyze_tone(script)
        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)
        visual_assets = get_visual_assets(script, tone=tone)

        # 1. Video Base
        scene_clips = []
        zoom_in = True
        for data, asset in zip(voice_data, visual_assets):
            clip = _make_clip_for_scene(asset, data["duracion"], zoom_in)
            if scene_clips: clip = clip.crossfadein(0.4)
            scene_clips.append(clip)
            zoom_in = not zoom_in
        
        base_video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")

        # 2. Subtítulos y Branding
        subtitle_clips, _, sentence_start_times = generate_subtitles(
            audio_path, script_data=script, return_segment_times=True, tone=tone
        )
        hook_clip = _make_hook_clip((script[0].get("keyword") or "Smartbuild").strip())
        watermark = _make_watermark_clip(base_video.duration)
        
        layers = [base_video]
        if watermark: layers.append(watermark)
        
        final_video = CompositeVideoClip(layers + subtitle_clips + [hook_clip], size=VIDEO_RES)

        # 3. Mezcla de Audio (CORREGIDO)
        logger.info("🔊 Mezclando capas de audio...")
        voice_audio = AudioFileClip(audio_path).volumex(1.4)
        audio_layers = [voice_audio]

        # Música de Fondo
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))] if os.path.isdir(tone_music_dir) else []
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        if m_files:
            bg_music = (AudioFileClip(random.choice(m_files))
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.08).audio_fadeout(2.0))
            audio_layers.append(bg_music)

        # Efectos (Ambiente y Pops)
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            audio_layers.append(AudioFileClip(a_path).fx(afx.audio_loop, duration=final_video.duration).volumex(0.03))
        
        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for t in sentence_start_times:
                audio_layers.append(AudioFileClip(p_path).volumex(0.15).set_start(t))

        final_video = final_video.set_audio(CompositeAudioClip(audio_layers))

        # 4. Render Final
        logger.info(f"💾 Exportando a: {OUTPUT_PATH}")
        final_video.write_videofile(
            OUTPUT_PATH, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac", 
            temp_audiofile=os.path.join(OUTPUT_DIR, "temp-audio.m4a"),
            remove_temp=True, 
            threads=4, 
            logger=None, 
            verbose=False,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",      # Color compatible con todos los celulares
                "-vf", "setsar=1:1",        # Asegura que cada píxel sea un cuadrado perfecto
                "-movflags", "+faststart",  # Mueve la metadata al principio del archivo (CLAVE para redes sociales)
                "-profile:v", "high",       # Perfil de compresión que prefiere Instagram
                "-level", "4.0"             # Nivel de compatibilidad estándar
            ]
        )
        logger.info("✅ ¡Proceso completado!")

    finally:
        # Limpieza manual para evitar bloqueos de archivos
        try:
            final_video.close()
            base_video.close()
            for c in scene_clips: c.close()
        except: pass

if __name__ == "__main__":
    asyncio.run(main())