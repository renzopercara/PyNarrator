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
from moviepy.audio.AudioClip import AudioArrayClip
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
_DEFAULT_AUDIO_FPS = 44100  # fallback sample rate when a clip has no fps set
_VOICE_ACTIVITY_THRESHOLD = 0.10  # normalised RMS above which narrator is considered speaking

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

def _apply_highpass_filter(clip, cutoff_hz=200.0):
    """Apply an FFT-based high-pass filter to remove frequencies below cutoff_hz.

    Processes the entire clip as a numpy array and returns an AudioArrayClip
    so the result can be looped or volume-adjusted like any other audio clip.
    """
    fps = clip.fps or _DEFAULT_AUDIO_FPS
    arr = clip.to_soundarray(fps=fps)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    n = arr.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = freqs >= cutoff_hz
    filtered = np.zeros_like(arr)
    for ch in range(arr.shape[1]):
        fft_data = np.fft.rfft(arr[:, ch])
        fft_data[~mask] = 0
        filtered[:, ch] = np.fft.irfft(fft_data, n=n)
    return AudioArrayClip(np.clip(filtered, -1.0, 1.0), fps=fps).set_duration(clip.duration)


def _build_ducking_music(music_clip, voice_clip, speak_vol=0.04, silence_vol=0.15, fade_dur=0.3):
    """Return music_clip with smart ducking driven by the energy of voice_clip.

    The music volume is lowered to speak_vol while the narrator is active and
    raised to silence_vol during pauses/silences.  A convolution-based
    smoothing kernel of fade_dur seconds ensures the transitions are gradual.
    """
    analysis_fps = 50  # 50 envelope samples per second
    voice_arr = voice_clip.to_soundarray(fps=analysis_fps)
    if voice_arr.ndim > 1:
        voice_mono = np.abs(voice_arr).mean(axis=1)
    else:
        voice_mono = np.abs(voice_arr)

    # Smooth amplitude envelope with a box-kernel to simulate fade-in/out
    fade_samples = max(1, int(fade_dur * analysis_fps))
    kernel = np.ones(fade_samples, dtype=np.float32) / fade_samples
    smoothed = np.convolve(voice_mono.astype(np.float32), kernel, mode='same')

    max_val = smoothed.max()
    norm = smoothed / max_val if max_val > 0 else smoothed

    # Build target-volume array and smooth it once more for gradual transitions
    raw_target = np.where(norm > _VOICE_ACTIVITY_THRESHOLD, speak_vol, silence_vol).astype(np.float32)
    target_vol = np.convolve(raw_target, kernel, mode='same')

    def _duck(gf, t):
        idx = min(int(t * analysis_fps), len(target_vol) - 1)
        return gf(t) * float(target_vol[idx])

    return music_clip.fl(_duck)


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

        # --- Música de fondo: HPF 200 Hz + Smart Ducking ---
        m_dir = MUSIC_CORPORATE_DIR if tone == "INFORMATIVO" else MUSIC_DIR
        m_files = [os.path.join(m_dir, f) for f in os.listdir(m_dir) if f.endswith(".mp3")]

        music_layers = []
        if m_files:
            source_music = AudioFileClip(random.choice(m_files))
            music_hpf = _apply_highpass_filter(source_music)
            music_looped = music_hpf.fx(afx.audio_loop, duration=final_video.duration)
            music_ducked = _build_ducking_music(music_looped, voice_audio)
            music_layers.append(music_ducked)

        # --- Ambience de obra: HPF 200 Hz + loop a volumen 0.02 ---
        ambience_layers = []
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            source_ambience = AudioFileClip(a_path)
            ambience_hpf = _apply_highpass_filter(source_ambience)
            ambience_looped = (ambience_hpf
                               .fx(afx.audio_loop, duration=final_video.duration)
                               .volumex(0.02))
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