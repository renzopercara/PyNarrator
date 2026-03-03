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

def _apply_high_pass_filter(clip, cutoff_hz=200):
    """Apply FFT-based brick-wall High-Pass Filter at cutoff_hz Hz.

    Removes bass frequencies below *cutoff_hz* from *clip* to keep the
    audio mix clear and prevent muddiness under the narrator's voice.
    """
    fps = clip.fps
    arr = clip.to_soundarray(fps=fps)
    is_mono = arr.ndim == 1
    if is_mono:
        arr = arr[:, np.newaxis]
    n = arr.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = freqs >= cutoff_hz
    filtered = np.zeros_like(arr)
    for ch in range(arr.shape[1]):
        spectrum = np.fft.rfft(arr[:, ch])
        spectrum[~mask] = 0.0
        filtered[:, ch] = np.fft.irfft(spectrum, n=n)
    return AudioArrayClip(filtered[:, 0] if is_mono else filtered, fps=fps)


def _apply_smart_ducking(clip, voice_data, total_duration,
                         duck_vol=0.04, full_vol=0.15, fade_s=0.3):
    """Apply smart (radio-style) volume ducking to *clip*.

    The volume is:
    - *duck_vol* (4 %) while the narrator is speaking.
    - *full_vol* (15 %) during silences and scene transitions.
    Transitions between the two levels are smoothed over *fade_s* seconds.
    """
    fps = clip.fps
    n = int(total_duration * fps)
    fade_n = max(1, int(fade_s * fps))

    envelope = np.full(n, full_vol, dtype=np.float64)
    fade_down = np.linspace(full_vol, duck_vol, fade_n)
    fade_up = np.linspace(duck_vol, full_vol, fade_n)

    for vd in voice_data:
        s = max(0, int(vd["start_time"] * fps))
        e = min(n, int((vd["start_time"] + vd["duracion"]) * fps))
        envelope[s:e] = duck_vol
        # Smooth fade-in (full → duck) before speech
        fd_start = max(0, s - fade_n)
        fd_len = s - fd_start
        if fd_len > 0:
            envelope[fd_start:s] = fade_down[-fd_len:]
        # Smooth fade-out (duck → full) after speech
        fu_end = min(n, e + fade_n)
        fu_len = fu_end - e
        if fu_len > 0:
            envelope[e:fu_end] = fade_up[:fu_len]

    arr = clip.to_soundarray(fps=fps)
    min_len = min(len(arr), n)
    arr = arr[:min_len].astype(np.float64)
    env = envelope[:min_len]
    if arr.ndim == 2:
        return AudioArrayClip(arr * env[:, np.newaxis], fps=fps)
    return AudioArrayClip(arr * env, fps=fps)


async def main():
    logger.info("🏗️ Generando video premium para Smartbuild...")
    
    # --- AGREGÁ ESTO PARA SOLUCIONAR EL ERROR ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"📁 Carpeta creada: {OUTPUT_DIR}")

    scene_clips, subtitle_clips, sfx_audio_clips, sfx_clips = [], [], [], []
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
        subtitle_clips, sub_times = generate_subtitles(audio_path, script_data=script, return_segment_times=True, tone=tone)

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
        total_dur = final_video.duration

        # --- 1. Background music: HPF at 200 Hz + Smart Ducking ---
        m_dir = MUSIC_CORPORATE_DIR if tone == "INFORMATIVO" else MUSIC_DIR
        m_files = [os.path.join(m_dir, f) for f in os.listdir(m_dir) if f.endswith(".mp3")]

        music_ducked = None
        if m_files:
            raw_music = (AudioFileClip(random.choice(m_files))
                         .fx(afx.audio_loop, duration=total_dur))
            music_hpf = _apply_high_pass_filter(raw_music, cutoff_hz=200)
            music_ducked = _apply_smart_ducking(music_hpf, voice_data, total_dur)
            raw_music.close()

        # --- 2. Construction ambience: looped at 0.02 + HPF at 200 Hz ---
        amb_ducked = None
        amb_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(amb_path):
            raw_amb = (AudioFileClip(amb_path)
                       .fx(afx.audio_loop, duration=total_dur))
            amb_hpf = _apply_high_pass_filter(raw_amb, cutoff_hz=200)
            amb_arr = amb_hpf.to_soundarray(fps=amb_hpf.fps)
            amb_ducked = AudioArrayClip(amb_arr * 0.02, fps=amb_hpf.fps)
            raw_amb.close()

        # --- 3. Transition SFX: 0.15 s before each scene change ---
        t_path = os.path.join(SFX_DIR, "transition.mp3")
        if os.path.exists(t_path):
            for vd in voice_data[1:]:  # skip the first scene
                t_start = max(0.0, vd["start_time"] - 0.15)
                sfx_clips.append(AudioFileClip(t_path).set_start(t_start))

        # --- 4. Sentence Pop: only after sentence-ending punctuation ---
        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for i, (vd, item) in enumerate(zip(voice_data, script)):
                if i == 0:
                    continue
                prev_text = script[i - 1].get("texto", "").rstrip()
                if prev_text and prev_text[-1] in ".!?":
                    sfx_clips.append(
                        AudioFileClip(p_path).volumex(0.15).set_start(vd["start_time"])
                    )

        # --- 5. Compose & normalize all layers ---
        audio_layers = [voice_audio]
        if music_ducked is not None:
            audio_layers.append(music_ducked)
        if amb_ducked is not None:
            audio_layers.append(amb_ducked)
        audio_layers.extend(sfx_clips)

        final_audio = CompositeAudioClip(audio_layers).fx(afx.audio_normalize)
        final_video = final_video.set_audio(final_audio)

        # Renderizar
        final_video.write_videofile(OUTPUT_PATH, fps=24, codec="libx264", audio_codec="aac", 
                                    threads=4, logger=None, verbose=False)
        
        logger.info("✅ ¡Video Smartbuild generado con éxito!")

    finally:
        # Limpieza de recursos
        for c in scene_clips + subtitle_clips + sfx_audio_clips + sfx_clips:
            try: c.close()
            except: pass
        if video: video.close()
        if final_video: final_video.close()

if __name__ == "__main__":
    asyncio.run(main())