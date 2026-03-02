import json
import asyncio
import logging
import os
import random

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

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
    SFX_DIR,
    VIDEO_RES, LOGO_PATH,
)

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
    vfx,
)
import moviepy.audio.fx.all as afx

# ---------------------------------------------------------------------------
# Logging configuration – timestamps + level + message
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = VIDEO_RES
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "video_final_py_narrator.mp4")

# Ken Burns zoom constants
_MIN_ZOOM_SCALE = 1.0   # starting scale for zoom-in / ending scale for zoom-out
_MAX_ZOOM_SCALE = 1.3   # ending scale for zoom-in / starting scale for zoom-out
_ZOOM_RATE_NORMAL = 0.02    # scale units per second – normal/relaxed pace
_ZOOM_RATE_ENERGICO = 0.05  # faster, more aggressive zoom for energetic content
_ZOOM_RATE_PER_SECOND = _ZOOM_RATE_NORMAL  # default kept for backward compat

# Warm color-grading multipliers (R, G, B): boosts red, neutral green, reduces blue
# to create a cosy lo-fi feel for RELAJADO content.
_WARM_FILTER_RGB = np.array([1.10, 1.00, 0.85], dtype=np.float32)

# Font candidates shared by hook and CTA renderers (Windows → Linux → macOS)
_HOOK_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_hook_font(size: int) -> "PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont":
    """Return a TrueType font at *size* pt, falling back to the bitmap default."""
    for path in _HOOK_FONT_CANDIDATES:
        if os.path.exists(path):
            return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()


def _make_hook_clip(topic: str, duration: float = 2.5) -> ImageClip:
    """Create a 'Breaking News' style title overlay for the first *duration* seconds.

    Renders a full-width red banner strip centred vertically on the frame with
    the *topic* text in white.  The clip fades out during the last 0.5 s so it
    disappears smoothly.

    Args:
        topic:    Short text summarising the video subject (≤ 45 chars shown).
        duration: How long the hook stays on screen (default 2.5 s).

    Returns:
        A positioned :class:`~moviepy.editor.ImageClip` ready to include in a
        ``CompositeVideoClip``.
    """
    strip_h = 160
    font = _load_hook_font(55)
    text = topic.upper()[:45]

    measure_draw = PIL.ImageDraw.Draw(PIL.Image.new("RGBA", (1, 1)))
    bbox = measure_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    img = PIL.Image.new("RGBA", (VIDEO_W, strip_h), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    # Red banner background
    draw.rectangle([0, 0, VIDEO_W, strip_h], fill=(200, 0, 0, 220))
    # Centred white text
    x = max(0, (VIDEO_W - text_w) // 2)
    y = max(0, (strip_h - text_h) // 2)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    return (
        ImageClip(np.array(img), ismask=False)
        .set_duration(duration)
        .set_position(("center", (VIDEO_H - strip_h) // 2))
        .crossfadeout(0.5)
    )


def _make_cta_clip(duration: float = 2.0) -> CompositeVideoClip:
    """Create a 2-second closing CTA plate.

    Renders a black background with '¡SEGUINOS PARA MÁS!' in white text that
    gently zooms in (scale 1.0 → 1.15).  If ``assets/logo.png`` exists it is
    displayed centred below the text.

    Args:
        duration: Duration of the CTA plate in seconds (default 2.0 s).

    Returns:
        A :class:`~moviepy.editor.CompositeVideoClip` ready to be appended to
        the main video.
    """
    bg = ColorClip(size=VIDEO_RES, color=(0, 0, 0), duration=duration)

    cta_text = "¡SEGUINOS PARA MÁS!"
    font = _load_hook_font(70)

    measure_draw = PIL.ImageDraw.Draw(PIL.Image.new("RGBA", (1, 1)))
    stroke_w = 3
    bbox = measure_draw.textbbox((0, 0), cta_text, font=font, stroke_width=stroke_w)
    text_w = bbox[2] - bbox[0] + stroke_w * 2
    text_h = bbox[3] - bbox[1] + stroke_w * 2

    img = PIL.Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.text(
        (stroke_w - bbox[0], stroke_w - bbox[1]),
        cta_text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke_w,
        stroke_fill=(0, 0, 0, 255),
    )

    # Soft zoom-in: scale grows from 1.0 to 1.15 over *duration* seconds.
    text_clip = (
        ImageClip(np.array(img), ismask=False)
        .set_duration(duration)
        .resize(lambda t: 1.0 + 0.15 * (t / duration))
        .set_position(("center", int(VIDEO_H * 0.40)))
    )

    layers = [bg, text_clip]

    if os.path.exists(LOGO_PATH):
        logo_clip = (
            ImageClip(LOGO_PATH)
            .resize(width=250)
            .set_duration(duration)
            .set_position(("center", int(VIDEO_H * 0.58)))
        )
        layers.append(logo_clip)

    return CompositeVideoClip(layers, size=VIDEO_RES)


def _build_sfx_audio_clips(
    sfx_dir: str,
    scene_transition_times: list[float],
    subtitle_segment_times: list[float],
) -> list:
    """Return a list of AudioFileClip objects for SFX mixing.

    If ``assets/sfx/transition.mp3`` exists it is placed at each value in
    *scene_transition_times* (at 20 % volume).  If ``assets/sfx/pop.mp3``
    exists it is placed at each value in *subtitle_segment_times* (at 20 %
    volume).

    Args:
        sfx_dir:                Path to the SFX assets folder.
        scene_transition_times: Timestamps (seconds) of each scene change.
        subtitle_segment_times: Timestamps (seconds) of each subtitle segment.

    Returns:
        List of positioned :class:`~moviepy.editor.AudioFileClip` objects
        ready to be added to a ``CompositeAudioClip``.
    """
    clips: list = []

    transition_path = os.path.join(sfx_dir, "transition.mp3")
    if os.path.exists(transition_path):
        for t in scene_transition_times:
            clips.append(
                AudioFileClip(transition_path).volumex(0.2).set_start(t)
            )
        logger.info(
            "🔔 SFX transition.mp3 añadido en %d punto(s) de escena.",
            len(scene_transition_times),
        )

    pop_path = os.path.join(sfx_dir, "pop.mp3")
    if os.path.exists(pop_path):
        for t in subtitle_segment_times:
            clips.append(
                AudioFileClip(pop_path).volumex(0.2).set_start(t)
            )
        logger.info(
            "🔔 SFX pop.mp3 añadido en %d segmento(s) de subtítulo.",
            len(subtitle_segment_times),
        )

    return clips


def _make_clip_for_scene(asset_path: str, duration: float, zoom_in: bool = True, zoom_rate: float = _ZOOM_RATE_NORMAL):
    """Return a fixed-size VideoClip for one script scene.

    - MP4 assets are trimmed or looped to match *duration*.
    - Image assets get a smooth Ken Burns zoom effect: zoom-in when *zoom_in*
      is ``True`` (scale grows from 1.0 to 1.3), zoom-out otherwise (scale
      shrinks from 1.3 to 1.0).  Alternating between scenes creates a
      dynamic visual rhythm.
    - *zoom_rate* controls how quickly the scale changes per second.  Use a
      higher value (e.g. ``_ZOOM_RATE_ENERGICO``) for more aggressive zooms.
    - If no valid asset is available a black fallback clip is returned.
    """
    if not asset_path or not os.path.exists(asset_path):
        return ColorClip(size=VIDEO_RES, color=(50, 50, 50), duration=duration)

    ext = os.path.splitext(asset_path)[1].lower()

    if ext == ".mp4":
        clip = VideoFileClip(asset_path)
        if clip.duration >= duration:
            clip = clip.subclip(0, duration)
        else:
            clip = clip.fx(vfx.loop, duration=duration)
        return clip

    # Static image – Ken Burns zoom effect alternating between in and out.
    # set_duration() is called explicitly after construction to satisfy
    # MoviePy's internal timeline even when duration is also given to the
    # ImageClip constructor.
    if zoom_in:
        resize_fn = lambda t: min(_MIN_ZOOM_SCALE + zoom_rate * t, _MAX_ZOOM_SCALE)   # noqa: E731
    else:
        resize_fn = lambda t: max(_MAX_ZOOM_SCALE - zoom_rate * t, _MIN_ZOOM_SCALE)  # noqa: E731

    img_clip = (
        ImageClip(asset_path)
        .set_duration(duration)
        .resize(resize_fn)
        .set_position("center")
    )
    return CompositeVideoClip([img_clip], size=VIDEO_RES).set_duration(duration)


async def main():
    logger.info("🚀 Iniciando PyNarrator...")

    scene_clips: list = []
    subtitle_clips: list = []
    sfx_audio_clips: list = []
    video = None
    final_video = None
    pre_cta_video = None
    cta_clip = None
    hook_clip = None
    audio = None
    music_audio = None
    watermark = None

    try:
        # 1. Leer guion
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)

        # 2. Analizar tono del guion (The Brain)
        tone = analyze_tone(script)
        logger.info("🎭 Tono detectado: %s", tone)

        # 3. Generar voces argentinas con parámetros adaptativos al tono
        logger.info("🎙️ Generando voces argentinas...")
        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)

        # 4. Descargar y procesar assets visuales (videos priorizados, 3 variantes)
        logger.info("🖼️  Descargando assets visuales...")
        visual_assets = get_visual_assets(script, tone=tone)

        # 5. Crear clips de video (uno por escena)
        if len(voice_data) != len(visual_assets):
            logger.warning(
                "Se generaron %d fragmentos de audio pero %d assets visuales. "
                "Se usarán %d escenas.",
                len(voice_data),
                len(visual_assets),
                min(len(voice_data), len(visual_assets)),
            )

        logger.info("🎬 Montando video...")
        total = len(voice_data)
        zoom_in = True  # scene 1 zooms in, scene 2 zooms out, alternating
        zoom_rate = _ZOOM_RATE_ENERGICO if tone == "ENERGICO" else _ZOOM_RATE_NORMAL
        # Crossfade duration: none for ENERGICO (hard cuts), longer for RELAJADO
        crossfade_duration = {
            "ENERGICO": 0.0,
            "INFORMATIVO": 0.5,
            "RELAJADO": 1.0,
        }.get(tone, 0.5)

        for i, (data, asset_path) in enumerate(zip(voice_data, visual_assets), start=1):
            logger.info("   [%d/%d] Procesando: %s...", i, total, data["texto"][:50])
            clip = _make_clip_for_scene(asset_path, data["duracion"], zoom_in=zoom_in, zoom_rate=zoom_rate)
            zoom_in = not zoom_in  # alternate direction for the next scene
            if scene_clips and crossfade_duration > 0:  # skip crossfade on the very first clip
                clip = clip.crossfadein(crossfade_duration)
            scene_clips.append(clip)

        # 5. Concatenar todas las escenas
        logger.info("🔗 Concatenando escenas...")
        video = concatenate_videoclips(scene_clips, method="compose")

        # 6. Generar subtítulos con Whisper (posición adaptativa al tono + tiempos para SFX)
        logger.info("📝 Generando subtítulos con Whisper...")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
        subtitle_clips, subtitle_segment_times = generate_subtitles(
            audio_path, return_segment_times=True, tone=tone
        )

        # 7. Superponer subtítulos y hook visual sobre el video
        logger.info("✍️  Superponiendo subtítulos...")
        hook_topic = (
            script[0].get("keyword") or script[0].get("texto") or "Contenido"
        ).strip()[:45]
        logger.info("🪝 Añadiendo hook visual: '%s'", hook_topic)
        hook_clip = _make_hook_clip(hook_topic)
        final_video = CompositeVideoClip(
            [video] + subtitle_clips + [hook_clip], size=VIDEO_RES
        )

        # 8. Añadir marca de agua (logo) en la esquina superior derecha
        if os.path.exists(LOGO_PATH):
            logger.info("🏷️  Añadiendo marca de agua...")
            watermark = (
                ImageClip(LOGO_PATH)
                .resize(width=200)
                .set_opacity(0.5)
                .set_duration(final_video.duration)
                .set_position(("right", "top"))
            )
            final_video = CompositeVideoClip([final_video, watermark], size=VIDEO_RES)
        else:
            logger.warning("Logo no encontrado en %s. Se omite la marca de agua.", LOGO_PATH)

        # 9. Color Grading adaptativo al tono
        if tone == "ENERGICO":
            logger.info("🎨 Aplicando color grading (contraste alto) para tono ENERGICO...")
            final_video = final_video.fx(vfx.lum_contrast, contrast=30, contrast_thr=127)
        elif tone == "RELAJADO":
            logger.info("🎨 Aplicando color grading (filtro cálido) para tono RELAJADO...")
            final_video = final_video.fl_image(
                lambda frame: np.clip(frame * _WARM_FILTER_RGB, 0, 255).astype(np.uint8)
            )

        # 9.5. Añadir placa de cierre CTA (2 segundos al final)
        logger.info("📢 Añadiendo placa de cierre CTA...")
        cta_clip = _make_cta_clip()
        pre_cta_video = final_video
        final_video = concatenate_videoclips([pre_cta_video, cta_clip], method="compose")

        # 9. Añadir pista de audio final (voz + música de fondo + SFX)
        logger.info("🔊 Añadiendo audio final...")
        audio = AudioFileClip(audio_path)

        # Mix background music at 10 % volume, preferring the tone-matched subdirectory
        tone_music_dirs = {
            "ENERGICO": MUSIC_FAST_DIR,
            "INFORMATIVO": MUSIC_CORPORATE_DIR,
            "RELAJADO": MUSIC_SLOW_DIR,
        }
        preferred_music_dir = tone_music_dirs.get(tone, MUSIC_DIR)

        music_files: list[str] = []
        for music_dir in (preferred_music_dir, MUSIC_DIR):
            if os.path.isdir(music_dir):
                music_files = [
                    os.path.join(music_dir, f)
                    for f in os.listdir(music_dir)
                    if f.lower().endswith(".mp3")
                ]
            if music_files:
                break
        if music_files:
            music_path = random.choice(music_files)
            logger.info("🎵 Mezclando música de fondo (%s): %s", tone, os.path.basename(music_path))
            music_audio = (
                AudioFileClip(music_path)
                .fx(afx.audio_loop, duration=final_video.duration)
                .volumex(0.1)
            )
            base_audio = CompositeAudioClip([audio, music_audio])
        else:
            logger.warning(
                "No se encontraron archivos MP3 en %s ni en %s. Se omite la música de fondo.",
                preferred_music_dir, MUSIC_DIR,
            )
            base_audio = audio

        # Mix SFX: transition.mp3 at scene changes and pop.mp3 at subtitle segments
        # Compute scene transition times (cumulative voice durations, skip last scene)
        scene_transition_times: list[float] = []
        cumulative = 0.0
        for data in voice_data[:-1]:
            cumulative += data["duracion"]
            scene_transition_times.append(cumulative)

        sfx_audio_clips = _build_sfx_audio_clips(
            SFX_DIR, scene_transition_times, subtitle_segment_times
        )
        if sfx_audio_clips:
            final_audio = CompositeAudioClip([base_audio] + sfx_audio_clips)
        else:
            final_audio = base_audio

        final_video = final_video.set_audio(final_audio)

        # 10. Exportar
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        total_duration = final_video.duration
        print(
            f"[DEBUG] Imágenes encontradas: {len(scene_clips)} | "
            f"Duración total del video: {total_duration:.2f}s | "
            f"Tono: {tone}"
        )
        logger.info("💾 Exportando a %s...", OUTPUT_PATH)
        final_video.write_videofile(
            OUTPUT_PATH,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            ffmpeg_params=["-profile:v", "high"],
            verbose=False,
            logger="bar",
        )

        logger.info("✅ ¡Video exportado exitosamente!")
        logger.info("   📁 Ubicación: %s", OUTPUT_PATH)

        # 11. Generar copy para redes sociales
        logger.info("📋 Generando copy para redes sociales...")
        copy_path = generate_social_copy(script)
        logger.info("   📁 Copy guardado en: %s", copy_path)

    finally:
        # 12. Liberar memoria RAM – always runs to prevent memory leaks
        logger.info("🧹 Liberando recursos...")
        for clip in scene_clips:
            clip.close()
        for clip in subtitle_clips:
            clip.close()
        for clip in sfx_audio_clips:
            clip.close()
        if hook_clip is not None:
            hook_clip.close()
        if cta_clip is not None:
            cta_clip.close()
        if pre_cta_video is not None:
            pre_cta_video.close()
        if video is not None:
            video.close()
        if final_video is not None:
            final_video.close()
        if audio is not None:
            audio.close()
        if music_audio is not None:
            music_audio.close()
        if watermark is not None:
            watermark.close()


if __name__ == "__main__":
    asyncio.run(main())