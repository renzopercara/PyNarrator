import json
import asyncio
import logging
import os
import random

from src.narrator import ArgentineNarrator
from src.image_manager import get_visual_assets
from src.subtitle_generator import generate_subtitles
from src.config import OUTPUT_DIR, AUDIO_DIR, MUSIC_DIR, VIDEO_RES, LOGO_PATH

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
_ZOOM_RATE_PER_SECOND = 0.02  # scale units added/subtracted per second


def _make_clip_for_scene(asset_path: str, duration: float, zoom_in: bool = True):
    """Return a fixed-size VideoClip for one script scene.

    - MP4 assets are trimmed or looped to match *duration*.
    - Image assets get a smooth Ken Burns zoom effect: zoom-in when *zoom_in*
      is ``True`` (scale grows from 1.0 to 1.3), zoom-out otherwise (scale
      shrinks from 1.3 to 1.0).  Alternating between scenes creates a
      dynamic visual rhythm.
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
        resize_fn = lambda t: min(_MIN_ZOOM_SCALE + _ZOOM_RATE_PER_SECOND * t, _MAX_ZOOM_SCALE)   # noqa: E731
    else:
        resize_fn = lambda t: max(_MAX_ZOOM_SCALE - _ZOOM_RATE_PER_SECOND * t, _MIN_ZOOM_SCALE)  # noqa: E731

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
    video = None
    final_video = None
    audio = None
    music_audio = None
    watermark = None

    try:
        # 1. Leer guion
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)

        # 2. Generar voces argentinas
        logger.info("🎙️ Generando voces argentinas...")
        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script)

        # 3. Descargar y procesar assets visuales
        logger.info("🖼️  Descargando assets visuales...")
        visual_assets = get_visual_assets(script)

        # 4. Crear clips de video (uno por escena)
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
        for i, (data, asset_path) in enumerate(zip(voice_data, visual_assets), start=1):
            logger.info("   [%d/%d] Procesando: %s...", i, total, data["texto"][:50])
            clip = _make_clip_for_scene(asset_path, data["duracion"], zoom_in=zoom_in)
            zoom_in = not zoom_in  # alternate direction for the next scene
            if scene_clips:  # skip crossfade on the very first clip
                clip = clip.crossfadein(0.5)
            scene_clips.append(clip)

        # 5. Concatenar todas las escenas
        logger.info("🔗 Concatenando escenas...")
        video = concatenate_videoclips(scene_clips, method="compose")

        # 6. Generar subtítulos con Whisper
        logger.info("📝 Generando subtítulos con Whisper...")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
        subtitle_clips = generate_subtitles(audio_path)

        # 7. Superponer subtítulos sobre el video
        logger.info("✍️  Superponiendo subtítulos...")
        final_video = CompositeVideoClip([video] + subtitle_clips, size=VIDEO_RES)

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

        # 9. Añadir pista de audio final (voz + música de fondo)
        logger.info("🔊 Añadiendo audio final...")
        audio = AudioFileClip(audio_path)

        # Mix background music at 10 % volume if any MP3 files exist in MUSIC_DIR
        music_files = []
        if os.path.isdir(MUSIC_DIR):
            music_files = [
                os.path.join(MUSIC_DIR, f)
                for f in os.listdir(MUSIC_DIR)
                if f.lower().endswith(".mp3")
            ]
        if music_files:
            music_path = random.choice(music_files)
            logger.info("🎵 Mezclando música de fondo: %s", os.path.basename(music_path))
            music_audio = (
                AudioFileClip(music_path)
                .fx(afx.audio_loop, duration=final_video.duration)
                .volumex(0.1)
            )
            final_audio = CompositeAudioClip([audio, music_audio])
            final_video = final_video.set_audio(final_audio)
        else:
            logger.warning(
                "No se encontraron archivos MP3 en %s. Se omite la música de fondo.", MUSIC_DIR
            )
            final_video = final_video.set_audio(audio)

        # 10. Exportar
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        total_duration = final_video.duration
        print(
            f"[DEBUG] Imágenes encontradas: {len(scene_clips)} | "
            f"Duración total del video: {total_duration:.2f}s"
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

    finally:
        # 11. Liberar memoria RAM – always runs to prevent memory leaks
        logger.info("🧹 Liberando recursos...")
        for clip in scene_clips:
            clip.close()
        for clip in subtitle_clips:
            clip.close()
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