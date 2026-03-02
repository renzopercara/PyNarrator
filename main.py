import json
import asyncio
import logging
import os

from src.narrator import ArgentineNarrator
from src.image_manager import get_visual_assets
from src.subtitle_generator import generate_subtitles
from src.config import OUTPUT_DIR, AUDIO_DIR, VIDEO_RES

from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
)

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


def _make_clip_for_scene(asset_path: str, duration: float):
    """Return a fixed-size VideoClip for one script scene.

    - MP4 assets are trimmed or looped to match *duration*.
    - Image assets get a smooth Ken Burns zoom-in effect so the video
      does not look static.
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

    # Static image – Ken Burns zoom-in (scale grows 2% per second).
    # set_duration() is called explicitly after construction to satisfy
    # MoviePy's internal timeline even when duration is also given to the
    # ImageClip constructor.
    img_clip = (
        ImageClip(asset_path)
        .set_duration(duration)
        .resize(lambda t: min(1 + 0.02 * t, 1.3))
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
        for i, (data, asset_path) in enumerate(zip(voice_data, visual_assets), start=1):
            logger.info("   [%d/%d] Procesando: %s...", i, total, data["texto"][:50])
            clip = _make_clip_for_scene(asset_path, data["duracion"])
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

        # 8. Añadir pista de audio final
        logger.info("🔊 Añadiendo audio final...")
        audio = AudioFileClip(audio_path)
        final_video = final_video.set_audio(audio)

        # 9. Exportar
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
        # 10. Liberar memoria RAM – always runs to prevent memory leaks
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


if __name__ == "__main__":
    asyncio.run(main())