import os
import shutil
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")
MUSIC_FAST_DIR = os.path.join(MUSIC_DIR, "fast")       # Upbeat, Rock, Electrónica
MUSIC_SLOW_DIR = os.path.join(MUSIC_DIR, "slow")       # Lo-Fi, Piano, Chill
MUSIC_CORPORATE_DIR = os.path.join(MUSIC_DIR, "corporate")  # Clean corporate/tutorial
SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Configuración de Voces (Microsoft Edge TTS - Argentina)
VOICES = {
    "H": "es-AR-TomasNeural",
    "M": "es-AR-ElenaNeural"
}

# Marca de Agua
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

# Configuración de Video
VIDEO_RES = (1080, 1920)  # Formato Vertical (TikTok/Reels)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")


def _configure_imagemagick():
    """Automatically find and configure the ImageMagick binary for MoviePy.

    Searches for ``magick`` / ``magick.exe`` (Windows) or ``convert``
    (Linux/macOS) on the system PATH and calls
    ``change_settings({"IMAGEMAGICK_BINARY": path})`` so that MoviePy's
    TextClip (used for subtitles) does not raise a binary-not-found error.
    """
    try:
        from moviepy.config import change_settings  # type: ignore

        for candidate in ("magick", "magick.exe", "convert"):
            path = shutil.which(candidate)
            if path:
                change_settings({"IMAGEMAGICK_BINARY": path})
                logger.debug("ImageMagick binary configured: %s", path)
                return

        logger.warning(
            "ImageMagick binary not found on PATH. "
            "Subtitle rendering may fail. Install ImageMagick and ensure it is on PATH."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not configure ImageMagick for MoviePy: %s", exc)


_configure_imagemagick()