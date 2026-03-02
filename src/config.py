import os
import shutil
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Rutas Base
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
MUSIC_DIR = os.path.join(ASSETS_DIR, "music")

# Subcarpetas para el "Director IA"
MUSIC_FAST_DIR = os.path.join(MUSIC_DIR, "fast")       # Música para tono ENERGICO
MUSIC_SLOW_DIR = os.path.join(MUSIC_DIR, "slow")       # Música para tono RELAJADO
MUSIC_CORPORATE_DIR = os.path.join(MUSIC_DIR, "corporate") # Música para tono INFORMATIVO

SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Configuración de Voces Argentinas
VOICES = {
    "H": "es-AR-TomasNeural",
    "M": "es-AR-ElenaNeural"
}

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
VIDEO_RES = (1080, 1920) # Formato Reels/TikTok
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

def _configure_imagemagick():
    """Configura automáticamente ImageMagick para MoviePy"""
    try:
        from moviepy.config import change_settings
        for candidate in ("magick", "magick.exe", "convert"):
            path = shutil.which(candidate)
            if path:
                change_settings({"IMAGEMAGICK_BINARY": path})
                return
        logger.warning("ImageMagick no encontrado. Subtítulos podrían fallar.")
    except Exception as e:
        logger.error(f"Error configurando ImageMagick: {e}")

_configure_imagemagick()