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

# --- DIRECTOR IA: Rutas de Música por Tono ---
# Estas carpetas permiten que el script elija música según el sentimiento del guion
MUSIC_FAST_DIR = os.path.join(MUSIC_DIR, "fast")           # Tono: ENERGICO
MUSIC_SLOW_DIR = os.path.join(MUSIC_DIR, "slow")           # Tono: RELAJADO
MUSIC_CORPORATE_DIR = os.path.join(MUSIC_DIR, "corporate") # Tono: INFORMATIVO

SFX_DIR = os.path.join(ASSETS_DIR, "sfx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Configuración de Voces Americanas (Edge-TTS / Azure)
VOICES = {
    "H": "en-US-AndrewNeural",  # Warm, professional, clear voice – podcast style
    "M": "en-US-AvaNeural",     # Bright, natural, expressive voice – narrator style
}

# ESL / Micro-Learning voice settings for maximum educational clarity
ESL_VOICE_RATE = "-5%"
ESL_VOICE_PITCH = "+0Hz"

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
WATERMARK_PATH = os.path.join(ASSETS_DIR, "logo.png")
WATERMARK_OPACITY = 0.8
WATERMARK_WIDTH_PERCENT = 0.25
VIDEO_RES = (1080, 1350) # Formato vertical (9:16) para Reels/TikTok/Shorts
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _configure_imagemagick():
    """
    Configura automáticamente ImageMagick para MoviePy buscando el binario
    en diferentes sistemas operativos.
    """
    try:
        from moviepy.config import change_settings
        # Lista de nombres comunes para el ejecutable de ImageMagick
        for candidate in ("magick", "magick.exe", "convert"):
            path = shutil.which(candidate)
            if path:
                change_settings({"IMAGEMAGICK_BINARY": path})
                logger.info(f"✅ ImageMagick configurado en: {path}")
                return
        logger.warning("⚠️ ImageMagick no encontrado. El renderizado de texto avanzado podría fallar.")
    except Exception as e:
        logger.error(f"❌ Error configurando ImageMagick: {e}")

# Ejecutar configuración al importar
_configure_imagemagick()