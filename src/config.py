import os
from dotenv import load_dotenv

load_dotenv()

# Rutas del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
AUDIO_DIR = os.path.join(ASSETS_DIR, "audio")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Configuración de Voces (Microsoft Edge TTS - Argentina)
VOICES = {
    "H": "es-AR-TomasNeural",
    "M": "es-AR-ElenaNeural"
}

# Configuración de Video
VIDEO_RES = (1080, 1920)  # Formato Vertical (TikTok/Reels)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")