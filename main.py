import json
import asyncio
from src.narrator import ArgentineNarrator
# Importaremos el resto luego

async def main():
    print("🚀 Iniciando PyNarrator...")
    # 1. Leer guion
    with open("script.json", "r", encoding="utf-8") as f:
        script = json.load(f)
    
    print("🎙️ Generando voces argentinas...")
    # Aquí llamaremos a las funciones que Copilot completará
    
    print("✅ Proceso finalizado.")

if __name__ == "__main__":
    asyncio.run(main())