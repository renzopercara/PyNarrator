# 📖 README INSTRUCTIVO — PyNarrator: Motor Dual de Contenido

> **Rol:** Principal Engineer & Creative Director  
> **Sistema:** PyNarrator — Fábrica de video vertical para Reels / TikTok / Shorts  
> **Verticales soportadas:** Smartbuild Construcciones · ESL Tech (inglés para IT)

---

## 📁 1. Estructura del Proyecto

```
PyNarrator/
├── main.py                  ← Motor único de renderizado
├── script.json              ← Guion activo (intercambiable por vertical)
├── script-2.json            ← Guion alternativo de ejemplo
├── README_INSTRUCTIVO.md    ← Este archivo
├── requirements.txt
├── .env                     ← PEXELS_API_KEY, OPENAI_API_KEY
│
├── src/
│   ├── config.py            ← Rutas, voces y resolución de video
│   ├── narrator.py          ← Síntesis de voz (Edge-TTS, acento argentino)
│   ├── image_manager.py     ← Descarga y gestión de assets visuales
│   ├── subtitle_generator.py← Subtítulos sincronizados con Whisper
│   ├── sentiment_analyzer.py← Detección de tono (ENERGICO / INFORMATIVO / RELAJADO)
│   ├── copy_generator.py    ← Copy para redes sociales
│   ├── vocabulary_annotator.py ← Anotación de vocabulario para ESL
│   ├── esl_narrative_generator.py ← Generador de narrativas en inglés
│   └── script_generator.py  ← Conversor texto → script.json
│
├── assets/
│   ├── brand/               ← Logos e imágenes de marca (SmartBuild_Construcciones.jpg, etc.)
│   ├── audio/               ← Audios generados (limpiados automáticamente)
│   ├── music/
│   │   ├── fast/            ← Música para tono ENERGICO
│   │   ├── slow/            ← Música para tono RELAJADO
│   │   └── corporate/       ← Música para tono INFORMATIVO
│   ├── sfx/                 ← Efectos de sonido (pop.mp3, ambience_construction.mp3)
│   └── logo.png             ← Watermark del video
│
└── output/                  ← Carpeta de salida (creada automáticamente)
    ├── video_final_smartbuild.mp4
    └── video_final_esl.mp4
```

### Convivencia de assets por vertical

| Vertical       | Assets de marca              | Música recomendada | SFX                          |
|---------------|------------------------------|--------------------|------------------------------|
| **Smartbuild** | `assets/brand/SmartBuild_*`  | `music/corporate/` | `ambience_construction.mp3`  |
| **ESL Tech**   | `assets/brand/` (logo IT)    | `music/fast/`      | `pop.mp3` en cada escena     |

Los assets de ambas verticales **conviven en la misma carpeta `assets/`** sin conflictos.
El motor los diferencia usando el campo `contexto` del `script.json` y el campo `source` de cada escena.

---

## 📋 2. El JSON Universal — `script.json`

Cada escena es un objeto JSON con los siguientes campos:

| Campo       | Tipo     | Obligatorio | Descripción                                                                                  |
|-------------|----------|-------------|----------------------------------------------------------------------------------------------|
| `contexto`  | `string` | ✅ Sí       | Tipo de contenido: `"smartbuild"` o `"esl"`. El motor detecta la vertical dominante.        |
| `voz`       | `string` | ✅ Sí       | `"H"` → Tomas (masculino, autoridad) · `"M"` → Elena (femenina, profesional)               |
| `texto`     | `string` | ✅ Sí       | Texto que se muestra en subtítulos. Máx. 10 palabras por escena.                            |
| `fonetica`  | `string` | ✅ Sí       | Texto que lee la IA en voz alta. **Obligatorio** para garantizar el acento argentino natural.|
| `keyword`   | `string` | ✅ Sí       | Palabra clave para la búsqueda de imagen en Pexels. Ver reglas en §4.                        |
| `source`    | `string` | ⬜ Opcional | URL o ruta local de imagen/video. Si se omite, el motor busca en Pexels usando `keyword`.   |

### Regla del motor para detectar el contexto

El motor lee el campo `contexto` de **todas** las escenas y elige la vertical más frecuente.
Esto significa que un `script.json` mixto (ej. 10 escenas Smartbuild + 2 ESL de cierre) renderizará
como video Smartbuild (`video_final_smartbuild.mp4`). Si el campo `contexto` **no está presente**
en ninguna escena, el motor usa `"smartbuild"` como valor por defecto para retrocompatibilidad.

---

## 🚀 3. Operación Dual

El sistema utiliza un **motor único** basado en MoviePy. Para alternar entre verticales,
solo cambiás el contenido de `script.json`:

### Smartbuild
```jsonc
// Usa voz: "H" (Tomas) para autoridad en la narración principal.
// La voz "H" tiene aplicado automáticamente rate: -10% para un ritmo pausado y experto.
// Keywords: términos arquitectónicos e industriales (ver §4).
{
  "contexto": "smartbuild",
  "voz": "H",
  "texto": "Smartbuild Construcciones.",
  "fonetica": "SMART-bild Construcciones.",
  "keyword": "steel framing dry wall construction argentina"
}
```

### ESL Tech Agent
```jsonc
// Usa alternancia de voces "H" y "M" para dinamismo educativo.
// Keywords: entorno de oficina IT moderno (ver §4).
{
  "contexto": "esl",
  "voz": "M",
  "texto": "Pull requests help teams review code.",
  "fonetica": "Pull requests help teams review code.",
  "keyword": "coding startup office github ui"
}
```

### Comando Único de Renderizado

```bash
python main.py
```

El video resultante se guarda automáticamente en:

```
output/video_final_smartbuild.mp4   ← si la vertical dominante es Smartbuild
output/video_final_esl.mp4          ← si la vertical dominante es ESL
```

---

## 🎨 4. Criterio de Senior Designer — Reglas para Keywords Visuales

La `keyword` de cada escena es la consulta que el motor envía a la API de Pexels.
Una buena keyword produce imágenes coherentes con la marca; una mala produce imágenes genéricas
o irrelevantes.

### 🏗️ Smartbuild — Keywords Arquitectónicas / Industriales

**Objetivo:** Transmitir solidez, expertise técnico y obra profesional.

| ✅ Buenas keywords                          | ❌ Evitar                          |
|---------------------------------------------|------------------------------------|
| `steel framing dry wall construction`       | `house home`                       |
| `architecture blueprint detail`             | `people smiling`                   |
| `construction worker argentina professional`| `abstract background`              |
| `concrete structure industrial building`    | `technology digital`               |
| `planos arquitecto presupuesto`             | `money finance`                    |
| `obra albanil casco profesional`            | `office interior`                  |

**Regla mnemotécnica Smartbuild:** Si la imagen podría aparecer en una revista de arquitectura
o en un catálogo de materiales de construcción, es válida.

### 💻 ESL Tech Agent — Keywords de Entorno IT Moderno

**Objetivo:** Transmitir modernidad tech, colaboración y entorno de startup.

| ✅ Buenas keywords                          | ❌ Evitar                          |
|---------------------------------------------|------------------------------------|
| `coding startup office developer laptop`    | `construction work`                |
| `github ui pull request code review`        | `outdoors nature`                  |
| `modern office team collaboration screen`   | `old technology`                   |
| `software developer woman typing code`      | `formal suit corporate`            |
| `open plan office tech startup`             | `heavy machinery`                  |
| `dark mode code editor monitor`             | `physical labor tools`             |

**Regla mnemotécnica ESL:** Si la imagen podría aparecer en el sitio web de una startup
de Silicon Valley o en la portada de un curso online de programación, es válida.

---

## 🎬 5. Exportación con `yuv420p` — Compatibilidad Total con Reels y TikTok

El motor exporta con el siguiente parámetro ffmpeg crítico:

```python
ffmpeg_params=[
    "-pix_fmt", "yuv420p",      # ← CLAVE para compatibilidad móvil
    "-vf", "setsar=1:1",
    "-movflags", "+faststart",
    "-profile:v", "high",
    "-level", "4.0"
]
```

### ¿Por qué `yuv420p`?

| Técnica          | Subsampling | Compatibilidad móvil | Tamaño de archivo |
|-----------------|-------------|---------------------|-------------------|
| `yuv444p`       | 4:4:4       | ❌ Baja (algunos celulares rechazan el video) | Grande |
| **`yuv420p`**   | **4:2:0**   | **✅ Universal (Instagram, TikTok, WhatsApp)** | **Óptimo** |
| `yuv422p`       | 4:2:2       | ⚠️ Media (ProRes, no web)                    | Mediano |

**Resumen:** `yuv420p` es el estándar de facto de todas las plataformas de video móvil.
Sin este parámetro, Instagram y TikTok pueden rechazar el video o mostrar una pantalla negra
al reproducirlo desde dispositivos Android de gama media.

---

## 📣 6. Criterio de CM — Contenido Viral

### Regla #1: Primera Escena < 3 Segundos

La primera escena del `script.json` es el "gancho" que determina si el usuario sigue viendo.
Las plataformas penalizan los videos donde los espectadores abandonan en los primeros 3 segundos.

**Cómo implementarlo:** El texto de la primera escena debe ser corto e impactante.
La duración del audio se genera automáticamente a partir del campo `fonetica`.
Para una duración < 3 segundos, el texto no debe superar **5-6 palabras**.

```jsonc
// ✅ Escena 1 gancho (~1.5 segundos)
{ "texto": "¿Construir sin sorpresas?", "fonetica": "¿Construir sin SORPRESAS?", ... }

// ❌ Escena 1 lenta (~4+ segundos → el usuario cierra)
{ "texto": "Bienvenidos a Smartbuild Construcciones, empresa líder.", ... }
```

### Regla #2: El Campo `fonetica` es Obligatorio

El campo `fonetica` controla exactamente lo que lee la IA (Edge-TTS, voz argentina).
Sin él, la IA lee el texto literal y puede pronunciar mal términos como:

| Texto (no apto para TTS) | Fonética correcta                              |
|--------------------------|------------------------------------------------|
| `Smartbuild`             | `SMART-bild`                                   |
| `+54 343 508-0085`       | `más cincuenta y cuatro. Tres cuarenta y tres.`|
| `steel framing`          | `stil fréiming`                                |
| `pull request`           | `pul ri-KUEST`                                 |

**Omitir `fonetica` rompe la identidad de marca.** La pronunciación incorrecta de "Smartbuild"
hace que la audiencia pierda confianza en la marca desde el primer segundo de audio.

---

## 🧩 7. Ejemplo Completo de `script.json` Dual

El siguiente ejemplo contiene **una escena de Smartbuild y una de ESL** en el mismo archivo.
El motor lo procesa sin errores (la vertical dominante es `"smartbuild"`, 1 vs 1, con
`"smartbuild"` ganando por orden de aparición como desempate).

```json
[
  {
    "contexto": "smartbuild",
    "voz": "H",
    "texto": "¿Construir sin sorpresas?",
    "fonetica": "¿Construir sin SORPRESAS?",
    "keyword": "steel framing dry wall construction argentina",
    "source": "assets/brand/SmartBuild_Construcciones.jpg"
  },
  {
    "contexto": "esl",
    "voz": "M",
    "texto": "Pull requests help teams review code.",
    "fonetica": "Pull requests help teams review code.",
    "keyword": "coding startup office github ui"
  }
]
```

> **Nota:** El campo `source` es opcional. Si se omite, el motor descarga una imagen desde
> Pexels usando el campo `keyword`. Si se proporciona una URL o ruta local válida, se usa
> directamente sin consultar la API.

---

## ⚙️ 8. Variables de Entorno Requeridas

Crear un archivo `.env` en la raíz del proyecto con:

```dotenv
PEXELS_API_KEY=your_pexels_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

Para verificar que el entorno está bien configurado antes del primer render:

```bash
python test_setup.py
```

---

## 🔢 9. Referencia Rápida de Voces

| Clave | Voz Edge-TTS        | Género    | Rate aplicado | Uso recomendado                          |
|-------|---------------------|-----------|---------------|------------------------------------------|
| `"H"` | `es-AR-TomasNeural` | Masculino | -10%          | Autoridad, CTA, narración Smartbuild     |
| `"M"` | `es-AR-ElenaNeural` | Femenino  | 0%            | Educación, ESL, escenas de diálogo       |

Ambas voces tienen acento **rioplatense argentino** nativo. El campo `fonetica` permite
ajustar la cadencia sin cambiar la voz base.

---

## ✅ 10. Checklist de Lanzamiento de un Nuevo Video

- [ ] `script.json` tiene el campo `contexto` en cada escena
- [ ] La primera escena tiene ≤ 5-6 palabras en `fonetica` (< 3 segundos)
- [ ] Todas las escenas tienen el campo `fonetica` completado
- [ ] Las keywords siguen las reglas de la vertical (§4)
- [ ] Variables de entorno configuradas en `.env`
- [ ] Assets de marca en `assets/brand/` (logo, imagen de cabecera)
- [ ] Música de fondo en `assets/music/{fast|slow|corporate}/`
- [ ] Ejecutar `python main.py`
- [ ] Verificar el `.mp4` en `output/`
