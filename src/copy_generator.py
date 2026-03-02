"""copy_generator.py – Social media copy generator for PyNarrator.

Analyses the script and writes ``output/INFO_POSTEO.txt`` containing:

- A clickbait-ethical title
- An Instagram/TikTok description with emojis
- 15 viral hashtags based on script keywords
"""

import logging
import os
import re

from src.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

_INFO_FILE = os.path.join(OUTPUT_DIR, "INFO_POSTEO.txt")

# Pool of generic viral hashtags used to pad up to the requested count.
_VIRAL_HASHTAGS_POOL = [
    "#Viral",
    "#Reels",
    "#Shorts",
    "#TikTok",
    "#Instagram",
    "#Trending",
    "#FYP",
    "#ParaTi",
    "#ForYou",
    "#Contenido",
    "#Video",
    "#Tendencia",
    "#CreadorDeContenido",
    "#MarketingDigital",
    "#SocialMedia",
]

# Rotating emoji pool used in the description body.
_EMOJIS = ["🚀", "🔥", "💡", "🤯", "✨", "💪", "🎯", "⚡", "🌟", "👀", "🙌", "💎"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_keywords(script: list[dict]) -> list[str]:
    """Return a deduplicated ordered list of keywords found in *script*."""
    seen: set[str] = set()
    result: list[str] = []
    for item in script:
        kw = item.get("keyword", "").strip()
        if kw and kw.lower() not in seen:
            seen.add(kw.lower())
            result.append(kw)
    return result


def _significant_words(texts: list[str], min_len: int = 4) -> list[str]:
    """Extract unique significant words (length ≥ *min_len*) from *texts*."""
    seen: set[str] = set()
    words: list[str] = []
    for text in texts:
        for word in re.findall(
            r"\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{" + str(min_len) + r",}\b", text
        ):
            lower = word.lower()
            if lower not in seen:
                seen.add(lower)
                words.append(lower)
    return words[:20]


def _generate_title(script: list[dict]) -> str:
    """Return a clickbait-ethical title derived from *script*."""
    keywords = _extract_keywords(script)
    texts = [item.get("texto", "").strip() for item in script if item.get("texto", "").strip()]

    if keywords:
        topic = keywords[0].title()
    elif texts:
        topic = " ".join(texts[0].split()[:5])
    else:
        topic = "Contenido Digital"

    return f"🚀 ¡Lo que nadie te contó sobre {topic}!"


def _generate_description(script: list[dict]) -> str:
    """Return an Instagram/TikTok-optimised description with emojis."""
    keywords = _extract_keywords(script)
    texts = [item.get("texto", "").strip() for item in script if item.get("texto", "").strip()]

    # Build a short summary from the first two script items.
    summary_parts: list[str] = []
    for text in texts[:2]:
        first_sentence = re.split(r"[.!?]", text)[0].strip()
        if first_sentence:
            summary_parts.append(first_sentence)

    summary = ". ".join(summary_parts) if summary_parts else "Contenido increíble te espera"
    topic = keywords[0].title() if keywords else "este tema"

    return (
        f"{_EMOJIS[1]} {summary} {_EMOJIS[0]}\n"
        f"¿Ya conocías todo sobre {topic}? {_EMOJIS[3]}\n"
        f"Guardá este video para no olvidarlo {_EMOJIS[4]}\n"
        f"¡Seguinos para más contenido como este! {_EMOJIS[6]}"
    )


def _generate_hashtags(script: list[dict], count: int = 15) -> list[str]:
    """Return *count* viral hashtags derived from *script* keywords and text."""
    keywords = _extract_keywords(script)
    texts = [item.get("texto", "").strip() for item in script if item.get("texto", "").strip()]

    hashtags: list[str] = []

    # 1. Hashtags from explicit keywords.
    for kw in keywords:
        tag = "#" + re.sub(r"\s+", "", kw.title())
        if tag not in hashtags:
            hashtags.append(tag)

    # 2. Hashtags from significant words in the script text.
    for word in _significant_words(texts):
        if len(hashtags) >= count:
            break
        tag = "#" + word.capitalize()
        if tag not in hashtags:
            hashtags.append(tag)

    # 3. Generic viral hashtags to fill remaining slots.
    for tag in _VIRAL_HASHTAGS_POOL:
        if len(hashtags) >= count:
            break
        if tag not in hashtags:
            hashtags.append(tag)

    return hashtags[:count]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_social_copy(script: list[dict]) -> str:
    """Analyse *script* and write ``output/INFO_POSTEO.txt``.

    Generates:

    - A clickbait-ethical title
    - An Instagram/TikTok description with emojis
    - 15 viral hashtags based on script keywords and text

    Args:
        script: The same list-of-dicts loaded from ``script.json``.

    Returns:
        The absolute path to the generated ``INFO_POSTEO.txt`` file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    title = _generate_title(script)
    description = _generate_description(script)
    hashtags = _generate_hashtags(script)
    hashtags_str = " ".join(hashtags)

    content = (
        f"Título: {title}\n\n"
        f"Descripción:\n{description}\n\n"
        f"Hashtags:\n{hashtags_str}\n"
    )

    with open(_INFO_FILE, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info("📋 Copy para redes generado en: %s", _INFO_FILE)
    return _INFO_FILE
