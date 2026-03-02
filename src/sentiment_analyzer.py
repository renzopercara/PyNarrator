"""sentiment_analyzer.py – Keyword-based script tone classifier.

Classifies a script's full text as one of three emotional tones:
  - ENERGICO    (Energetic / Motivational)
  - INFORMATIVO (Informative / Serious)
  - RELAJADO    (Relaxed / Lo-Fi)
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists (lower-case substrings)
# ---------------------------------------------------------------------------

# Energetic / motivational language
_ENERGICO_KEYWORDS = [
    "increíble", "increible", "impresionante", "¡", "ahora mismo",
    "actúa", "actua", "cambia", "cambiar", "poder", "potencia",
    "motiva", "motivac", "energía", "energia", "fuerza", "logra",
    "lograr", "éxito", "exito", "triunfa", "vamos", "arriba",
    "explosiv", "dinamic", "epic", "épic", "ganar", "ganá",
    "revolución", "revolucion", "transforma", "boom", "wow",
    "genial", "brutal", "fantástic", "fantastico", "súper", "super",
    "acción", "accion", "rápido", "rapido", "urgente", "ya",
]

# Relaxed / lo-fi / calm language
_RELAJADO_KEYWORDS = [
    "relaj", "tranquil", "tranqui", "calma", "calm", "paz",
    "disfrut", "suave", "respira", "medita", "serenidad",
    "sereno", "pausado", "pausa", "lentamente", "despacio",
    "naturaleza", "bienestar", "zen", "armonía", "armonia",
    "flow", "dulce", "silencio", "contempla", "mindful",
    "sosiego", "reposar", "descanso", "descansar", "relajación",
]

# Informative / tutorial / educational language
_INFORMATIVO_KEYWORDS = [
    "cómo", "como", "aprende", "tutorial", "explica", "paso",
    "proceso", "método", "metodo", "técnica", "tecnica",
    "estrategia", "guía", "guia", "conoce", "conocimiento",
    "información", "informacion", "datos", "hecho", "importante",
    "recuerda", "tip", "consejo", "anali", "veamos", "aprenderás",
    "enseña", "enseñamos", "revisar", "revisamos",
    "descripción", "descripcion", "detalle", "definición",
]


def analyze_tone(script_data: list[dict]) -> str:
    """Classify the overall tone of *script_data*.

    Counts keyword matches for each tone category across all scene texts
    and returns the category with the highest score.  Defaults to
    ``"INFORMATIVO"`` when scores are tied or no keywords are found.

    Args:
        script_data: The parsed ``script.json`` list of scene dicts.

    Returns:
        One of ``"ENERGICO"``, ``"INFORMATIVO"``, or ``"RELAJADO"``.
    """
    full_text = " ".join(item.get("texto", "") for item in script_data).lower()

    scores: dict[str, int] = {"ENERGICO": 0, "INFORMATIVO": 0, "RELAJADO": 0}

    for kw in _ENERGICO_KEYWORDS:
        if kw in full_text:
            scores["ENERGICO"] += 1

    for kw in _RELAJADO_KEYWORDS:
        if kw in full_text:
            scores["RELAJADO"] += 1

    for kw in _INFORMATIVO_KEYWORDS:
        if kw in full_text:
            scores["INFORMATIVO"] += 1

    best_score = max(scores.values())
    # Default to INFORMATIVO on a tie or when nothing matched
    if best_score == 0 or list(scores.values()).count(best_score) > 1:
        tone = "INFORMATIVO"
    else:
        tone = max(scores, key=lambda k: scores[k])

    logger.info(
        "🧠 Análisis de tono: ENERGICO=%d, INFORMATIVO=%d, RELAJADO=%d → %s",
        scores["ENERGICO"],
        scores["INFORMATIVO"],
        scores["RELAJADO"],
        tone,
    )
    return tone
