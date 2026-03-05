"""vocabulary_annotator.py – English vocabulary annotator for Spanish-speaking learners.

Receives an English short story and annotates B1/B2/C1 level words by inserting
their Spanish translations in parentheses immediately after each target word.

Rules enforced:
- At most 2 words are annotated per sentence.
- Basic (A1/A2) words are never annotated.
- The original sentence structure and casing are preserved.

Example usage::

    from src.vocabulary_annotator import annotate_story

    story = "The challenge was big. She had to overcome many obstacles."
    result = annotate_story(story)
    # → "The challenge (desafío) was big. She had to overcome (superar) many obstacles."
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# B1/B2/C1 vocabulary: lowercase English word → Spanish translation
# ---------------------------------------------------------------------------

_VOCABULARY: dict[str, str] = {
    # --- B1 ---
    "achieve": "lograr",
    "achievement": "logro",
    "adventure": "aventura",
    "affect": "afectar",
    "although": "aunque",
    "ancient": "antiguo",
    "announce": "anunciar",
    "appreciate": "apreciar",
    "approach": "enfoque",
    "aware": "consciente",
    "behavior": "comportamiento",
    "behaviour": "comportamiento",
    "beneath": "debajo de",
    "blame": "culpar",
    "brave": "valiente",
    "brilliant": "brillante",
    "calm": "calma",
    "capable": "capaz",
    "challenge": "desafío",
    "clever": "inteligente",
    "communicate": "comunicar",
    "compare": "comparar",
    "condition": "condición",
    "confident": "seguro/confiado",
    "confused": "confundido",
    "consider": "considerar",
    "curious": "curioso",
    "decision": "decisión",
    "despite": "a pesar de",
    "develop": "desarrollar",
    "disagree": "estar en desacuerdo",
    "discuss": "discutir",
    "distance": "distancia",
    "doubt": "duda",
    "effort": "esfuerzo",
    "embarrassed": "avergonzado",
    "encourage": "alentar",
    "environment": "entorno",
    "escape": "escapar",
    "eventually": "finalmente",
    "exhausted": "agotado",
    "experience": "experiencia",
    "explore": "explorar",
    "fail": "fallar",
    "familiar": "familiar",
    "fear": "miedo",
    "forest": "bosque",
    "freedom": "libertad",
    "friendly": "amigable",
    "furthermore": "además",
    "gentle": "suave/amable",
    "glad": "contento",
    "goal": "meta",
    "grateful": "agradecido",
    "handle": "manejar",
    "happen": "ocurrir",
    "honest": "honesto",
    "however": "sin embargo",
    "imagine": "imaginar",
    "improve": "mejorar",
    "include": "incluir",
    "increase": "aumentar",
    "indeed": "de hecho",
    "individual": "individuo",
    "journey": "viaje",
    "joy": "alegría",
    "judge": "juzgar",
    "kindness": "amabilidad",
    "knowledge": "conocimiento",
    "laugh": "reír",
    "mention": "mencionar",
    "message": "mensaje",
    "mistake": "error",
    "moment": "momento",
    "mysterious": "misterioso",
    "nervous": "nervioso",
    "occasionally": "ocasionalmente",
    "opportunity": "oportunidad",
    "overcome": "superar",
    "perhaps": "quizás",
    "predict": "predecir",
    "prepare": "preparar",
    "prevent": "prevenir",
    "protect": "proteger",
    "purpose": "propósito",
    "realize": "darse cuenta",
    "recently": "recientemente",
    "recognize": "reconocer",
    "reduce": "reducir",
    "relate": "relacionar",
    "remember": "recordar",
    "require": "requerir",
    "research": "investigación",
    "responsible": "responsable",
    "reward": "recompensa",
    "search": "buscar",
    "skill": "habilidad",
    "solve": "resolver",
    "strange": "extraño",
    "struggle": "lucha/dificultad",
    "suggest": "sugerir",
    "support": "apoyar",
    "therefore": "por lo tanto",
    "tradition": "tradición",
    "trust": "confianza",
    "unfortunately": "desafortunadamente",
    "unique": "único",
    "various": "varios",
    "wealth": "riqueza",
    "wonder": "maravilla",
    "worried": "preocupado",
    # --- B2 ---
    "abandon": "abandonar",
    "abstract": "abstracto",
    "abundant": "abundante",
    "acquire": "adquirir",
    "adapt": "adaptarse",
    "adequate": "adecuado",
    "ambition": "ambición",
    "anticipate": "anticipar",
    "apparent": "aparente",
    "assess": "evaluar",
    "assume": "asumir",
    "atmosphere": "atmósfera",
    "attempt": "intento",
    "attitude": "actitud",
    "benefit": "beneficio",
    "boundary": "límite",
    "capacity": "capacidad",
    "circumstances": "circunstancias",
    "collaborate": "colaborar",
    "commitment": "compromiso",
    "complex": "complejo",
    "comprehend": "comprender",
    "concentrate": "concentrarse",
    "consequence": "consecuencia",
    "contribute": "contribuir",
    "convince": "convencer",
    "crucial": "crucial",
    "decline": "declinar",
    "dedicate": "dedicar",
    "define": "definir",
    "demonstrate": "demostrar",
    "desire": "deseo",
    "determine": "determinar",
    "diverse": "diverso",
    "eliminate": "eliminar",
    "emerge": "surgir",
    "emphasize": "enfatizar",
    "encounter": "encuentro",
    "enhance": "mejorar/realzar",
    "enormous": "enorme",
    "establish": "establecer",
    "evaluate": "evaluar",
    "exceed": "superar",
    "explicit": "explícito",
    "extraordinary": "extraordinario",
    "facilitate": "facilitar",
    "flourish": "florecer",
    "fragile": "frágil",
    "fundamental": "fundamental",
    "generate": "generar",
    "genuine": "genuino",
    "global": "global",
    "identify": "identificar",
    "illustrate": "ilustrar",
    "implement": "implementar",
    "imply": "implicar",
    "incorporate": "incorporar",
    "indicate": "indicar",
    "influence": "influencia",
    "intense": "intenso",
    "investigate": "investigar",
    "involve": "involucrar",
    "maintain": "mantener",
    "methodology": "metodología",
    "modify": "modificar",
    "motivate": "motivar",
    "negotiate": "negociar",
    "numerous": "numeroso",
    "objective": "objetivo",
    "obtain": "obtener",
    "overwhelm": "abrumar",
    "perception": "percepción",
    "perspective": "perspectiva",
    "phenomenon": "fenómeno",
    "potential": "potencial",
    "precise": "preciso",
    "principle": "principio",
    "promote": "promover",
    "pursue": "perseguir",
    "relevant": "relevante",
    "remarkable": "notable",
    "represent": "representar",
    "resolve": "resolver",
    "reveal": "revelar",
    "significant": "significativo",
    "stimulate": "estimular",
    "strategy": "estrategia",
    "sustain": "sostener",
    "tolerate": "tolerar",
    "transform": "transformar",
    "transmit": "transmitir",
    "tremendous": "tremendo",
    "urgent": "urgente",
    "utilize": "utilizar",
    "witness": "presenciar",
    # --- C1 ---
    "accomplish": "lograr",
    "acknowledge": "reconocer/admitir",
    "alleviate": "aliviar",
    "ambiguous": "ambiguo",
    "articulate": "articular",
    "attain": "alcanzar",
    "autonomous": "autónomo",
    "coherent": "coherente",
    "compelling": "convincente",
    "conceive": "concebir",
    "consolidate": "consolidar",
    "contradict": "contradecir",
    "controversial": "controvertido",
    "deduce": "deducir",
    "diminish": "disminuir",
    "disperse": "dispersar",
    "elaborate": "elaborar",
    "encompass": "abarcar",
    "endure": "perdurar",
    "exploit": "aprovechar",
    "feasible": "factible",
    "formulate": "formular",
    "inevitably": "inevitablemente",
    "inherent": "inherente",
    "innovative": "innovador",
    "instigate": "instigar",
    "manifest": "manifestar",
    "perceive": "percibir",
    "profound": "profundo",
    "reconcile": "reconciliar",
    "refine": "perfeccionar",
    "reinforce": "reforzar",
    "restrict": "restringir",
    "scrutinize": "examinar minuciosamente",
    "subsequent": "posterior",
    "supplement": "complementar",
    "trigger": "desencadenar",
    "underlying": "subyacente",
    "versatile": "versátil",
    "vulnerable": "vulnerable",
    "withstand": "resistir",
    "yield": "producir/ceder",
}

_MAX_TRANSLATIONS_PER_SENTENCE = 2

# Pattern to split text into sentences while keeping the delimiter attached
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Pattern that matches a single word token (letters only, no punctuation)
_WORD_RE = re.compile(r"\b([A-Za-z]+)\b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _annotate_sentence(sentence: str) -> str:
    """Insert Spanish translations for up to ``_MAX_TRANSLATIONS_PER_SENTENCE``
    B1/B2/C1 words found in *sentence*.

    Translations are added in parentheses immediately after the matched word,
    preserving the original word's casing in the output.

    Args:
        sentence: A single English sentence.

    Returns:
        The sentence with Spanish glosses inserted.
    """
    translations_added = 0
    result_parts: list[str] = []
    last_end = 0

    for match in _WORD_RE.finditer(sentence):
        if translations_added >= _MAX_TRANSLATIONS_PER_SENTENCE:
            break

        word = match.group(1)
        translation = _VOCABULARY.get(word.lower())

        if translation is None:
            continue

        # Append everything before this word unchanged
        result_parts.append(sentence[last_end : match.end()])
        # Append the gloss right after the word
        result_parts.append(f" ({translation})")
        last_end = match.end()
        translations_added += 1

    # Append the remainder of the sentence
    result_parts.append(sentence[last_end:])
    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_story(story: str) -> str:
    """Annotate a short English story for Spanish-speaking learners.

    Splits *story* into sentences, then for each sentence inserts Spanish
    translations of up to two B1/B2/C1 level words in parentheses
    immediately after the original word.  Basic (A1/A2) words are never
    annotated.

    Args:
        story: A short English story (one or more sentences).

    Returns:
        The annotated story with Spanish glosses inserted.

    Example::

        >>> annotate_story("The challenge was big. She must overcome many obstacles.")
        'The challenge (desafío) was big. She must overcome (superar) many obstacles.'
    """
    if not story or not story.strip():
        return story

    # Split on sentence boundaries.  Multiple spaces between sentences are
    # normalised to a single space in the output; sentence-internal spacing
    # is always preserved verbatim.
    sentences = _SENTENCE_SPLIT_RE.split(story.strip())
    annotated = [_annotate_sentence(s) for s in sentences]

    logger.info(
        "📚 Vocabulario anotado: %d oraciones procesadas.",
        len(annotated),
    )
    return " ".join(annotated)
