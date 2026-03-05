"""phonetic_transcriber.py – English-to-Spanish phonetic transcription for TTS.

Given an English sentence, this module generates its Spanish phonetic
approximation so that a Spanish TTS engine reads the text with a natural
English accent.

Example::

    >>> transcriber = EnglishPhoneticTranscriber()
    >>> transcriber.transcribe("The challenge was huge")
    'Di chá-lench uoz hiuch'

The system prompt follows the rules below:

- Spanish phonetic approximation of English sounds
- Syllables divided with hyphens
- No translation – same meaning is preserved through pronunciation only
- Same number of words as the input
"""

import logging
import os

import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
Actúa como experto en pronunciación inglesa para hispanohablantes.

Recibirás una oración en inglés.

Escribe su pronunciación figurada para que un motor TTS
la lea con acento inglés natural.

Reglas:

- Usa pronunciación aproximada en español
- Divide las sílabas con guiones
- No traduzcas
- Mantén el mismo número de palabras

Ejemplo:

Input:
"The challenge was huge"

Output:
"Di chá-lench uoz hiuch"
"""

_DEFAULT_MODEL = "gpt-4o-mini"


class EnglishPhoneticTranscriber:
    """Transcribe English sentences into Spanish phonetic approximations.

    Uses the OpenAI Chat Completions API with a specialised system prompt
    so that the output can be fed into an Argentine TTS engine
    (e.g. ``es-AR-TomasNeural``) while preserving natural English pronunciation.

    Args:
        api_key: OpenAI API key.  If omitted, the value of the
                 ``OPENAI_API_KEY`` environment variable is used.
        model:   Chat model to use.  Defaults to ``"gpt-4o-mini"``.
    """

    def __init__(self, api_key: str | None = None, model: str = _DEFAULT_MODEL) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key is required.  "
                "Set OPENAI_API_KEY in your .env file or pass it explicitly."
            )
        self._client = OpenAI(api_key=resolved_key)
        self._model = model

    def transcribe(self, text: str) -> str:
        """Return the Spanish phonetic approximation of an English *text*.

        Args:
            text: An English sentence or phrase.

        Returns:
            Phonetic representation using Spanish sounds, with syllables
            separated by hyphens.  The number of words matches *text*.

        Raises:
            ValueError: If *text* is empty.
            RuntimeError: If the OpenAI API call fails.
        """
        text = text.strip()
        if not text:
            raise ValueError("Input text must not be empty.")

        logger.debug("Transcribing English text: %r", text)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
            )
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc

        result = response.choices[0].message.content.strip()
        # Strip surrounding quotes that the model sometimes adds
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        logger.debug("Phonetic transcription: %r", result)
        return result
