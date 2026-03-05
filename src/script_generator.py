"""script_generator.py – Convert plain sentences to PyNarrator script JSON.

Takes plain-text sentences and produces a JSON array ready for the video
engine.  Each object in the output contains exactly two fields:

  - ``"texto"``    – the display text (≤ 10 words)
  - ``"fonetica"`` – the phonetic pronunciation (≤ 10 words; defaults to the
                    same value as ``"texto"`` when no override is provided)

Strict rules enforced:
  * Every ``"texto"`` and ``"fonetica"`` value has at most 10 words.
  * Sentences that exceed the limit are automatically split into consecutive
    objects so that no data is lost.
  * Output is valid JSON and nothing else (no extra logging on stdout).

Usage (CLI)::

    python -m src.script_generator "Primera oración." "Segunda oración."

    # Or pipe sentences, one per line:
    echo -e "Primera oración.\nSegunda oración." | python -m src.script_generator

"""

from __future__ import annotations

import json
import sys

_MAX_WORDS: int = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_chunks(text: str, max_words: int = _MAX_WORDS) -> list[str]:
    """Split *text* into chunks of at most *max_words* words.

    Consecutive whitespace is collapsed; leading/trailing whitespace is
    stripped before splitting.

    Args:
        text:      Input text to split.
        max_words: Maximum number of words per chunk (default: 10).

    Returns:
        Non-empty list of strings, each containing at most *max_words* words.
    """
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + max_words])
        for i in range(0, len(words), max_words)
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sentences_to_script(
    sentences: list[str],
    fonetica_overrides: dict[int, str] | None = None,
) -> list[dict]:
    """Convert *sentences* to the PyNarrator script JSON format.

    Each sentence is split into one or more objects whose ``"texto"`` and
    ``"fonetica"`` fields each contain at most :data:`_MAX_WORDS` words.

    Args:
        sentences:          Ordered list of plain-text sentences.
        fonetica_overrides: Optional mapping of sentence index → phonetic
                            override text.  When provided, the override text
                            is used for the ``"fonetica"`` field of all chunks
                            that originate from that sentence (also split to
                            ≤ 10 words).  Sentence indices not present in the
                            mapping fall back to the ``"texto"`` value.

                            When the phonetic override produces a *different*
                            number of chunks than the display text, output
                            length is driven by the **texto** chunk count:
                            excess fonetica chunks are dropped and missing
                            ones are filled by repeating the last available
                            fonetica chunk.

    Returns:
        List of ``{"texto": ..., "fonetica": ...}`` dicts ready for JSON
        serialisation.
    """
    fonetica_overrides = fonetica_overrides or {}
    script: list[dict] = []

    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue

        texto_chunks = _split_into_chunks(sentence)

        # Resolve phonetic text for this sentence
        raw_fonetica = fonetica_overrides.get(idx, sentence).strip()
        fonetica_chunks = _split_into_chunks(raw_fonetica)

        # Pair texto chunks with fonetica chunks positionally.
        # The number of output objects equals the number of texto chunks.
        # If fonetica_overrides produces a different number of chunks the two
        # lists are aligned by position: any excess fonetica chunks beyond the
        # texto count are dropped; any missing fonetica chunks are filled with
        # the last available fonetica chunk (or the full fonetica text when
        # only one fonetica chunk exists).  This keeps the output length
        # predictable (always driven by the display text) while ensuring the
        # 10-word cap is never violated on either field.
        for i, texto in enumerate(texto_chunks):
            fonetica = fonetica_chunks[min(i, len(fonetica_chunks) - 1)]
            script.append({"texto": texto, "fonetica": fonetica})

    return script


def text_to_script(text: str) -> list[dict]:
    """Convert a multi-sentence *text* block to the script JSON format.

    Sentences are detected by splitting on newlines first, then on
    sentence-ending punctuation (``'.'``, ``'!'``, ``'?'``).  Each resulting
    sentence is processed with :func:`sentences_to_script`.

    Args:
        text: Raw multi-sentence text block.

    Returns:
        List of ``{"texto": ..., "fonetica": ...}`` dicts.
    """
    import re  # local import to keep module-level namespace clean

    # Split on newlines first, then on sentence-ending punctuation, keeping
    # the punctuation attached to its sentence.
    raw_sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on '.', '!', '?' while keeping the delimiter
        parts = re.split(r"(?<=[.!?])\s+", line)
        raw_sentences.extend(p.strip() for p in parts if p.strip())

    return sentences_to_script(raw_sentences)


def generate_script_json(
    sentences: list[str],
    fonetica_overrides: dict[int, str] | None = None,
    indent: int = 1,
) -> str:
    """Return a JSON string for *sentences* following the video engine format.

    The output contains **only** the JSON array – no preamble, no logging, no
    trailing newline added beyond what :func:`json.dumps` produces.

    Args:
        sentences:          Ordered list of plain-text sentences.
        fonetica_overrides: Optional phonetic overrides (see
                            :func:`sentences_to_script`).
        indent:             JSON indentation level (default: 1).

    Returns:
        Valid JSON string containing a list of ``{"texto", "fonetica"}`` dicts.
    """
    script = sentences_to_script(sentences, fonetica_overrides)
    return json.dumps(script, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> None:
    """CLI: print the script JSON for the given sentences and exit.

    Sentences can be passed as positional arguments, or one per line via
    stdin when no arguments are provided.

    Output is strictly valid JSON printed to stdout – nothing else.
    """
    args = argv if argv is not None else sys.argv[1:]

    if args:
        sentences = [a for a in args if a.strip()]
    else:
        sentences = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    print(generate_script_json(sentences))


if __name__ == "__main__":
    _main()
