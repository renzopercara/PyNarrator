"""context.py – Context detection helpers for PyNarrator's dual-vertical engine.

This module is intentionally kept free of heavy dependencies (numpy, MoviePy,
Pillow) so that it can be imported by tests and lightweight utilities without
triggering the full rendering stack.
"""

from __future__ import annotations

import os

from src.config import OUTPUT_DIR

# Maps known context keys to their output filenames.
CONTEXT_OUTPUT_NAMES: dict[str, str] = {
    "smartbuild": "video_final_smartbuild.mp4",
    "esl": "video_final_esl.mp4",
}

DEFAULT_CONTEXT: str = "smartbuild"


def detect_context(script: list[dict]) -> str:
    """Return the dominant context string found in *script*.

    Each scene object may carry an optional ``"contexto"`` field whose value is
    either ``"smartbuild"`` or ``"esl"``.  The function counts occurrences and
    returns the most common value.  When the field is absent from all scenes,
    ``"smartbuild"`` is returned as the safe default so that existing scripts
    that pre-date this field continue to work without modification.

    Args:
        script: List of scene objects loaded from ``script.json``.

    Returns:
        A context string key, e.g. ``"smartbuild"`` or ``"esl"``.
    """
    counts: dict[str, int] = {}
    for item in script:
        ctx = item.get("contexto", "").strip().lower()
        if ctx:
            counts[ctx] = counts.get(ctx, 0) + 1
    if not counts:
        return DEFAULT_CONTEXT
    return max(counts, key=lambda k: counts[k])


def output_path_for_context(context: str) -> str:
    """Return the absolute output file path for the given *context* key.

    Args:
        context: A context key such as ``"smartbuild"`` or ``"esl"``.  Unknown
                 keys are mapped to ``video_final_<context>.mp4``.

    Returns:
        Absolute path inside :data:`src.config.OUTPUT_DIR`.
    """
    filename = CONTEXT_OUTPUT_NAMES.get(context, f"video_final_{context}.mp4")
    return os.path.join(OUTPUT_DIR, filename)
