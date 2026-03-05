"""tests/test_context_detection.py – Unit tests for context-detection helpers in src/context.py."""

import sys
import os

# Ensure the project root is on the path so that ``src`` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.context import (
    detect_context,
    output_path_for_context,
    DEFAULT_CONTEXT,
    CONTEXT_OUTPUT_NAMES,
)
from src.config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# detect_context
# ---------------------------------------------------------------------------


def test_detect_context_all_smartbuild():
    script = [
        {"contexto": "smartbuild", "texto": "Escena 1"},
        {"contexto": "smartbuild", "texto": "Escena 2"},
    ]
    assert detect_context(script) == "smartbuild"


def test_detect_context_all_esl():
    script = [
        {"contexto": "esl", "texto": "Scene 1"},
        {"contexto": "esl", "texto": "Scene 2"},
    ]
    assert detect_context(script) == "esl"


def test_detect_context_dominant_wins():
    script = [
        {"contexto": "smartbuild", "texto": "Escena 1"},
        {"contexto": "esl", "texto": "Scene 2"},
        {"contexto": "smartbuild", "texto": "Escena 3"},
    ]
    assert detect_context(script) == "smartbuild"


def test_detect_context_missing_field_returns_default():
    script = [
        {"texto": "Sin contexto"},
        {"voz": "H", "texto": "Tampoco"},
    ]
    assert detect_context(script) == DEFAULT_CONTEXT


def test_detect_context_empty_script_returns_default():
    assert detect_context([]) == DEFAULT_CONTEXT


def test_detect_context_case_insensitive():
    script = [{"contexto": "Smartbuild"}, {"contexto": "SMARTBUILD"}]
    assert detect_context(script) == "smartbuild"


def test_detect_context_mixed_present_and_absent():
    # Two scenes have no contexto; one has "esl" → esl wins (1 vs 0).
    script = [
        {"texto": "Sin campo"},
        {"contexto": "esl", "texto": "ESL scene"},
        {"texto": "Otro sin campo"},
    ]
    assert detect_context(script) == "esl"


def test_detect_context_unknown_value_returned_as_is():
    # An unknown context value is returned unchanged (no validation here).
    script = [{"contexto": "newvertical"}, {"contexto": "newvertical"}]
    assert detect_context(script) == "newvertical"


# ---------------------------------------------------------------------------
# output_path_for_context
# ---------------------------------------------------------------------------


def test_output_path_smartbuild_ends_with_expected_filename():
    path = output_path_for_context("smartbuild")
    assert path.endswith("video_final_smartbuild.mp4")
    assert os.path.dirname(path) == OUTPUT_DIR


def test_output_path_esl_ends_with_expected_filename():
    path = output_path_for_context("esl")
    assert path.endswith("video_final_esl.mp4")
    assert os.path.dirname(path) == OUTPUT_DIR


def test_output_path_unknown_context_uses_fallback_naming():
    path = output_path_for_context("newvertical")
    assert path.endswith("video_final_newvertical.mp4")
    assert os.path.dirname(path) == OUTPUT_DIR


def test_output_path_is_absolute():
    for ctx in ("smartbuild", "esl"):
        assert os.path.isabs(output_path_for_context(ctx))


if __name__ == "__main__":
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
        except Exception:  # noqa: BLE001
            print(f"  ERROR {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
