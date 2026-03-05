"""tests/test_script_generator.py – Unit tests for src/script_generator.py."""

import json
import sys
import os

# Ensure the project root is on the path so that `src` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.script_generator import (
    _MAX_WORDS,
    _split_into_chunks,
    sentences_to_script,
    text_to_script,
    generate_script_json,
)


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------

def test_split_short_text_single_chunk():
    result = _split_into_chunks("Hola mundo")
    assert result == ["Hola mundo"]


def test_split_empty_text():
    assert _split_into_chunks("") == []
    assert _split_into_chunks("   ") == []


def test_split_exact_max_words():
    text = " ".join(f"word{i}" for i in range(_MAX_WORDS))
    result = _split_into_chunks(text)
    assert len(result) == 1
    assert len(result[0].split()) == _MAX_WORDS


def test_split_exceeds_max_words():
    text = " ".join(f"word{i}" for i in range(_MAX_WORDS + 3))
    result = _split_into_chunks(text)
    assert len(result) == 2
    assert len(result[0].split()) == _MAX_WORDS
    assert len(result[1].split()) == 3


def test_split_many_words():
    text = " ".join(f"w{i}" for i in range(25))
    result = _split_into_chunks(text)
    assert len(result) == 3
    for chunk in result:
        assert len(chunk.split()) <= _MAX_WORDS


# ---------------------------------------------------------------------------
# sentences_to_script
# ---------------------------------------------------------------------------

def test_empty_sentences():
    assert sentences_to_script([]) == []
    assert sentences_to_script([""]) == []
    assert sentences_to_script(["   "]) == []


def test_single_short_sentence():
    result = sentences_to_script(["Hola mundo."])
    assert len(result) == 1
    assert result[0]["texto"] == "Hola mundo."
    assert result[0]["fonetica"] == "Hola mundo."


def test_required_fields_present():
    result = sentences_to_script(["Una oración de prueba."])
    assert "texto" in result[0]
    assert "fonetica" in result[0]


def test_no_extra_fields():
    result = sentences_to_script(["Texto de ejemplo."])
    assert set(result[0].keys()) == {"texto", "fonetica"}


def test_max_words_enforced_texto():
    long_sentence = " ".join(f"palabra{i}" for i in range(15))
    result = sentences_to_script([long_sentence])
    for item in result:
        assert len(item["texto"].split()) <= _MAX_WORDS


def test_max_words_enforced_fonetica():
    long_sentence = " ".join(f"palabra{i}" for i in range(15))
    result = sentences_to_script([long_sentence])
    for item in result:
        assert len(item["fonetica"].split()) <= _MAX_WORDS


def test_long_sentence_splits_into_multiple_objects():
    long_sentence = " ".join(f"word{i}" for i in range(25))
    result = sentences_to_script([long_sentence])
    assert len(result) == 3


def test_multiple_sentences():
    sentences = [
        "Primera oración corta.",
        "Segunda oración también corta.",
    ]
    result = sentences_to_script(sentences)
    assert len(result) == 2
    assert result[0]["texto"] == "Primera oración corta."
    assert result[1]["texto"] == "Segunda oración también corta."


def test_fonetica_override_applied():
    result = sentences_to_script(
        ["Smartbuild Construcciones."],
        fonetica_overrides={0: "Smartbild Construcciones."},
    )
    assert result[0]["texto"] == "Smartbuild Construcciones."
    assert result[0]["fonetica"] == "Smartbild Construcciones."


def test_fonetica_override_fewer_chunks_pads_last():
    # texto has 11 words → 2 chunks; fonetica has 1 word → 1 chunk.
    # The second texto chunk should reuse the single fonetica chunk.
    long_texto = "uno dos tres cuatro cinco seis siete ocho nueve diez once"
    short_fonetica = "fonema"
    result = sentences_to_script([long_texto], fonetica_overrides={0: short_fonetica})
    assert len(result) == 2
    assert result[0]["fonetica"] == "fonema"
    assert result[1]["fonetica"] == "fonema"
    for item in result:
        assert len(item["texto"].split()) <= _MAX_WORDS


def test_fonetica_defaults_to_texto():
    result = sentences_to_script(["Sin override."])
    assert result[0]["texto"] == result[0]["fonetica"]


def test_all_words_preserved_across_chunks():
    words = [f"w{i}" for i in range(25)]
    sentence = " ".join(words)
    result = sentences_to_script([sentence])
    recovered = " ".join(item["texto"] for item in result)
    assert recovered == sentence


# ---------------------------------------------------------------------------
# text_to_script
# ---------------------------------------------------------------------------

def test_text_single_line():
    result = text_to_script("Hola mundo.")
    assert len(result) == 1
    assert result[0]["texto"] == "Hola mundo."


def test_text_multiline():
    text = "Primera línea.\nSegunda línea."
    result = text_to_script(text)
    assert len(result) == 2


def test_text_empty():
    assert text_to_script("") == []
    assert text_to_script("   \n  ") == []


# ---------------------------------------------------------------------------
# generate_script_json
# ---------------------------------------------------------------------------

def test_generate_script_json_is_valid_json():
    output = generate_script_json(["Hola mundo."])
    parsed = json.loads(output)
    assert isinstance(parsed, list)


def test_generate_script_json_structure():
    output = generate_script_json(["Una prueba de JSON."])
    parsed = json.loads(output)
    assert len(parsed) == 1
    assert "texto" in parsed[0]
    assert "fonetica" in parsed[0]


def test_generate_script_json_no_extra_text():
    output = generate_script_json(["Hola."])
    # The output must parse cleanly without stripping any prefix/suffix
    json.loads(output)


def test_generate_script_json_respects_max_words():
    long_sentence = " ".join(f"palabra{i}" for i in range(20))
    output = generate_script_json([long_sentence])
    parsed = json.loads(output)
    for item in parsed:
        assert len(item["texto"].split()) <= _MAX_WORDS
        assert len(item["fonetica"].split()) <= _MAX_WORDS


def test_generate_script_json_multiple_sentences():
    sentences = ["Oración uno.", "Oración dos.", "Oración tres."]
    output = generate_script_json(sentences)
    parsed = json.loads(output)
    assert len(parsed) == 3


if __name__ == "__main__":
    # Run all tests manually when executed as a script
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
        except Exception:  # noqa: BLE001 – surface unexpected errors too
            print(f"  ERROR {name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
