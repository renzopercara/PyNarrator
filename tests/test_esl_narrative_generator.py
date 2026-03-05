"""tests/test_esl_narrative_generator.py – Unit tests for src/esl_narrative_generator.py."""

import sys
import os

# Ensure the project root is on the path so that `src` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.esl_narrative_generator import (
    _MAX_WORDS_PER_SENTENCE,
    _NARRATIVES,
    _SENTENCE_COUNT,
    _TOPIC_ALIASES,
    _VALID_CEFR_LEVELS,
    _normalise_level,
    _normalise_topic,
    format_esl_narrative,
    generate_esl_narrative,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_TOPICS = list(_TOPIC_ALIASES.keys())


# ---------------------------------------------------------------------------
# _normalise_topic
# ---------------------------------------------------------------------------


def test_normalise_topic_canonical_keys():
    for canonical in _TOPIC_ALIASES:
        assert _normalise_topic(canonical) == canonical


def test_normalise_topic_aliases():
    assert _normalise_topic("pull requests") == "pull_requests"
    assert _normalise_topic("PR") == "pull_requests"
    assert _normalise_topic("prs") == "pull_requests"
    assert _normalise_topic("code review") == "code_review"
    assert _normalise_topic("review") == "code_review"
    assert _normalise_topic("software development") == "software_development"
    assert _normalise_topic("development") == "software_development"
    assert _normalise_topic("git") == "git"


def test_normalise_topic_case_insensitive():
    assert _normalise_topic("GIT") == "git"
    assert _normalise_topic("Git") == "git"
    assert _normalise_topic("Pull Requests") == "pull_requests"
    assert _normalise_topic("CODE REVIEW") == "code_review"


def test_normalise_topic_strips_whitespace():
    assert _normalise_topic("  git  ") == "git"
    assert _normalise_topic(" pull requests ") == "pull_requests"


def test_normalise_topic_invalid_raises_value_error():
    with pytest.raises(ValueError):
        _normalise_topic("unknown topic")
    with pytest.raises(ValueError):
        _normalise_topic("")
    with pytest.raises(ValueError):
        _normalise_topic("javascript")


# ---------------------------------------------------------------------------
# _normalise_level
# ---------------------------------------------------------------------------


def test_normalise_level_valid():
    for level in _VALID_CEFR_LEVELS:
        assert _normalise_level(level) == level


def test_normalise_level_case_insensitive():
    assert _normalise_level("a2") == "A2"
    assert _normalise_level("b1") == "B1"
    assert _normalise_level("b2") == "B2"
    assert _normalise_level("c1") == "C1"


def test_normalise_level_strips_whitespace():
    assert _normalise_level("  B1  ") == "B1"


def test_normalise_level_invalid_raises_value_error():
    with pytest.raises(ValueError):
        _normalise_level("C2")
    with pytest.raises(ValueError):
        _normalise_level("A1")
    with pytest.raises(ValueError):
        _normalise_level("")
    with pytest.raises(ValueError):
        _normalise_level("beginner")


# ---------------------------------------------------------------------------
# generate_esl_narrative – count constraints
# ---------------------------------------------------------------------------


def test_generate_returns_exactly_six_sentences():
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            result = generate_esl_narrative(topic, level)
            assert len(result) == _SENTENCE_COUNT, (
                f"Expected {_SENTENCE_COUNT} sentences for ({topic}, {level}), "
                f"got {len(result)}"
            )


def test_each_sentence_at_most_ten_words():
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            result = generate_esl_narrative(topic, level)
            for i, sentence in enumerate(result):
                word_count = len(sentence.split())
                assert word_count <= _MAX_WORDS_PER_SENTENCE, (
                    f"Sentence {i} in ({topic}, {level}) has {word_count} words "
                    f"(max {_MAX_WORDS_PER_SENTENCE}): {sentence!r}"
                )


# ---------------------------------------------------------------------------
# generate_esl_narrative – content constraints
# ---------------------------------------------------------------------------


def test_sentences_are_non_empty_strings():
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            for sentence in generate_esl_narrative(topic, level):
                assert isinstance(sentence, str)
                assert sentence.strip() != ""


def test_sentences_contain_no_numbers():
    """Output must contain only sentences – no numbering."""
    import re

    number_prefix = re.compile(r"^\s*\d+[.)]\s")
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            for sentence in generate_esl_narrative(topic, level):
                assert not number_prefix.match(sentence), (
                    f"Sentence starts with a number in ({topic}, {level}): {sentence!r}"
                )


def test_returns_new_list_each_call():
    """Mutating the returned list must not affect subsequent calls."""
    result1 = generate_esl_narrative("git", "A2")
    result1.clear()
    result2 = generate_esl_narrative("git", "A2")
    assert len(result2) == _SENTENCE_COUNT


# ---------------------------------------------------------------------------
# generate_esl_narrative – error handling
# ---------------------------------------------------------------------------


def test_invalid_topic_raises_value_error():
    with pytest.raises(ValueError):
        generate_esl_narrative("docker", "B1")


def test_invalid_cefr_level_raises_value_error():
    with pytest.raises(ValueError):
        generate_esl_narrative("git", "C2")


def test_invalid_topic_and_level_raises_value_error():
    with pytest.raises(ValueError):
        generate_esl_narrative("blockchain", "D1")


# ---------------------------------------------------------------------------
# generate_esl_narrative – topic aliases
# ---------------------------------------------------------------------------


def test_topic_alias_pull_request_singular():
    assert generate_esl_narrative("pull request", "A2") == generate_esl_narrative(
        "pull_requests", "A2"
    )


def test_topic_alias_pr():
    assert generate_esl_narrative("PR", "B1") == generate_esl_narrative(
        "pull_requests", "B1"
    )


def test_topic_alias_code_review_spaced():
    assert generate_esl_narrative("code review", "B2") == generate_esl_narrative(
        "code_review", "B2"
    )


def test_topic_alias_software_development_spaced():
    assert generate_esl_narrative(
        "software development", "C1"
    ) == generate_esl_narrative("software_development", "C1")


# ---------------------------------------------------------------------------
# format_esl_narrative
# ---------------------------------------------------------------------------


def test_format_esl_narrative_one_sentence_per_line():
    sentences = generate_esl_narrative("git", "B1")
    formatted = format_esl_narrative(sentences)
    lines = formatted.splitlines()
    assert lines == sentences


def test_format_esl_narrative_sentence_count():
    sentences = generate_esl_narrative("code_review", "B2")
    formatted = format_esl_narrative(sentences)
    assert len(formatted.splitlines()) == _SENTENCE_COUNT


def test_format_esl_narrative_no_extra_content():
    """Formatted output must contain only the sentences – nothing else."""
    sentences = generate_esl_narrative("pull_requests", "C1")
    formatted = format_esl_narrative(sentences)
    for sentence in sentences:
        assert sentence in formatted
    # Verify no prefix/suffix outside the sentences themselves
    assert formatted == "\n".join(sentences)


# ---------------------------------------------------------------------------
# Content bank completeness
# ---------------------------------------------------------------------------


def test_all_topic_level_combinations_present():
    """Every (topic, level) combination must have a narrative in the bank."""
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            key = (topic, level)
            assert key in _NARRATIVES, f"Missing narrative for {key}"


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
