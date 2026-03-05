"""tests/test_esl_story_generator.py – Unit tests for the ESL story generator.

Run with:
    python -m pytest tests/test_esl_story_generator.py -v
"""

import pytest

from src.esl_story_generator import (
    _GENERIC_TEMPLATES,
    _MAX_WORDS_PER_SENTENCE,
    _STORY_TEMPLATES,
    _VALID_LEVELS,
    _format_story,
    _get_story_sentences,
    _validate_sentences,
    _word_count,
    generate_esl_story,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_words(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------------------------
# _word_count
# ---------------------------------------------------------------------------

class TestWordCount:
    def test_basic(self):
        assert _word_count("Hello world") == 2

    def test_single_word(self):
        assert _word_count("Hello") == 1

    def test_ten_words(self):
        sentence = "She goes to work at nine every morning."
        assert _word_count(sentence) == 8

    def test_empty(self):
        assert _word_count("") == 0


# ---------------------------------------------------------------------------
# _validate_sentences
# ---------------------------------------------------------------------------

class TestValidateSentences:
    def test_valid_sentences_pass(self):
        sentences = ["She goes to work at nine.", "The office is in town."]
        _validate_sentences(sentences)  # should not raise

    def test_sentence_at_limit_passes(self):
        # exactly 10 words
        sentence = "She goes to work at nine o'clock every single morning."
        assert _word_count(sentence) == 10
        _validate_sentences([sentence])  # should not raise

    def test_sentence_over_limit_raises(self):
        # 11 words
        sentence = "She goes to work at nine o'clock every single busy morning."
        assert _word_count(sentence) == 11
        with pytest.raises(ValueError, match="11 words"):
            _validate_sentences([sentence])

    def test_error_identifies_sentence_index(self):
        ok = "Short sentence here."
        bad = "This sentence is clearly too long and has more than ten words total."
        with pytest.raises(ValueError, match="Sentence 2"):
            _validate_sentences([ok, bad])


# ---------------------------------------------------------------------------
# Built-in templates: structural integrity
# ---------------------------------------------------------------------------

class TestBuiltinTemplates:
    """All built-in story sentences must respect the 10-word limit."""

    @pytest.mark.parametrize("topic", list(_STORY_TEMPLATES.keys()))
    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_builtin_sentence_word_count(self, topic, level):
        sentences = _STORY_TEMPLATES[topic][level]
        for sentence in sentences:
            count = _count_words(sentence)
            assert count <= _MAX_WORDS_PER_SENTENCE, (
                f"[{topic}/{level}] '{sentence}' has {count} words "
                f"(limit {_MAX_WORDS_PER_SENTENCE})"
            )

    @pytest.mark.parametrize("topic", list(_STORY_TEMPLATES.keys()))
    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_builtin_has_exactly_six_sentences(self, topic, level):
        sentences = _STORY_TEMPLATES[topic][level]
        assert len(sentences) == 6, (
            f"[{topic}/{level}] expected 6 sentences, got {len(sentences)}"
        )

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_generic_template_has_exactly_six_sentences(self, level):
        assert len(_GENERIC_TEMPLATES[level]) == 6

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_generic_template_word_count(self, level):
        for sentence in _GENERIC_TEMPLATES[level]:
            rendered = sentence.format(topic="cooking")
            count = _count_words(rendered)
            assert count <= _MAX_WORDS_PER_SENTENCE, (
                f"[generic/{level}] '{rendered}' has {count} words"
            )


# ---------------------------------------------------------------------------
# _format_story
# ---------------------------------------------------------------------------

class TestFormatStory:
    def test_numbering(self):
        sentences = ["First.", "Second.", "Third.", "Fourth.", "Fifth.", "Sixth."]
        result = _format_story(sentences)
        lines = result.splitlines()
        assert lines[0] == "1. First."
        assert lines[5] == "6. Sixth."

    def test_line_count(self):
        sentences = ["a", "b", "c", "d", "e", "f"]
        result = _format_story(sentences)
        assert len(result.splitlines()) == 6


# ---------------------------------------------------------------------------
# _get_story_sentences
# ---------------------------------------------------------------------------

class TestGetStorySentences:
    def test_known_topic_returns_six_sentences(self):
        sentences = _get_story_sentences("work", "A2")
        assert len(sentences) == 6

    def test_known_topic_partial_match(self):
        # "travelling" should match the "travel" key
        sentences = _get_story_sentences("travelling", "B1")
        assert len(sentences) == 6

    def test_unknown_topic_uses_generic_fallback(self):
        sentences = _get_story_sentences("astronomy", "B2")
        assert len(sentences) == 6
        # Generic fallback fills in the topic placeholder
        assert any("astronomy" in s for s in sentences)

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_fallback_respects_word_limit(self, level):
        sentences = _get_story_sentences("gardening", level)
        for s in sentences:
            assert _count_words(s) <= _MAX_WORDS_PER_SENTENCE


# ---------------------------------------------------------------------------
# generate_esl_story – main public API
# ---------------------------------------------------------------------------

class TestGenerateEslStory:
    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_returns_six_lines(self, level):
        result = generate_esl_story("work", level)
        lines = result.strip().splitlines()
        assert len(lines) == 6

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_lines_are_numbered(self, level):
        result = generate_esl_story("travel", level)
        for i, line in enumerate(result.strip().splitlines(), start=1):
            assert line.startswith(f"{i}. "), (
                f"Line {i} does not start with '{i}. ': {line!r}"
            )

    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_max_words_per_sentence(self, level):
        result = generate_esl_story("food", level)
        for line in result.strip().splitlines():
            # Strip the "N. " prefix before counting
            sentence = line.split(". ", 1)[1]
            count = _count_words(sentence)
            assert count <= _MAX_WORDS_PER_SENTENCE, (
                f"[food/{level}] '{sentence}' has {count} words"
            )

    def test_no_translations_in_output(self):
        """Output must not contain Spanish text (no translations)."""
        result = generate_esl_story("work", "A2")
        spanish_indicators = ["traducción", "traduccion", "fonética", "fonetica"]
        lower = result.lower()
        for indicator in spanish_indicators:
            assert indicator not in lower

    def test_invalid_level_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid level"):
            generate_esl_story("work", "X9")

    def test_invalid_level_message_lists_valid_levels(self):
        with pytest.raises(ValueError) as exc_info:
            generate_esl_story("work", "C2")
        message = str(exc_info.value)
        for valid in _VALID_LEVELS:
            assert valid in message

    def test_empty_topic_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            generate_esl_story("", "A2")

    def test_whitespace_only_topic_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            generate_esl_story("   ", "B1")

    def test_empty_level_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid level"):
            generate_esl_story("work", "")

    def test_level_case_insensitive(self):
        """Lowercase and mixed-case levels should be accepted."""
        result_lower = generate_esl_story("work", "a2")
        result_upper = generate_esl_story("work", "A2")
        assert result_lower == result_upper

    def test_topic_whitespace_stripped(self):
        """Leading/trailing whitespace in topic should not affect results."""
        result_stripped = generate_esl_story("work", "B1")
        result_padded = generate_esl_story("  work  ", "B1")
        assert result_stripped == result_padded

    def test_unknown_topic_fallback(self):
        """An unknown topic should use the generic template, not raise."""
        result = generate_esl_story("astronomy", "C1")
        lines = result.strip().splitlines()
        assert len(lines) == 6

    @pytest.mark.parametrize("topic", list(_STORY_TEMPLATES.keys()))
    @pytest.mark.parametrize("level", sorted(_VALID_LEVELS))
    def test_all_builtin_combinations(self, topic, level):
        """Every built-in topic × level combination must produce 6 valid lines."""
        result = generate_esl_story(topic, level)
        lines = result.strip().splitlines()
        assert len(lines) == 6
        for i, line in enumerate(lines, start=1):
            assert line.startswith(f"{i}. ")
            sentence = line.split(". ", 1)[1]
            assert _count_words(sentence) <= _MAX_WORDS_PER_SENTENCE
