"""tests/test_micro_learning_generator.py – Unit tests for src/micro_learning_generator.py."""

import json
import sys
import os

# Ensure the project root is on the path so that `src` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.micro_learning_generator import (
    _KEYWORD_KNOWLEDGE,
    _OPPOSITE_VOICE,
    _TOPIC_KEYWORDS,
    _VOICE_NAMES,
    extract_keywords,
    generate_language_coach_json,
    generate_language_coach_script,
    generate_micro_learning_json,
    generate_micro_learning_script,
    get_keyword_info,
)
from src.esl_narrative_generator import _TOPIC_ALIASES, _VALID_CEFR_LEVELS

_ALL_TOPICS = list(_TOPIC_ALIASES.keys())

# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------


def test_extract_keywords_returns_three_items():
    for topic in _ALL_TOPICS:
        result = extract_keywords(topic)
        assert len(result) == 3, f"Expected 3 keywords for '{topic}', got {len(result)}"


def test_extract_keywords_returns_list_of_strings():
    for topic in _ALL_TOPICS:
        result = extract_keywords(topic)
        for kw in result:
            assert isinstance(kw, str)
            assert kw.strip() != ""


def test_extract_keywords_accepts_aliases():
    assert extract_keywords("pull request") == extract_keywords("pull_requests")
    assert extract_keywords("PR") == extract_keywords("pull_requests")
    assert extract_keywords("code review") == extract_keywords("code_review")
    assert extract_keywords("development") == extract_keywords("software_development")


def test_extract_keywords_case_insensitive():
    assert extract_keywords("GIT") == extract_keywords("git")
    assert extract_keywords("Pull Requests") == extract_keywords("pull_requests")


def test_extract_keywords_invalid_topic_raises():
    with pytest.raises(ValueError):
        extract_keywords("docker")
    with pytest.raises(ValueError):
        extract_keywords("")


def test_extract_keywords_returns_new_list_each_call():
    result1 = extract_keywords("git")
    result1.clear()
    result2 = extract_keywords("git")
    assert len(result2) == 3


# ---------------------------------------------------------------------------
# get_keyword_info
# ---------------------------------------------------------------------------


def test_get_keyword_info_has_definition_and_example():
    for topic in _ALL_TOPICS:
        for kw in extract_keywords(topic):
            info = get_keyword_info(kw)
            assert "definition" in info
            assert "example" in info


def test_get_keyword_info_values_are_non_empty_strings():
    for topic in _ALL_TOPICS:
        for kw in extract_keywords(topic):
            info = get_keyword_info(kw)
            assert isinstance(info["definition"], str) and info["definition"].strip()
            assert isinstance(info["example"], str) and info["example"].strip()


def test_get_keyword_info_returns_copy():
    info1 = get_keyword_info("commit")
    info1["definition"] = "modified"
    info2 = get_keyword_info("commit")
    assert info2["definition"] != "modified"


def test_get_keyword_info_unknown_keyword_raises():
    with pytest.raises(KeyError):
        get_keyword_info("nonexistent_term")


# ---------------------------------------------------------------------------
# generate_micro_learning_script – structure
# ---------------------------------------------------------------------------


def test_script_has_required_top_level_keys():
    script = generate_micro_learning_script("clip.mp4", "git", "B1")
    assert "metadata" in script
    assert "video_source" in script
    assert "scenes" in script


def test_script_video_source_is_preserved():
    script = generate_micro_learning_script("path/to/clip.mp4", "pull requests", "A2")
    assert script["video_source"] == "path/to/clip.mp4"


def test_script_metadata_fields():
    script = generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="H")
    meta = script["metadata"]
    assert meta["tone"] == "INFORMATIVE"
    assert meta["language"] == "en-US"
    assert meta["cefr_level"] == "B1"
    assert meta["narrator_voice"] == "H"


def test_script_metadata_narrator_voice_male():
    script = generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="H")
    assert script["metadata"]["narrator_voice"] == "H"


def test_script_metadata_narrator_voice_female():
    script = generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="M")
    assert script["metadata"]["narrator_voice"] == "M"


def test_script_scenes_count_is_six():
    # 1 original + 1 highlighted + 3 educational + 1 review = 6
    script = generate_micro_learning_script("clip.mp4", "git", "B1")
    assert len(script["scenes"]) == 6


def test_script_scene_types_order():
    script = generate_micro_learning_script("clip.mp4", "pull requests", "B2")
    scenes = script["scenes"]
    assert scenes[0]["type"] == "original"
    assert scenes[1]["type"] == "highlighted"
    assert scenes[-1]["type"] == "review"
    educational = [s for s in scenes if s["type"] == "educational"]
    assert len(educational) == 3


def test_script_original_scene_has_duration_auto():
    script = generate_micro_learning_script("clip.mp4", "git", "A2")
    assert script["scenes"][0]["duration"] == "auto"


def test_script_review_scene_has_duration_auto():
    script = generate_micro_learning_script("clip.mp4", "git", "A2")
    assert script["scenes"][-1]["duration"] == "auto"


def test_script_highlighted_scene_has_keywords():
    script = generate_micro_learning_script("clip.mp4", "git", "B1")
    highlighted = script["scenes"][1]
    assert "keywords" in highlighted
    assert len(highlighted["keywords"]) == 3


def test_script_highlighted_keywords_match_extract_keywords():
    script = generate_micro_learning_script("clip.mp4", "code_review", "B2")
    expected = extract_keywords("code_review")
    assert script["scenes"][1]["keywords"] == expected


def test_script_educational_scenes_have_required_fields():
    script = generate_micro_learning_script("clip.mp4", "pull requests", "C1")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert "term" in scene
            assert "definition" in scene
            assert "example" in scene
            assert "narrator_voice" in scene


# ---------------------------------------------------------------------------
# generate_micro_learning_script – voice switching (SLA best practice)
# ---------------------------------------------------------------------------


def test_educational_voice_is_opposite_of_narrator_male():
    script = generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="H")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert scene["narrator_voice"] == "M"


def test_educational_voice_is_opposite_of_narrator_female():
    script = generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="M")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert scene["narrator_voice"] == "H"


# ---------------------------------------------------------------------------
# generate_micro_learning_script – error handling
# ---------------------------------------------------------------------------


def test_invalid_topic_raises_value_error():
    with pytest.raises(ValueError):
        generate_micro_learning_script("clip.mp4", "docker", "B1")


def test_invalid_cefr_level_raises_value_error():
    with pytest.raises(ValueError):
        generate_micro_learning_script("clip.mp4", "git", "C2")


def test_invalid_narrator_voice_raises_value_error():
    with pytest.raises(ValueError):
        generate_micro_learning_script("clip.mp4", "git", "B1", narrator_voice="X")


# ---------------------------------------------------------------------------
# generate_micro_learning_script – all topic/level combos
# ---------------------------------------------------------------------------


def test_all_topic_level_combinations_produce_valid_script():
    for topic in _ALL_TOPICS:
        for level in _VALID_CEFR_LEVELS:
            script = generate_micro_learning_script("clip.mp4", topic, level)
            assert len(script["scenes"]) == 6
            assert script["metadata"]["cefr_level"] == level


# ---------------------------------------------------------------------------
# generate_micro_learning_json
# ---------------------------------------------------------------------------


def test_json_output_is_valid_json():
    output = generate_micro_learning_json("clip.mp4", "git", "B1")
    parsed = json.loads(output)
    assert isinstance(parsed, dict)


def test_json_output_has_scenes():
    output = generate_micro_learning_json("clip.mp4", "pull requests", "A2")
    parsed = json.loads(output)
    assert "scenes" in parsed
    assert len(parsed["scenes"]) == 6


def test_json_output_matches_script():
    script = generate_micro_learning_script("clip.mp4", "git", "C1")
    json_str = generate_micro_learning_json("clip.mp4", "git", "C1")
    assert json.loads(json_str) == script


def test_json_output_default_narrator_voice():
    output = generate_micro_learning_json("clip.mp4", "git", "B1")
    parsed = json.loads(output)
    assert parsed["metadata"]["narrator_voice"] == "H"


# ---------------------------------------------------------------------------
# _OPPOSITE_VOICE consistency
# ---------------------------------------------------------------------------


def test_opposite_voice_mapping_is_symmetric():
    for key, opposite in _OPPOSITE_VOICE.items():
        assert _OPPOSITE_VOICE[opposite] == key


# ---------------------------------------------------------------------------
# _KEYWORD_KNOWLEDGE – it_example entries
# ---------------------------------------------------------------------------


def test_all_keywords_have_it_example():
    for kw, info in _KEYWORD_KNOWLEDGE.items():
        assert "it_example" in info, f"Missing 'it_example' for keyword {kw!r}"


def test_it_example_values_are_non_empty_strings():
    for kw, info in _KEYWORD_KNOWLEDGE.items():
        val = info["it_example"]
        assert isinstance(val, str), f"'it_example' for {kw!r} is not a string"
        assert val.strip(), f"'it_example' for {kw!r} is empty"


def test_it_example_max_ten_words():
    for kw, info in _KEYWORD_KNOWLEDGE.items():
        word_count = len(info["it_example"].split())
        assert word_count <= 10, (
            f"'it_example' for {kw!r} has {word_count} words (max 10): "
            f"{info['it_example']!r}"
        )


# ---------------------------------------------------------------------------
# generate_language_coach_script – structure
# ---------------------------------------------------------------------------

_SAMPLE_KEYWORDS = ["commit", "branch", "merge"]


def test_coach_script_has_required_top_level_keys():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert "metadata" in script
    assert "video_source" in script
    assert "scenes" in script


def test_coach_script_video_source_is_preserved():
    script = generate_language_coach_script("path/to/clip.mp4", _SAMPLE_KEYWORDS, "A2")
    assert script["video_source"] == "path/to/clip.mp4"


def test_coach_script_metadata_tone_is_professional_tech():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert script["metadata"]["tone"] == "Professional Tech"


def test_coach_script_metadata_language_is_en_us():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert script["metadata"]["language"] == "en-US"


def test_coach_script_metadata_cefr_level():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "C1")
    assert script["metadata"]["cefr_level"] == "C1"


def test_coach_script_metadata_narrator_voice_is_full_name_male():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="H")
    assert script["metadata"]["narrator_voice"] == "en-US-AndrewNeural"


def test_coach_script_metadata_narrator_voice_is_full_name_female():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="M")
    assert script["metadata"]["narrator_voice"] == "en-US-AvaNeural"


def test_coach_script_metadata_glossary_voice_is_opposite_of_narrator_male():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="H")
    assert script["metadata"]["glossary_voice"] == "en-US-AvaNeural"


def test_coach_script_metadata_glossary_voice_is_opposite_of_narrator_female():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="M")
    assert script["metadata"]["glossary_voice"] == "en-US-AndrewNeural"


def test_coach_script_metadata_has_title():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert "title" in script["metadata"]
    assert isinstance(script["metadata"]["title"], str)
    assert script["metadata"]["title"].strip()


def test_coach_script_metadata_title_contains_keywords():
    kws = ["commit", "branch", "merge"]
    script = generate_language_coach_script("clip.mp4", kws, "B1")
    title = script["metadata"]["title"]
    for kw in kws:
        assert kw in title


def test_coach_script_scenes_count_is_six():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert len(script["scenes"]) == 6


def test_coach_script_scene_types_order():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B2")
    scenes = script["scenes"]
    assert scenes[0]["type"] == "original"
    assert scenes[1]["type"] == "highlighted"
    assert scenes[-1]["type"] == "review"
    educational = [s for s in scenes if s["type"] == "educational"]
    assert len(educational) == 3


def test_coach_script_scenes_have_scene_ids():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    for i, scene in enumerate(script["scenes"], start=1):
        assert scene.get("scene_id") == i


# ---------------------------------------------------------------------------
# generate_language_coach_script – scene 1 (original)
# ---------------------------------------------------------------------------


def test_coach_original_scene_has_start_time():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "A2")
    assert "start_time" in script["scenes"][0]


def test_coach_original_scene_has_end_time():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert "end_time" in script["scenes"][0]


def test_coach_original_scene_has_description():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert "description" in script["scenes"][0]
    assert script["scenes"][0]["description"].strip()


# ---------------------------------------------------------------------------
# generate_language_coach_script – scene 2 (highlighted)
# ---------------------------------------------------------------------------


def test_coach_highlighted_scene_has_keywords():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    highlighted = script["scenes"][1]
    assert "keywords" in highlighted
    assert len(highlighted["keywords"]) == 3


def test_coach_highlighted_keywords_match_input():
    kws = ["review", "feedback", "approve"]
    script = generate_language_coach_script("clip.mp4", kws, "B2")
    assert script["scenes"][1]["keywords"] == kws


def test_coach_highlighted_scene_has_description():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    assert "description" in script["scenes"][1]
    assert script["scenes"][1]["description"].strip()


# ---------------------------------------------------------------------------
# generate_language_coach_script – scenes 3–5 (educational)
# ---------------------------------------------------------------------------


def test_coach_educational_scenes_have_required_fields():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "C1")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert "term" in scene
            assert "definition" in scene
            assert "it_example" in scene
            assert "narrator_voice" in scene


def test_coach_educational_scenes_use_it_example_not_example():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert "it_example" in scene
            assert "example" not in scene


def test_coach_educational_narrator_voice_is_full_name():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="H")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert scene["narrator_voice"] == "en-US-AvaNeural"


def test_coach_educational_narrator_voice_opposite_female():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="M")
    for scene in script["scenes"]:
        if scene["type"] == "educational":
            assert scene["narrator_voice"] == "en-US-AndrewNeural"


def test_coach_educational_terms_match_input_keywords():
    kws = ["sprint", "CI/CD", "refactoring"]
    script = generate_language_coach_script("clip.mp4", kws, "B2")
    edu_terms = [s["term"] for s in script["scenes"] if s["type"] == "educational"]
    assert edu_terms == kws


# ---------------------------------------------------------------------------
# generate_language_coach_script – scene 6 (review / Listening Challenge)
# ---------------------------------------------------------------------------


def test_coach_review_scene_has_description():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "A2")
    assert "description" in script["scenes"][-1]


def test_coach_review_scene_has_subtitles_false():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    review_scene = script["scenes"][-1]
    assert review_scene.get("subtitles") is False


# ---------------------------------------------------------------------------
# generate_language_coach_script – error handling
# ---------------------------------------------------------------------------


def test_coach_wrong_keyword_count_raises():
    with pytest.raises(ValueError):
        generate_language_coach_script("clip.mp4", ["commit", "branch"], "B1")
    with pytest.raises(ValueError):
        generate_language_coach_script("clip.mp4", ["commit", "branch", "merge", "rebase"], "B1")


def test_coach_unknown_keyword_raises():
    with pytest.raises(ValueError):
        generate_language_coach_script("clip.mp4", ["commit", "branch", "docker"], "B1")


def test_coach_invalid_cefr_level_raises():
    with pytest.raises(ValueError):
        generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "C2")


def test_coach_invalid_narrator_voice_raises():
    with pytest.raises(ValueError):
        generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "B1", narrator_voice="X")


# ---------------------------------------------------------------------------
# generate_language_coach_json
# ---------------------------------------------------------------------------


def test_coach_json_output_is_valid_json():
    output = generate_language_coach_json("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    parsed = json.loads(output)
    assert isinstance(parsed, dict)


def test_coach_json_output_has_six_scenes():
    output = generate_language_coach_json("clip.mp4", _SAMPLE_KEYWORDS, "A2")
    parsed = json.loads(output)
    assert len(parsed["scenes"]) == 6


def test_coach_json_output_matches_script():
    script = generate_language_coach_script("clip.mp4", _SAMPLE_KEYWORDS, "C1")
    json_str = generate_language_coach_json("clip.mp4", _SAMPLE_KEYWORDS, "C1")
    assert json.loads(json_str) == script


def test_coach_json_subtitles_false_serialised_correctly():
    output = generate_language_coach_json("clip.mp4", _SAMPLE_KEYWORDS, "B1")
    # subtitles: false must appear as JSON false (not string "false")
    parsed = json.loads(output)
    review = parsed["scenes"][-1]
    assert review["subtitles"] is False


# ---------------------------------------------------------------------------
# _VOICE_NAMES consistency
# ---------------------------------------------------------------------------


def test_voice_names_contain_h_and_m():
    assert "H" in _VOICE_NAMES
    assert "M" in _VOICE_NAMES


def test_voice_names_h_is_andrew():
    assert _VOICE_NAMES["H"] == "en-US-AndrewNeural"


def test_voice_names_m_is_ava():
    assert _VOICE_NAMES["M"] == "en-US-AvaNeural"


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
