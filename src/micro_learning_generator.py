"""micro_learning_generator.py – Micro-Learning PR JSON generator for PyNarrator.

Generates a structured JSON script for a 4-scene American English micro-learning
video based on a technical Pull Request clip.

Scene workflow:
  1. original    – Plays the full video clip with standard subtitles.
  2. highlighted – Replays the clip with the 3 top keywords visually highlighted.
  3. educational – One card per keyword: definition + usage example, voiced by the
                   gender opposite to the original narrator (SLA best practice).
  4. review      – Replays the original clip for comprehension validation.

Voice specification (USA Natural):
  Male   (H): en-US-AndrewNeural  – warm, professional, podcast-style.
  Female (M): en-US-AvaNeural     – bright, natural, narrator-style.
  Rate: -5%  |  Pitch: +0Hz  (maximum educational clarity).

Language Coach workflow (6-scene "Educational Sandwich"):
  1. original    – Source clip with the key concept (10–15 s target).
  2. highlighted – Replay with 3 PR-sourced IT terms visually highlighted.
  3–5. educational – Deep-dive card per term: definition + PR-context it_example,
                     voiced by the glossary voice (opposite gender, SLA practice).
  6. review      – Replay without subtitles for Listening Challenge.

Usage (module)::

    from src.micro_learning_generator import generate_micro_learning_script

    script = generate_micro_learning_script(
        video_source="clips/pr_demo.mp4",
        topic="pull requests",
        cefr_level="B1",
        narrator_voice="H",
    )

    # Language Coach (PR-aware):
    from src.micro_learning_generator import generate_language_coach_script
    from src.pr_analyzer import extract_pr_keywords

    keywords = extract_pr_keywords(pr_diff, pr_description)
    coach = generate_language_coach_script(
        video_source="clips/pr_demo.mp4",
        keywords=keywords,
        cefr_level="B1",
        narrator_voice="H",
    )

Usage (CLI)::

    python -m src.micro_learning_generator "clips/pr_demo.mp4" "pull requests" B1
    python -m src.micro_learning_generator "clips/git_demo.mp4" git C1 --voice M

"""

from __future__ import annotations

import json
import sys

from src.esl_narrative_generator import _normalise_topic, _normalise_level, _TOPIC_ALIASES, _VALID_CEFR_LEVELS

# ---------------------------------------------------------------------------
# Keyword bank – top 3 technical terms per topic
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "pull_requests": ["pull request", "branch", "merge"],
    "git": ["commit", "branch", "rebase"],
    "code_review": ["review", "feedback", "approve"],
    "software_development": ["sprint", "CI/CD", "refactoring"],
}

# ---------------------------------------------------------------------------
# Keyword knowledge bank – definition + IT usage example per term
# ---------------------------------------------------------------------------

_KEYWORD_KNOWLEDGE: dict[str, dict[str, str]] = {
    "pull request": {
        "definition": "A request to merge code changes into the main branch for peer review.",
        "example": "Open a pull request so your team can review your new feature before merging.",
        "it_example": "Reviewers check your pull request before it merges.",
    },
    "branch": {
        "definition": "An independent line of development isolated from the main codebase.",
        "example": "Create a branch named 'fix/login-bug' to isolate your bug fix work.",
        "it_example": "Create a feature branch to isolate your changes.",
    },
    "merge": {
        "definition": "The process of integrating changes from one branch into another.",
        "example": "After approval, merge your feature branch into the main branch.",
        "it_example": "Merge only after all reviewers approve the PR.",
    },
    "commit": {
        "definition": "A snapshot of changes saved permanently in the version control history.",
        "example": "Run 'git commit -m \"fix: resolve null pointer\"' to record your changes.",
        "it_example": "Write a clear commit message for each change.",
    },
    "rebase": {
        "definition": "Rewriting commit history by moving commits onto a new base branch.",
        "example": "Rebase your branch onto main to include the latest changes before opening a PR.",
        "it_example": "Rebase your branch on main before opening a PR.",
    },
    "review": {
        "definition": "The process of reading another developer's code to check quality and correctness.",
        "example": "Leave a comment on the pull request to suggest a more efficient algorithm.",
        "it_example": "Leave inline comments during your code review.",
    },
    "feedback": {
        "definition": "Specific, actionable comments provided during a code review to improve the code.",
        "example": "Write clear feedback like 'Extract this logic into a helper function.'",
        "it_example": "Write specific feedback to improve the pull request.",
    },
    "approve": {
        "definition": "A formal sign-off indicating that a reviewer accepts the proposed code changes.",
        "example": "Click 'Approve' after verifying all tests pass and the logic is sound.",
        "it_example": "Click Approve after reviewing all changes carefully.",
    },
    "sprint": {
        "definition": "A fixed-length development cycle, typically one to two weeks, in agile methodology.",
        "example": "Plan which user stories to finish during the two-week sprint planning meeting.",
        "it_example": "Add this PR to close the sprint goal.",
    },
    "CI/CD": {
        "definition": "Automated pipelines for continuously building, testing, and deploying code changes.",
        "example": "The CI/CD pipeline runs tests automatically whenever you push a commit to GitHub.",
        "it_example": "CI/CD runs tests on every pull request commit.",
    },
    "refactoring": {
        "definition": "Restructuring existing code to improve readability without changing its behavior.",
        "example": "Refactor the 200-line function into smaller, well-named helper functions.",
        "it_example": "Open a PR to refactor the login module.",
    },
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_OPPOSITE_VOICE: dict[str, str] = {"H": "M", "M": "H"}


def extract_keywords(topic: str) -> list[str]:
    """Return the 3 canonical technical keywords for *topic*.

    Args:
        topic: Topic string accepted by :func:`src.esl_narrative_generator._normalise_topic`.

    Returns:
        A list of exactly 3 keyword strings.

    Raises:
        ValueError: If *topic* is not recognised.
    """
    topic_key = _normalise_topic(topic)
    return list(_TOPIC_KEYWORDS[topic_key])


def get_keyword_info(keyword: str) -> dict[str, str]:
    """Return the ``definition`` and ``example`` for *keyword*.

    Args:
        keyword: A keyword string as returned by :func:`extract_keywords`.

    Returns:
        A ``{"definition": ..., "example": ...}`` dict.

    Raises:
        KeyError: If *keyword* is not present in the knowledge bank.
    """
    return dict(_KEYWORD_KNOWLEDGE[keyword])


def generate_micro_learning_script(
    video_source: str,
    topic: str,
    cefr_level: str,
    narrator_voice: str = "H",
) -> dict:
    """Return the full micro-learning JSON structure as a Python dict.

    The returned dict follows this schema::

        {
            "metadata": {
                "tone": "INFORMATIVE",
                "language": "en-US",
                "cefr_level": "<level>",
                "narrator_voice": "<H|M>"
            },
            "video_source": "<path_or_url>",
            "scenes": [
                {"type": "original",     "duration": "auto"},
                {"type": "highlighted",  "keywords": [...]},
                {"type": "educational",  "term": "...", "definition": "...", "example": "...",
                 "narrator_voice": "<opposite>"},
                ...  (one per keyword)
                {"type": "review",       "duration": "auto"}
            ]
        }

    The educational scenes use the *opposite* voice to the original narrator –
    a key SLA technique that helps learners separate content from instruction.

    Args:
        video_source:   Path or URL to the source video clip.
        topic:          Topic of the PR clip (e.g. ``"pull requests"``).
        cefr_level:     Target CEFR level (``"A2"``–``"C1"``).
        narrator_voice: Voice key of the original video narrator (``"H"`` or
                        ``"M"``).  Educational scenes will use the opposite key.

    Returns:
        A dict ready to be serialised with :func:`json.dumps`.

    Raises:
        ValueError: If *topic*, *cefr_level*, or *narrator_voice* is invalid.
    """
    topic_key = _normalise_topic(topic)
    level_key = _normalise_level(cefr_level)

    narrator_voice = narrator_voice.strip().upper()
    if narrator_voice not in _OPPOSITE_VOICE:
        raise ValueError(
            f"Invalid narrator_voice {narrator_voice!r}. Must be 'H' (male) or 'M' (female)."
        )
    educational_voice = _OPPOSITE_VOICE[narrator_voice]

    keywords = _TOPIC_KEYWORDS[topic_key]

    scenes: list[dict] = [
        {"type": "original", "duration": "auto"},
        {"type": "highlighted", "keywords": list(keywords)},
    ]

    for kw in keywords:
        info = _KEYWORD_KNOWLEDGE[kw]
        scenes.append(
            {
                "type": "educational",
                "term": kw,
                "definition": info["definition"],
                "example": info["example"],
                "narrator_voice": educational_voice,
            }
        )

    scenes.append({"type": "review", "duration": "auto"})

    return {
        "metadata": {
            "tone": "INFORMATIVE",
            "language": "en-US",
            "cefr_level": level_key,
            "narrator_voice": narrator_voice,
        },
        "video_source": video_source,
        "scenes": scenes,
    }


def generate_micro_learning_json(
    video_source: str,
    topic: str,
    cefr_level: str,
    narrator_voice: str = "H",
    indent: int = 2,
) -> str:
    """Return the micro-learning script as a formatted JSON string.

    This is a convenience wrapper around :func:`generate_micro_learning_script`
    that serialises the result to a JSON string.

    Args:
        video_source:   Path or URL to the source video clip.
        topic:          Topic of the PR clip.
        cefr_level:     Target CEFR level.
        narrator_voice: Voice key of the original narrator (``"H"`` or ``"M"``).
        indent:         JSON indentation level (default: 2).

    Returns:
        A valid JSON string.
    """
    script = generate_micro_learning_script(video_source, topic, cefr_level, narrator_voice)
    return json.dumps(script, ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Language Coach – PR-aware 6-scene "Educational Sandwich"
# ---------------------------------------------------------------------------

#: Maps voice key to the full Azure / Edge-TTS voice name.
_VOICE_NAMES: dict[str, str] = {
    "H": "en-US-AndrewNeural",
    "M": "en-US-AvaNeural",
}


def generate_language_coach_script(
    video_source: str,
    keywords: list[str],
    cefr_level: str,
    narrator_voice: str = "H",
) -> dict:
    """Return a Language Coach micro-lesson JSON structure as a Python dict.

    Implements the "Educational Sandwich" technique (Exposure → Instruction →
    Review) described in the Language Coach prompt specification:

    * **Scene 1** (``"original"``) – Source clip with the key concept.
    * **Scene 2** (``"highlighted"``) – Replay highlighting the 3 PR-sourced IT
      terms.
    * **Scenes 3–5** (``"educational"``) – One deep-dive card per term:
      ``definition`` + ``it_example`` (a ≤10-word PR/Code-Review sentence),
      voiced by the *opposite* gender from the original narrator (SLA best
      practice).
    * **Scene 6** (``"review"``) – Replay without subtitles for Listening
      Challenge.

    The returned dict follows this schema::

        {
            "metadata": {
                "title": "PR Language Coach: <kw1>, <kw2> & <kw3>",
                "tone": "Professional Tech",
                "language": "en-US",
                "cefr_level": "<level>",
                "narrator_voice": "en-US-AndrewNeural",
                "glossary_voice": "en-US-AvaNeural"
            },
            "video_source": "<path_or_url>",
            "scenes": [
                {"scene_id": 1, "type": "original", "start_time": "00:00:00",
                 "end_time": "00:00:15", "description": "Original context clip"},
                {"scene_id": 2, "type": "highlighted", "keywords": [...],
                 "description": "Visual identification of key terms found in this PR"},
                {"scene_id": 3, "type": "educational", "term": "...",
                 "definition": "...", "it_example": "...",
                 "narrator_voice": "en-US-AvaNeural"},
                ...  (one per keyword, scene_id 3-5)
                {"scene_id": 6, "type": "review", "subtitles": false,
                 "description": "Final listening challenge without text"}
            ]
        }

    Args:
        video_source:   Path or URL to the source video clip.
        keywords:       Exactly 3 IT keyword strings drawn from the knowledge
                        bank (as returned by
                        :func:`src.pr_analyzer.extract_pr_keywords`).
        cefr_level:     Target CEFR level (``"A2"``–``"C1"``).
        narrator_voice: Voice key of the original video narrator (``"H"`` or
                        ``"M"``).  Educational scenes will use the opposite key.

    Returns:
        A dict ready to be serialised with :func:`json.dumps`.

    Raises:
        ValueError: If *keywords* does not contain exactly 3 items, any keyword
                    is unknown, *cefr_level* is invalid, or *narrator_voice* is
                    invalid.
    """
    if len(keywords) != 3:
        raise ValueError(
            f"keywords must contain exactly 3 items, got {len(keywords)!r}."
        )

    level_key = _normalise_level(cefr_level)

    narrator_voice = narrator_voice.strip().upper()
    if narrator_voice not in _OPPOSITE_VOICE:
        raise ValueError(
            f"Invalid narrator_voice {narrator_voice!r}. Must be 'H' (male) or 'M' (female)."
        )

    educational_voice_key = _OPPOSITE_VOICE[narrator_voice]
    narrator_voice_name = _VOICE_NAMES[narrator_voice]
    glossary_voice_name = _VOICE_NAMES[educational_voice_key]

    # Validate all keywords exist in the knowledge bank
    for kw in keywords:
        if kw not in _KEYWORD_KNOWLEDGE:
            raise ValueError(
                f"Unknown keyword {kw!r}. "
                f"Valid keywords: {', '.join(sorted(_KEYWORD_KNOWLEDGE))}."
            )

    # Build a short descriptive title from the keywords
    if len(keywords) >= 2:
        title = "PR Language Coach: " + ", ".join(keywords[:-1]) + " & " + keywords[-1]
    else:
        title = "PR Language Coach: " + keywords[0]

    scenes: list[dict] = [
        {
            "scene_id": 1,
            "type": "original",
            "start_time": "00:00:00",
            "end_time": "00:00:15",
            "description": "Original context clip",
        },
        {
            "scene_id": 2,
            "type": "highlighted",
            "keywords": list(keywords),
            "description": "Visual identification of key terms found in this PR",
        },
    ]

    for idx, kw in enumerate(keywords, start=3):
        info = _KEYWORD_KNOWLEDGE[kw]
        scenes.append(
            {
                "scene_id": idx,
                "type": "educational",
                "term": kw,
                "definition": info["definition"],
                "it_example": info["it_example"],
                "narrator_voice": glossary_voice_name,
            }
        )

    scenes.append(
        {
            "scene_id": len(scenes) + 1,
            "type": "review",
            "subtitles": False,
            "description": "Final listening challenge without text",
        }
    )

    return {
        "metadata": {
            "title": title,
            "tone": "Professional Tech",
            "language": "en-US",
            "cefr_level": level_key,
            "narrator_voice": narrator_voice_name,
            "glossary_voice": glossary_voice_name,
        },
        "video_source": video_source,
        "scenes": scenes,
    }


def generate_language_coach_json(
    video_source: str,
    keywords: list[str],
    cefr_level: str,
    narrator_voice: str = "H",
    indent: int = 2,
) -> str:
    """Return the Language Coach script as a formatted JSON string.

    Convenience wrapper around :func:`generate_language_coach_script` that
    serialises the result to a JSON string.

    Args:
        video_source:   Path or URL to the source video clip.
        keywords:       Exactly 3 IT keyword strings (see
                        :func:`generate_language_coach_script`).
        cefr_level:     Target CEFR level.
        narrator_voice: Voice key of the original narrator (``"H"`` or ``"M"``).
        indent:         JSON indentation level (default: 2).

    Returns:
        A valid JSON string.
    """
    script = generate_language_coach_script(video_source, keywords, cefr_level, narrator_voice)
    return json.dumps(script, ensure_ascii=False, indent=indent)


def _main(argv: list[str] | None = None) -> None:
    """CLI: print the micro-learning JSON for a given video clip and topic.

    Usage::

        python -m src.micro_learning_generator <video_source> <topic> <cefr_level> [--voice H|M]

    Output is valid JSON printed to stdout – nothing else.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a micro-learning JSON script for a PR video clip.",
    )
    parser.add_argument("video_source", help="Path or URL to the video clip.")
    parser.add_argument(
        "topic",
        help=f"Topic of the clip. One of: {', '.join(sorted(_TOPIC_ALIASES))} (or alias).",
    )
    parser.add_argument(
        "cefr_level",
        help=f"Target CEFR level. One of: {', '.join(_VALID_CEFR_LEVELS)}.",
    )
    parser.add_argument(
        "--voice",
        default="H",
        choices=["H", "M"],
        help="Voice key of the original narrator (H=male, M=female). Default: H.",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        output = generate_micro_learning_json(
            args.video_source,
            args.topic,
            args.cefr_level,
            args.voice,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    _main()
