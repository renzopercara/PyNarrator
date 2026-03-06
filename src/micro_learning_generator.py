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

Usage (module)::

    from src.micro_learning_generator import generate_micro_learning_script

    script = generate_micro_learning_script(
        video_source="clips/pr_demo.mp4",
        topic="pull requests",
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
    },
    "branch": {
        "definition": "An independent line of development isolated from the main codebase.",
        "example": "Create a branch named 'fix/login-bug' to isolate your bug fix work.",
    },
    "merge": {
        "definition": "The process of integrating changes from one branch into another.",
        "example": "After approval, merge your feature branch into the main branch.",
    },
    "commit": {
        "definition": "A snapshot of changes saved permanently in the version control history.",
        "example": "Run 'git commit -m \"fix: resolve null pointer\"' to record your changes.",
    },
    "rebase": {
        "definition": "Rewriting commit history by moving commits onto a new base branch.",
        "example": "Rebase your branch onto main to include the latest changes before opening a PR.",
    },
    "review": {
        "definition": "The process of reading another developer's code to check quality and correctness.",
        "example": "Leave a comment on the pull request to suggest a more efficient algorithm.",
    },
    "feedback": {
        "definition": "Specific, actionable comments provided during a code review to improve the code.",
        "example": "Write clear feedback like 'Extract this logic into a helper function.'",
    },
    "approve": {
        "definition": "A formal sign-off indicating that a reviewer accepts the proposed code changes.",
        "example": "Click 'Approve' after verifying all tests pass and the logic is sound.",
    },
    "sprint": {
        "definition": "A fixed-length development cycle, typically one to two weeks, in agile methodology.",
        "example": "Plan which user stories to finish during the two-week sprint planning meeting.",
    },
    "CI/CD": {
        "definition": "Automated pipelines for continuously building, testing, and deploying code changes.",
        "example": "The CI/CD pipeline runs tests automatically whenever you push a commit to GitHub.",
    },
    "refactoring": {
        "definition": "Restructuring existing code to improve readability without changing its behavior.",
        "example": "Refactor the 200-line function into smaller, well-named helper functions.",
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
# CLI entry-point
# ---------------------------------------------------------------------------


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
