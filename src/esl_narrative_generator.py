"""esl_narrative_generator.py – ESL tech narrative generator for PyNarrator.

Generates minimal technical narratives of exactly 6 sentences targeting
developers learning English as a Second Language (ESL).

Topics covered (case-insensitive): pull requests, git, code review,
software development.
CEFR levels supported: A2, B1, B2, C1.

Rules enforced:

- Exactly 6 sentences per narrative.
- Each sentence contains at most 10 words.
- Language is natural, conversational, and professional IT English.
- Content reflects real situations within a GitHub workflow.
- Output contains ONLY sentences in English (no introductions,
  no translations, no phonetics, no numbering).

Usage (module)::

    from src.esl_narrative_generator import generate_esl_narrative, format_esl_narrative

    sentences = generate_esl_narrative("pull requests", "B1")
    print(format_esl_narrative(sentences))

Usage (CLI)::

    python -m src.esl_narrative_generator "pull requests" B1
    python -m src.esl_narrative_generator git C1

"""

from __future__ import annotations

import sys

_SENTENCE_COUNT: int = 6
_MAX_WORDS_PER_SENTENCE: int = 10

_VALID_CEFR_LEVELS: tuple[str, ...] = ("A2", "B1", "B2", "C1")

# ---------------------------------------------------------------------------
# Narrative content bank
#
# Each key is (topic_key, cefr_level); each value is a list of exactly
# _SENTENCE_COUNT sentences with at most _MAX_WORDS_PER_SENTENCE words each.
# Sentences describe real developer situations within a GitHub workflow and
# are calibrated to the grammatical complexity of the requested CEFR level.
# ---------------------------------------------------------------------------

_NARRATIVES: dict[tuple[str, str], list[str]] = {
    # ── Pull Requests ────────────────────────────────────────────────────────
    ("pull_requests", "A2"): [
        "You push your code to a new branch.",
        "Then you open a pull request on GitHub.",
        "Your team can see your changes there.",
        "They read the code and add comments.",
        "You fix the issues they point out.",
        "Your branch gets merged into the main.",
    ],
    ("pull_requests", "B1"): [
        "A pull request lets you share code changes.",
        "Your teammates review the code before merging.",
        "You should keep pull requests small and focused.",
        "Reviewers leave comments and suggest improvements.",
        "You address feedback and push new commits.",
        "The team approves and merges the branch.",
    ],
    ("pull_requests", "B2"): [
        "Pull requests facilitate peer review in collaborative workflows.",
        "Keeping pull requests focused speeds up code review.",
        "Reviewers assess logic, style, and potential edge cases.",
        "Constructive feedback improves both the code and skills.",
        "Resolved discussions signal that concerns have been addressed.",
        "Squashing commits keeps the main branch history clean.",
    ],
    ("pull_requests", "C1"): [
        "Pull requests embody the principle of incremental reviewed delivery.",
        "Granular commits facilitate precise forensic analysis of regressions.",
        "Reviewers evaluate not only correctness but architectural coherence.",
        "Iterative feedback cycles cultivate a culture of shared ownership.",
        "Automated checks reinforce human review with consistent validation.",
        "Merging only approved code sustains long-term codebase integrity.",
    ],
    # ── Git ──────────────────────────────────────────────────────────────────
    ("git", "A2"): [
        "Git helps you save versions of your code.",
        "You use commit to save your changes.",
        "A branch is a copy of your code.",
        "You can go back to an old version.",
        "Push sends your commits to GitHub.",
        "Pull gets the latest changes from your team.",
    ],
    ("git", "B1"): [
        "Git tracks every change made to your codebase.",
        "You create a branch to work on features.",
        "Commits should describe what and why you changed.",
        "Merging combines your branch with the main branch.",
        "Conflicts happen when two people change the same line.",
        "You resolve conflicts by editing and committing the file.",
    ],
    ("git", "B2"): [
        "Git's distributed model lets everyone work independently offline.",
        "Descriptive commit messages create a meaningful project history.",
        "Rebasing rewrites commit history to keep a linear log.",
        "Feature branches isolate work and prevent conflicts in main.",
        "Git stash temporarily shelves uncommitted changes for context switching.",
        "Tags mark specific commits as stable release points.",
    ],
    ("git", "C1"): [
        "Git's directed acyclic graph models history as immutable snapshots.",
        "Interactive rebasing enables retroactive restructuring of commit granularity.",
        "Bisect leverages binary search to pinpoint regression-introducing commits.",
        "Shallow clones optimize CI pipelines by limiting history depth.",
        "Reflog preserves a local safety net against destructive operations.",
        "Cherry-picking selectively applies commits across divergent branches.",
    ],
    # ── Code Review ──────────────────────────────────────────────────────────
    ("code_review", "A2"): [
        "Code review means reading another developer's code.",
        "You look for bugs and style issues.",
        "Leave comments to explain your suggestions clearly.",
        "Be kind and helpful with your feedback.",
        "The developer can accept or reject changes.",
        "Good reviews help the team write better code.",
    ],
    ("code_review", "B1"): [
        "Code review helps catch bugs before they reach production.",
        "Reviewers check logic, readability, and test coverage.",
        "Good feedback is specific, polite, and actionable.",
        "Authors should explain the reason for each change.",
        "Small pull requests are easier and faster to review.",
        "Approving code is a shared responsibility on the team.",
    ],
    ("code_review", "B2"): [
        "Effective reviews balance thoroughness with team velocity.",
        "Nitpicks should be labeled to clarify their relative priority.",
        "Automated linters reduce cognitive load during manual review.",
        "Blocking comments require resolution before a PR can merge.",
        "Contextual comments explain the reasoning behind design decisions.",
        "Review rotations share ownership and spread knowledge evenly.",
    ],
    ("code_review", "C1"): [
        "Code review enforces collective ownership and architectural consistency.",
        "Distinguishing blocking from non-blocking comments calibrates review expectations.",
        "Asynchronous review demands unambiguous, self-contained change descriptions.",
        "Cognitive biases subtly undermine objective evaluation of unfamiliar patterns.",
        "Pair reviews on critical paths accelerate knowledge transfer significantly.",
        "Iterating on review culture yields compounding improvements in quality.",
    ],
    # ── Software Development ─────────────────────────────────────────────────
    ("software_development", "A2"): [
        "Software development means writing programs on a computer.",
        "Developers work in teams to build applications.",
        "You write code, test it, and fix bugs.",
        "Version control helps you not lose your work.",
        "Sprints are short periods to finish small tasks.",
        "The goal is to ship working software often.",
    ],
    ("software_development", "B1"): [
        "Agile development breaks work into short, focused sprints.",
        "Developers write tests to verify that code works.",
        "Code reviews improve quality before merging new features.",
        "Continuous integration checks that every commit builds correctly.",
        "Good documentation helps teammates understand the codebase faster.",
        "Refactoring improves existing code without changing its behavior.",
    ],
    ("software_development", "B2"): [
        "Iterative delivery reduces risk by exposing working software early.",
        "Test-driven development promotes cleaner interfaces and explicit requirements.",
        "CI/CD pipelines automate build, test, and deployment processes.",
        "Technical debt accumulates when shortcuts bypass best practices.",
        "Observability tools provide insight into production system behavior.",
        "Design patterns address recurring architectural problems with proven solutions.",
    ],
    ("software_development", "C1"): [
        "Evolutionary architecture accommodates change without sacrificing systemic coherence.",
        "Domain-driven design aligns technical models with business language.",
        "Dependency inversion decouples high-level policy from implementation details.",
        "Observability transcends monitoring by enabling arbitrary exploratory queries.",
        "Trunk-based development minimizes integration overhead through continuous merging.",
        "Post-mortems cultivate systemic learning rather than individual accountability.",
    ],
}

# ---------------------------------------------------------------------------
# Topic normalisation
# ---------------------------------------------------------------------------

# Map of canonical topic keys to accepted aliases (all lowercase).
_TOPIC_ALIASES: dict[str, list[str]] = {
    "pull_requests": ["pull request", "pull requests", "pr", "prs", "pull_request"],
    "git": ["git"],
    "code_review": ["code review", "review", "code reviews"],
    "software_development": [
        "software development",
        "development",
        "software",
    ],
}

# Flatten alias map for O(1) lookup: alias → canonical key.
_ALIAS_TO_TOPIC: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _TOPIC_ALIASES.items()
    for alias in aliases
}


def _normalise_topic(topic: str) -> str:
    """Return the canonical topic key for *topic*, or raise ``ValueError``.

    Matching is case-insensitive and trims surrounding whitespace.  Canonical
    keys (e.g. ``"pull_requests"``) are accepted directly in addition to all
    listed aliases.

    Args:
        topic: User-supplied topic string.

    Returns:
        One of ``"pull_requests"``, ``"git"``, ``"code_review"``,
        ``"software_development"``.

    Raises:
        ValueError: If *topic* does not match any known alias.
    """
    key = topic.strip().lower()
    if key in _TOPIC_ALIASES:
        return key
    canonical = _ALIAS_TO_TOPIC.get(key)
    if canonical is None:
        valid = ", ".join(sorted(_TOPIC_ALIASES))
        raise ValueError(
            f"Unknown topic {topic!r}. Valid canonical keys: {valid}. "
            "See _TOPIC_ALIASES for accepted aliases."
        )
    return canonical


def _normalise_level(cefr_level: str) -> str:
    """Return the normalised CEFR level string, or raise ``ValueError``.

    Args:
        cefr_level: User-supplied CEFR level (case-insensitive).

    Returns:
        One of ``"A2"``, ``"B1"``, ``"B2"``, ``"C1"``.

    Raises:
        ValueError: If *cefr_level* is not a supported level.
    """
    normalised = cefr_level.strip().upper()
    if normalised not in _VALID_CEFR_LEVELS:
        valid = ", ".join(_VALID_CEFR_LEVELS)
        raise ValueError(
            f"Unknown CEFR level {cefr_level!r}. Valid levels: {valid}."
        )
    return normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_esl_narrative(topic: str, cefr_level: str) -> list[str]:
    """Return exactly 6 English sentences for *topic* at *cefr_level*.

    Each returned sentence contains at most :data:`_MAX_WORDS_PER_SENTENCE`
    words.  The sentences describe real developer situations within a GitHub
    workflow and are calibrated to the grammatical complexity implied by the
    requested CEFR level.

    Args:
        topic:      Topic of the narrative.  Accepted values (case-insensitive):
                    ``"pull requests"``, ``"git"``, ``"code review"``,
                    ``"software development"`` (and their aliases – see
                    :data:`_TOPIC_ALIASES`).
        cefr_level: Target CEFR level.  One of ``"A2"``, ``"B1"``, ``"B2"``,
                    ``"C1"`` (case-insensitive).

    Returns:
        A list of exactly :data:`_SENTENCE_COUNT` English sentences.

    Raises:
        ValueError: If *topic* or *cefr_level* is not recognised.
    """
    topic_key = _normalise_topic(topic)
    level_key = _normalise_level(cefr_level)
    return list(_NARRATIVES[(topic_key, level_key)])


def format_esl_narrative(sentences: list[str]) -> str:
    """Return *sentences* as a newline-separated string, one sentence per line.

    This is the canonical output format described in the problem statement:
    *only* the sentences appear in the output – no introductions, no
    translations, no phonetics, no numbering.

    Args:
        sentences: List of sentence strings (as returned by
                   :func:`generate_esl_narrative`).

    Returns:
        A string with each sentence on its own line.
    """
    return "\n".join(sentences)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:
    """CLI: print the 6-sentence ESL narrative for the given topic and level.

    Usage::

        python -m src.esl_narrative_generator "pull requests" B1
        python -m src.esl_narrative_generator git C1

    Output is strictly the 6 sentences, one per line – nothing else.
    """
    args = argv if argv is not None else sys.argv[1:]

    if len(args) != 2:
        print(
            "Usage: python -m src.esl_narrative_generator <topic> <cefr_level>",
            file=sys.stderr,
        )
        print(
            f"  topic      : one of {', '.join(sorted(_TOPIC_ALIASES))} (or alias)",
            file=sys.stderr,
        )
        print(
            f"  cefr_level : one of {', '.join(_VALID_CEFR_LEVELS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    topic, cefr_level = args
    try:
        sentences = generate_esl_narrative(topic, cefr_level)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_esl_narrative(sentences))


if __name__ == "__main__":
    _main()
