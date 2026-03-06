"""pr_analyzer.py – Extract technical IT keywords from a GitHub Pull Request.

Scans the PR diff and description for known high-frequency IT terms and
returns the top three by mention count.  When fewer than three terms are
found in the PR content, the remainder is filled from the canonical
``pull_requests`` topic keyword bank so that the returned list always
contains exactly three terms.

Usage (module)::

    from src.pr_analyzer import extract_pr_keywords

    keywords = extract_pr_keywords(pr_diff, pr_description)
    # e.g. ["commit", "branch", "pull request"]

Usage (CLI)::

    python -m src.pr_analyzer "clips/pr_demo.mp4" B1 \\
        --pr-diff path/to/diff.txt \\
        --pr-description "Fix async handler in login module"

"""

from __future__ import annotations

import re
import sys

# ---------------------------------------------------------------------------
# Internal helpers – imported lazily from the knowledge bank to avoid
# circular-import issues at module load time.
# ---------------------------------------------------------------------------

def _get_known_terms() -> list[str]:
    """Return all IT terms from the keyword knowledge bank, longest first.

    Longest-first ordering prevents shorter sub-strings from matching
    before a longer, more-specific term (e.g. "pull request" before "pull").
    """
    from src.micro_learning_generator import _KEYWORD_KNOWLEDGE  # noqa: PLC0415
    return sorted(_KEYWORD_KNOWLEDGE.keys(), key=lambda t: -len(t))


def _get_fallback_terms() -> list[str]:
    """Return the canonical ``pull_requests`` keywords used as a fallback."""
    from src.micro_learning_generator import _TOPIC_KEYWORDS  # noqa: PLC0415
    return list(_TOPIC_KEYWORDS["pull_requests"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_pr_keywords(
    pr_diff: str,
    pr_description: str,
    max_terms: int = 3,
) -> list[str]:
    """Return up to *max_terms* technical keywords extracted from a pull request.

    The function scans both *pr_diff* and *pr_description* (case-insensitive)
    for any term present in the IT keyword knowledge bank.  Terms are ranked
    by total mention count across both inputs; ties are broken by the
    canonical order of the knowledge bank (alphabetical).  When fewer than
    *max_terms* distinct terms are found, the shortfall is filled from the
    ``pull_requests`` fallback list so that the return value always contains
    exactly *max_terms* elements.

    Args:
        pr_diff:        Raw unified-diff text of the pull request.
        pr_description: Pull request title and body text.
        max_terms:      Maximum number of terms to return (default: 3).

    Returns:
        A list of exactly *max_terms* keyword strings drawn from the IT
        knowledge bank.

    Raises:
        ValueError: If *max_terms* is less than 1.
    """
    if max_terms < 1:
        raise ValueError(f"max_terms must be at least 1, got {max_terms!r}.")

    combined = f"{pr_description}\n{pr_diff}".lower()
    known_terms = _get_known_terms()

    counts: dict[str, int] = {}
    for term in known_terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        matches = re.findall(pattern, combined)
        if matches:
            counts[term] = len(matches)

    # Sort by descending count; ties broken by term length (longer = more specific)
    ranked = sorted(counts.keys(), key=lambda t: (-counts[t], -len(t)))
    selected: list[str] = ranked[:max_terms]

    # Fill any shortfall from the fallback list
    if len(selected) < max_terms:
        fallback = _get_fallback_terms()
        for fallback_term in fallback:
            if fallback_term not in selected:
                selected.append(fallback_term)
            if len(selected) == max_terms:
                break

    return selected[:max_terms]


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:
    """CLI: generate a Language Coach JSON for a PR video clip and PR context.

    Usage::

        python -m src.pr_analyzer <video_source> <cefr_level> \\
            [--pr-diff <text_or_file>] \\
            [--pr-description <text>] \\
            [--voice H|M]

    Output is valid JSON printed to stdout – nothing else.
    """
    import argparse  # noqa: PLC0415

    from src.esl_narrative_generator import _TOPIC_ALIASES, _VALID_CEFR_LEVELS  # noqa: PLC0415
    from src.micro_learning_generator import generate_language_coach_json  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Generate a Language Coach JSON from a PR video clip and PR context.",
    )
    parser.add_argument("video_source", help="Path or URL to the video clip.")
    parser.add_argument(
        "cefr_level",
        help=f"Target CEFR level. One of: {', '.join(_VALID_CEFR_LEVELS)}.",
    )
    parser.add_argument(
        "--pr-diff",
        default="",
        help="Raw unified-diff text (or path to a .txt file containing the diff).",
    )
    parser.add_argument(
        "--pr-description",
        default="",
        help="Pull request title and body text.",
    )
    parser.add_argument(
        "--voice",
        default="H",
        choices=["H", "M"],
        help="Voice key of the original narrator (H=male, M=female). Default: H.",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Allow --pr-diff to be a file path.
    # Heuristic: if the value is shorter than the maximum path length on most
    # operating systems (260 chars on Windows, 4096 on Linux) and does not
    # start with "---" (the unified-diff header), treat it as a file path.
    _MAX_PATH_HEURISTIC = 260
    pr_diff = args.pr_diff
    if pr_diff and not pr_diff.startswith("---") and len(pr_diff) < _MAX_PATH_HEURISTIC:
        import os  # noqa: PLC0415
        if os.path.isfile(pr_diff):
            with open(pr_diff, encoding="utf-8") as fh:
                pr_diff = fh.read()

    keywords = extract_pr_keywords(pr_diff, args.pr_description)

    try:
        output = generate_language_coach_json(
            args.video_source,
            keywords,
            args.cefr_level,
            args.voice,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    _main()
