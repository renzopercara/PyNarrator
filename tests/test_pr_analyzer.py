"""tests/test_pr_analyzer.py – Unit tests for src/pr_analyzer.py."""

import sys
import os

# Ensure the project root is on the path so that `src` is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.pr_analyzer import extract_pr_keywords
from src.micro_learning_generator import _KEYWORD_KNOWLEDGE, _TOPIC_KEYWORDS

_FALLBACK_TERMS = list(_TOPIC_KEYWORDS["pull_requests"])

# ---------------------------------------------------------------------------
# extract_pr_keywords – basic contract
# ---------------------------------------------------------------------------


def test_returns_exactly_three_terms_when_content_is_empty():
    result = extract_pr_keywords("", "")
    assert len(result) == 3


def test_returns_list_of_strings():
    result = extract_pr_keywords("add commit message", "fix branch naming")
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)
        assert item.strip() != ""


def test_returns_exactly_three_terms_default():
    result = extract_pr_keywords("commit branch merge review", "fix CI/CD pipeline")
    assert len(result) == 3


def test_all_returned_terms_are_in_knowledge_bank():
    result = extract_pr_keywords("commit and branch merge rebase", "open pull request")
    for term in result:
        assert term in _KEYWORD_KNOWLEDGE, f"Unknown term returned: {term!r}"


# ---------------------------------------------------------------------------
# extract_pr_keywords – keyword detection from content
# ---------------------------------------------------------------------------


def test_detects_commit_in_diff():
    diff = "--- a/main.py\n+++ b/main.py\n@@ commit message format changed"
    result = extract_pr_keywords(diff, "")
    assert "commit" in result


def test_detects_branch_in_description():
    result = extract_pr_keywords("", "Create a feature branch for the new login page")
    assert "branch" in result


def test_detects_merge_in_description():
    result = extract_pr_keywords("", "merge the feature branch into main after review")
    assert "merge" in result


def test_detects_review_in_description():
    result = extract_pr_keywords("", "code review requested for the new API endpoint")
    assert "review" in result


def test_detects_pull_request_in_description():
    result = extract_pr_keywords("", "open a pull request to merge the fix")
    assert "pull request" in result


def test_detects_cicd_in_diff():
    diff = "# CI/CD pipeline config\n+runs-on: ubuntu-latest"
    result = extract_pr_keywords(diff, "")
    assert "CI/CD" in result


def test_case_insensitive_detection():
    result = extract_pr_keywords("COMMIT the changes", "Fix BRANCH naming")
    assert "commit" in result
    assert "branch" in result


# ---------------------------------------------------------------------------
# extract_pr_keywords – ranking by frequency
# ---------------------------------------------------------------------------


def test_higher_frequency_term_ranked_first():
    # "commit" appears 5 times, "branch" appears 1 time
    pr_diff = "commit commit commit commit commit"
    pr_description = "branch"
    result = extract_pr_keywords(pr_diff, pr_description)
    assert result[0] == "commit"


def test_multiple_terms_detected_in_realistic_diff():
    diff = (
        "--- a/git_helper.py\n"
        "+++ b/git_helper.py\n"
        "@@ -1,5 +1,6 @@\n"
        "+def create_commit(message):\n"
        "+    '''Create a commit with the given message.'''\n"
        "+    # rebase branch before committing\n"
    )
    description = "Refactor git helper: add commit and rebase utilities"
    result = extract_pr_keywords(diff, description)
    assert len(result) == 3
    # Both "commit" and "rebase" appear in the content
    assert "commit" in result
    assert "rebase" in result


# ---------------------------------------------------------------------------
# extract_pr_keywords – fallback behaviour
# ---------------------------------------------------------------------------


def test_fallback_fills_to_three_when_content_has_no_terms():
    result = extract_pr_keywords("hello world foo bar baz", "no tech terms here")
    assert len(result) == 3
    # All results must be valid knowledge bank terms
    for term in result:
        assert term in _KEYWORD_KNOWLEDGE


def test_fallback_terms_are_from_pull_requests_bank():
    result = extract_pr_keywords("", "")
    # When nothing is found, the fallback is the pull_requests bank
    for term in result:
        assert term in _FALLBACK_TERMS


def test_fallback_does_not_duplicate_detected_terms():
    # "pull request" is both detected AND in the fallback list
    result = extract_pr_keywords("", "open a pull request for review")
    assert len(result) == len(set(result)), "Returned list contains duplicates"


def test_content_with_one_term_uses_two_fallbacks():
    result = extract_pr_keywords("approve the change", "")
    assert "approve" in result
    # Should have 3 distinct terms
    assert len(result) == 3
    assert len(set(result)) == 3


# ---------------------------------------------------------------------------
# extract_pr_keywords – edge cases
# ---------------------------------------------------------------------------


def test_invalid_max_terms_raises_value_error():
    with pytest.raises(ValueError):
        extract_pr_keywords("commit", "branch", max_terms=0)

    with pytest.raises(ValueError):
        extract_pr_keywords("commit", "branch", max_terms=-1)


def test_max_terms_one_returns_single_term():
    result = extract_pr_keywords("commit message", "fix commit", max_terms=1)
    assert len(result) == 1
    assert "commit" in result


def test_max_terms_two_returns_two_terms():
    result = extract_pr_keywords("commit branch merge", "", max_terms=2)
    assert len(result) == 2


def test_returns_new_list_each_call():
    result1 = extract_pr_keywords("commit branch", "")
    result1.clear()
    result2 = extract_pr_keywords("commit branch", "")
    assert len(result2) > 0


def test_whitespace_only_inputs():
    result = extract_pr_keywords("   ", "   ")
    assert len(result) == 3


def test_unicode_inputs_do_not_crash():
    result = extract_pr_keywords("Añadir commit con mensaje", "branch de feature")
    assert len(result) == 3


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
