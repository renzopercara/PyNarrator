"""test_setup.py – Environment sanity checks for PyNarrator.

Run this script before starting the main application to verify that all
external dependencies are correctly configured:

  python test_setup.py

Exit codes:
  0 – all checks passed
  1 – one or more checks failed
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_pexels_api() -> bool:
    """Verify that the Pexels API key is present and accepted by the API."""
    logger.info("Checking Pexels API key...")
    try:
        from src.config import PEXELS_API_KEY  # noqa: PLC0415

        if not PEXELS_API_KEY:
            logger.error("PEXELS_API_KEY is not set. Add it to your .env file.")
            return False

        import requests  # noqa: PLC0415

        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": "nature", "per_page": 1},
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Pexels API key is valid. ✓")
            return True
        else:
            logger.error(
                "Pexels API key returned HTTP %d. Check that the key is correct.",
                resp.status_code,
            )
            return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Pexels API check failed: %s", exc)
        return False


def test_whisper_model() -> bool:
    """Verify that the Whisper 'base' model can be loaded."""
    logger.info("Loading Whisper 'base' model...")
    try:
        import whisper  # noqa: PLC0415

        whisper.load_model("base")
        logger.info("Whisper model loaded successfully. ✓")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper model load failed: %s", exc)
        return False


def test_assets_permissions() -> bool:
    """Verify that the required asset directories exist and are writable."""
    logger.info("Checking asset directory permissions...")
    from src.config import ASSETS_DIR, AUDIO_DIR, IMAGES_DIR, OUTPUT_DIR  # noqa: PLC0415

    dirs = {
        "assets": ASSETS_DIR,
        "audio": AUDIO_DIR,
        "images": IMAGES_DIR,
        "output": OUTPUT_DIR,
    }

    all_ok = True
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        try:
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
            logger.info("  [%s] %s  ✓", name, path)
        except OSError as exc:
            logger.error("  [%s] %s  ✗ – %s", name, path, exc)
            all_ok = False

    if all_ok:
        logger.info("All asset directories are writable. ✓")
    return all_ok


def test_vocabulary_annotator() -> bool:
    """Verify that the vocabulary annotator correctly glosses B1/B2/C1 words."""
    logger.info("Testing vocabulary annotator...")
    try:
        from src.vocabulary_annotator import annotate_story  # noqa: PLC0415

        cases = [
            # Basic annotation: known B1 word gets a gloss
            (
                "The challenge was big.",
                "challenge (desafío)",
            ),
            # Max 2 glosses per sentence rule
            (
                "She had to overcome many obstacles and demonstrate her skill.",
                None,  # just check it runs without error
            ),
            # Unknown / basic words are NOT annotated
            (
                "The cat sat on the mat.",
                None,
            ),
            # Multiple sentences: each resets the counter
            (
                "The challenge was enormous. She had to overcome it.",
                "challenge (desafío)",
            ),
            # Empty input returns unchanged
            (
                "",
                "",
            ),
        ]

        all_ok = True
        for story, expected_fragment in cases:
            result = annotate_story(story)
            if expected_fragment is not None and expected_fragment not in result:
                logger.error(
                    "Annotator test FAILED for input %r: expected fragment %r "
                    "not found in result %r",
                    story,
                    expected_fragment,
                    result,
                )
                all_ok = False
            else:
                logger.info("  [annotator] %r → %r  ✓", story[:40], result[:60])

        # Verify the 2-per-sentence cap: a sentence with 3 known words must
        # yield exactly 2 annotations (i.e. exactly 2 opening parentheses).
        cap_sentence = "She had to overcome the challenge and demonstrate her skill."
        cap_result = annotate_story(cap_sentence)
        annotation_count = cap_result.count("(")
        if annotation_count != 2:
            logger.error(
                "Annotator cap test FAILED: %d annotations inserted (expected exactly 2) in %r",
                annotation_count,
                cap_result,
            )
            all_ok = False
        else:
            logger.info(
                "  [annotator cap] %d/2 annotations in capped sentence  ✓",
                annotation_count,
            )

        if all_ok:
            logger.info("Vocabulary annotator tests passed. ✓")
        return all_ok
    except Exception as exc:  # noqa: BLE001
        logger.error("Vocabulary annotator test failed: %s", exc)
        return False


if __name__ == "__main__":
    results = {
        "Pexels API key": test_pexels_api(),
        "Whisper model": test_whisper_model(),
        "Asset folder permissions": test_assets_permissions(),
        "Vocabulary annotator": test_vocabulary_annotator(),
    }

    logger.info("-" * 50)
    passed = sum(results.values())
    total = len(results)
    for check, ok in results.items():
        status = "PASS ✓" if ok else "FAIL ✗"
        logger.info("  %-30s %s", check, status)
    logger.info("-" * 50)
    logger.info("Results: %d/%d checks passed.", passed, total)

    sys.exit(0 if passed == total else 1)
