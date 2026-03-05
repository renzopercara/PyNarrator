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


def test_openai_api() -> bool:
    """Verify that the OpenAI API key is present and the phonetic transcriber works."""
    logger.info("Checking OpenAI API key and phonetic transcriber...")
    try:
        from src.config import OPENAI_API_KEY  # noqa: PLC0415

        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set. Add it to your .env file.")
            return False

        from src.phonetic_transcriber import EnglishPhoneticTranscriber  # noqa: PLC0415

        transcriber = EnglishPhoneticTranscriber(api_key=OPENAI_API_KEY)
        result = transcriber.transcribe("The challenge was huge")
        if result:
            logger.info("Phonetic transcriber works. Sample: %r  ✓", result)
            return True
        else:
            logger.error("Phonetic transcriber returned an empty result.")
            return False
    except Exception as exc:  # noqa: BLE001
        logger.error("OpenAI API / phonetic transcriber check failed: %s", exc)
        return False


if __name__ == "__main__":
    results = {
        "Pexels API key": test_pexels_api(),
        "Whisper model": test_whisper_model(),
        "Asset folder permissions": test_assets_permissions(),
        "OpenAI API key / phonetic transcriber": test_openai_api(),
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
