"""tests/test_main_pipeline.py – Unit tests for new main.py pipeline features.

Tests cover:
- Platform resolution constants (_PLATFORM_RESOLUTIONS, _TARGET_RESOLUTION)
- Asset error handling (ColorClip fallback via _BRAND_COLOR)
- Social copy generation (_save_social_post)
- main() target_platform argument updating _TARGET_RESOLUTION
"""

import os
import sys
import asyncio
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module
from main import (
    _BRAND_COLOR,
    _PLATFORM_RESOLUTIONS,
    _TARGET_RESOLUTION,
    _save_social_post,
)


# ---------------------------------------------------------------------------
# Platform resolution constants
# ---------------------------------------------------------------------------


def test_platform_resolutions_contains_instagram():
    assert "Instagram" in _PLATFORM_RESOLUTIONS


def test_platform_resolutions_contains_linkedin():
    assert "LinkedIn" in _PLATFORM_RESOLUTIONS


def test_instagram_resolution():
    assert _PLATFORM_RESOLUTIONS["Instagram"] == (1080, 1920)


def test_linkedin_resolution():
    assert _PLATFORM_RESOLUTIONS["LinkedIn"] == (1080, 1080)


def test_target_resolution_is_tuple_of_two_ints():
    assert isinstance(_TARGET_RESOLUTION, tuple)
    assert len(_TARGET_RESOLUTION) == 2
    assert all(isinstance(v, int) for v in _TARGET_RESOLUTION)


# ---------------------------------------------------------------------------
# Brand colour constant
# ---------------------------------------------------------------------------


def test_brand_color_is_rgb_tuple():
    assert isinstance(_BRAND_COLOR, tuple)
    assert len(_BRAND_COLOR) == 3
    assert all(0 <= c <= 255 for c in _BRAND_COLOR)


# ---------------------------------------------------------------------------
# _save_social_post
# ---------------------------------------------------------------------------


def test_save_social_post_copies_content(tmp_path):
    """_save_social_post should copy content to output/social_post.txt."""
    # Arrange: write a fake INFO_POSTEO.txt
    info_file = tmp_path / "INFO_POSTEO.txt"
    info_file.write_text("Título: Mi Post\n\nHashtags: #Test", encoding="utf-8")

    # Patch OUTPUT_DIR so the social_post.txt lands in tmp_path
    with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
        _save_social_post(str(info_file))

    social_post = tmp_path / "social_post.txt"
    assert social_post.exists(), "social_post.txt should have been created"
    assert social_post.read_text(encoding="utf-8") == info_file.read_text(encoding="utf-8")


def test_save_social_post_missing_source_does_not_raise(tmp_path):
    """_save_social_post should log a warning but not raise when source is missing."""
    with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
        # Should NOT raise even though the source file does not exist.
        _save_social_post(str(tmp_path / "nonexistent.txt"))


def test_save_social_post_preserves_utf8_content(tmp_path):
    """_save_social_post should preserve non-ASCII / emoji content."""
    info_file = tmp_path / "INFO_POSTEO.txt"
    info_file.write_text("🚀 Título: Aprender Español\n#Educación", encoding="utf-8")

    with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
        _save_social_post(str(info_file))

    social_post = tmp_path / "social_post.txt"
    assert "🚀" in social_post.read_text(encoding="utf-8")
    assert "Educación" in social_post.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# main() – target_platform sets _TARGET_RESOLUTION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_sets_instagram_resolution(tmp_path):
    """main('Instagram') should set _TARGET_RESOLUTION to (1080, 1920)."""
    # We only want to test that _TARGET_RESOLUTION is updated; abort early by
    # raising when open('script.json') is called.
    with mock.patch("builtins.open", side_effect=FileNotFoundError("no script")):
        with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError):
                await main_module.main(target_platform="Instagram")

    assert main_module._TARGET_RESOLUTION == (1080, 1920)


@pytest.mark.asyncio
async def test_main_sets_linkedin_resolution(tmp_path):
    """main('LinkedIn') should set _TARGET_RESOLUTION to (1080, 1080)."""
    with mock.patch("builtins.open", side_effect=FileNotFoundError("no script")):
        with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError):
                await main_module.main(target_platform="LinkedIn")

    assert main_module._TARGET_RESOLUTION == (1080, 1080)


@pytest.mark.asyncio
async def test_main_unknown_platform_falls_back_to_video_res(tmp_path):
    """main() with an unknown platform should fall back to VIDEO_RES."""
    with mock.patch("builtins.open", side_effect=FileNotFoundError("no script")):
        with mock.patch.object(main_module, "OUTPUT_DIR", str(tmp_path)):
            with pytest.raises(FileNotFoundError):
                await main_module.main(target_platform="Unknown")

    assert main_module._TARGET_RESOLUTION == main_module.VIDEO_RES


# ---------------------------------------------------------------------------
# Asset error handling – ColorClip fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_falls_back_to_brand_colorclip_when_get_visual_assets_raises(
    tmp_path, monkeypatch
):
    """When get_visual_assets raises, scene clips should use brand ColorClips."""
    import json as _json

    # Minimal legacy script (not micro-learning format)
    script = [{"keyword": "CI/CD", "texto": "Automation is key."}]
    script_path = tmp_path / "script.json"
    script_path.write_text(_json.dumps(script), encoding="utf-8")

    captured_assets = []

    def fake_make_clip_for_scene(asset, duration, zoom_in=True):
        captured_assets.append(asset)
        # Return a mock clip so we don't need MoviePy
        c = mock.MagicMock()
        c.duration = duration
        c.crossfadein.return_value = c
        return c

    # Patch all heavy dependencies
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))

    # get_visual_assets raises → fallback expected
    monkeypatch.setattr(
        main_module, "get_visual_assets", mock.Mock(side_effect=RuntimeError("API error"))
    )

    voice_data_stub = [{"duracion": 3.0}]
    narrator_mock = mock.AsyncMock()
    narrator_mock.generate_voice_overs.return_value = voice_data_stub
    monkeypatch.setattr(
        main_module, "ArgentineNarrator", mock.Mock(return_value=narrator_mock)
    )
    monkeypatch.setattr(main_module, "analyze_tone", mock.Mock(return_value="informative"))
    monkeypatch.setattr(main_module, "_detect_context", mock.Mock(return_value="default"))
    monkeypatch.setattr(
        main_module, "_output_path_for_context", mock.Mock(return_value=str(tmp_path / "out.mp4"))
    )

    colorclip_calls = []

    def spy_colorclip(*args, **kwargs):
        colorclip_calls.append(kwargs.get("color"))
        clip = mock.MagicMock()
        clip.duration = kwargs.get("duration", 3.0)
        clip.crossfadein.return_value = clip
        return clip

    monkeypatch.setattr(main_module, "ColorClip", spy_colorclip)

    # Mock AudioFileClip so MoviePy doesn't try to open missing audio files
    audio_mock = mock.MagicMock()
    audio_mock.volumex.return_value = audio_mock
    monkeypatch.setattr(main_module, "AudioFileClip", mock.Mock(return_value=audio_mock))

    # Stub out the rest of the render pipeline so we don't actually render
    mock_concat = mock.MagicMock()
    mock_concat.duration = 3.0
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))
    monkeypatch.setattr(
        main_module,
        "generate_subtitles",
        mock.Mock(return_value=([], None, [])),
    )
    monkeypatch.setattr(main_module, "_make_hook_clip", mock.Mock(return_value=mock.MagicMock()))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    composite_mock = mock.MagicMock()
    composite_mock.duration = 3.0
    composite_mock.audio = None
    monkeypatch.setattr(
        main_module, "CompositeVideoClip", mock.Mock(return_value=composite_mock)
    )

    # Raise during write_videofile to short-circuit without needing ffmpeg
    composite_mock.write_videofile.side_effect = RuntimeError("abort render")
    composite_mock.set_audio.return_value = composite_mock

    monkeypatch.setattr(
        main_module,
        "generate_social_copy",
        mock.Mock(return_value=str(tmp_path / "INFO_POSTEO.txt")),
    )
    monkeypatch.setattr(main_module, "_save_social_post", mock.Mock())

    with pytest.raises(RuntimeError, match="abort render"):
        await main_module.main(target_platform="Instagram")

    # ColorClip was called with _BRAND_COLOR as the colour
    assert colorclip_calls, "Expected at least one ColorClip to be created as fallback"
    assert all(c == _BRAND_COLOR for c in colorclip_calls)
