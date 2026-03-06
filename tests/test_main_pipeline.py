"""tests/test_main_pipeline.py – Unit tests for new main.py pipeline features.

Tests cover:
- Platform resolution constants (_PLATFORM_RESOLUTIONS, _TARGET_RESOLUTION)
- Asset error handling (ColorClip fallback via _BRAND_COLOR)
- Social copy generation (_save_social_post)
- main() target_platform argument updating _TARGET_RESOLUTION
- _enhance_frame: output dtype, shape preservation, NaN safety
- _make_clip_for_scene / _make_video_clip: missing-asset fallback,
  invalid-clip fallback, exception fallback
"""

import os
import sys
import asyncio
import unittest.mock as mock

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as main_module
from main import (
    _BRAND_COLOR,
    _PLATFORM_RESOLUTIONS,
    _TARGET_RESOLUTION,
    _enhance_frame,
    _make_clip_for_scene,
    _make_video_clip,
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


# ---------------------------------------------------------------------------
# _enhance_frame – numpy alignment and safety
# ---------------------------------------------------------------------------


def test_enhance_frame_output_dtype_is_uint8():
    """_enhance_frame must return a uint8 array."""
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = _enhance_frame(frame)
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def test_enhance_frame_preserves_shape():
    """_enhance_frame must return an array with the same shape as the input."""
    frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    result = _enhance_frame(frame)
    assert result.shape == frame.shape, (
        f"Shape mismatch: input={frame.shape} output={result.shape}"
    )


def test_enhance_frame_no_nan_values():
    """_enhance_frame output values must all be within the valid [0, 255] range.

    Verifies that the gamma correction and clipping steps do not produce
    out-of-range pixel values (which would be coerced to NaN/overflow in a
    float pipeline before the final uint8 cast).
    """
    frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = _enhance_frame(frame)
    assert result.min() >= 0, "Pixel values must be >= 0"
    assert result.max() <= 255, "Pixel values must be <= 255"


def test_enhance_frame_all_zeros_input():
    """_enhance_frame must handle an all-zero (black) frame without error."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = _enhance_frame(frame)
    assert result.dtype == np.uint8
    assert result.shape == frame.shape


def test_enhance_frame_all_max_input():
    """_enhance_frame must handle an all-255 (white) frame without error."""
    frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    result = _enhance_frame(frame)
    assert result.dtype == np.uint8
    assert result.shape == frame.shape
    assert result.max() <= 255


# ---------------------------------------------------------------------------
# _make_clip_for_scene – missing asset returns visible ColorClip
# ---------------------------------------------------------------------------


def test_make_clip_for_scene_missing_asset_returns_colorclip(tmp_path):
    """_make_clip_for_scene must return a ColorClip when the asset is missing."""
    from moviepy.editor import ColorClip

    clip = _make_clip_for_scene("/nonexistent/path/video.mp4", duration=3.0)
    assert isinstance(clip, ColorClip), "Expected a ColorClip fallback for missing asset"


def test_make_clip_for_scene_none_asset_returns_colorclip():
    """_make_clip_for_scene(None, …) must return a ColorClip fallback."""
    from moviepy.editor import ColorClip

    clip = _make_clip_for_scene(None, duration=3.0)
    assert isinstance(clip, ColorClip), "Expected a ColorClip fallback for None asset"


def test_make_clip_for_scene_missing_asset_uses_brand_color(tmp_path):
    """The fallback ColorClip must use _BRAND_COLOR, not black."""
    from moviepy.editor import ColorClip

    clip = _make_clip_for_scene("/nonexistent/video.mp4", duration=2.0)
    assert isinstance(clip, ColorClip)
    # MoviePy 1.x stores the colour in clip.img; read the top-left pixel
    color = tuple(int(c) for c in clip.img[0, 0])
    assert color == _BRAND_COLOR, f"Expected brand color {_BRAND_COLOR}, got {color}"


# ---------------------------------------------------------------------------
# _make_video_clip – exception fallback (mocked VideoFileClip)
# ---------------------------------------------------------------------------


def test_make_video_clip_returns_colorclip_when_videofileclip_raises(tmp_path):
    """_make_video_clip must return a visible fallback if VideoFileClip fails."""
    from moviepy.editor import ColorClip

    # Create a real (but invalid/empty) .mp4 file so path existence passes
    fake_mp4 = tmp_path / "fake.mp4"
    fake_mp4.write_bytes(b"not-a-real-video")

    # Patch VideoFileClip to raise an error simulating a corrupt file
    with mock.patch.object(main_module, "VideoFileClip", side_effect=OSError("corrupt")):
        clip = _make_video_clip(str(fake_mp4), duration=3.0, start_time=0, end_time=None)

    assert isinstance(clip, ColorClip), "Expected ColorClip fallback when VideoFileClip raises"


def test_make_video_clip_returns_colorclip_for_zero_dimension_clip(tmp_path):
    """_make_video_clip must return a visible fallback for zero-dimension clips."""
    from moviepy.editor import ColorClip

    fake_mp4 = tmp_path / "fake.mp4"
    fake_mp4.write_bytes(b"not-a-real-video")

    # Simulate a clip with zero width (invalid).  Wire the mock chain so that
    # .without_mask().set_opacity() returns the same object, giving us a
    # single mock whose attributes we can control and inspect.
    mock_clip = mock.MagicMock()
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 0
    mock_clip.h = 0
    mock_clip.duration = 5.0
    mock_clip.size = (0, 0)

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        clip = _make_video_clip(str(fake_mp4), duration=3.0, start_time=0, end_time=None)

    assert isinstance(clip, ColorClip), "Expected ColorClip fallback for zero-dimension clip"
    mock_clip.close.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_video_source – URL detection and download
# ---------------------------------------------------------------------------


def test_resolve_video_source_returns_empty_string_unchanged():
    """_resolve_video_source('') must return '' without attempting a download."""
    from main import _resolve_video_source

    assert _resolve_video_source("") == ""


def test_resolve_video_source_local_path_returned_as_absolute(tmp_path):
    """_resolve_video_source with a local file path resolves it to absolute."""
    from main import _resolve_video_source

    local_file = tmp_path / "clip.mp4"
    local_file.write_bytes(b"")
    result = _resolve_video_source(str(local_file))
    assert os.path.isabs(result), "Expected an absolute path"
    assert result == str(local_file.resolve())


def test_resolve_video_source_detects_http_url(tmp_path, monkeypatch):
    """_resolve_video_source must call download_video for http:// URLs."""
    import main as main_module
    from main import _resolve_video_source

    downloaded_file = tmp_path / "youtube_raw.mp4"
    downloaded_file.write_bytes(b"")
    download_mock = mock.Mock(return_value=str(downloaded_file))
    monkeypatch.setattr(main_module, "download_video", download_mock)

    result = _resolve_video_source("http://example.com/video", tmp_dir=str(tmp_path))

    download_mock.assert_called_once()
    assert os.path.isabs(result)


def test_resolve_video_source_detects_https_url(tmp_path, monkeypatch):
    """_resolve_video_source must call download_video for https:// URLs."""
    import main as main_module
    from main import _resolve_video_source

    downloaded_file = tmp_path / "youtube_raw.mp4"
    downloaded_file.write_bytes(b"")
    download_mock = mock.Mock(return_value=str(downloaded_file))
    monkeypatch.setattr(main_module, "download_video", download_mock)

    result = _resolve_video_source("https://www.youtube.com/watch?v=vHBrMxIEEwY", tmp_dir=str(tmp_path))

    download_mock.assert_called_once_with(
        url="https://www.youtube.com/watch?v=vHBrMxIEEwY",
        output_dir=str(tmp_path),
        filename="youtube_raw.mp4",
    )
    assert result == str(downloaded_file.resolve())


def test_resolve_video_source_returns_url_on_download_failure(tmp_path, monkeypatch):
    """When download_video raises, _resolve_video_source returns the original URL."""
    import main as main_module
    from main import _resolve_video_source

    monkeypatch.setattr(
        main_module, "download_video", mock.Mock(side_effect=RuntimeError("network error"))
    )

    url = "https://www.youtube.com/watch?v=vHBrMxIEEwY"
    result = _resolve_video_source(url, tmp_dir=str(tmp_path))

    # On failure the original URL is returned so the caller's fallback path
    # (ColorClip) is triggered with an appropriate warning already logged.
    assert result == url
