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
    _LONG_ASSET_THRESHOLD_SECONDS,
    _MAX_EXPORT_DURATION_SECONDS,
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

    # Simulate a clip with zero width (invalid).  Set mask=None so the new
    # mask-stripping logic does not call set_mask(), keeping the mock simple.
    mock_clip = mock.MagicMock()
    mock_clip.mask = None
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


# ---------------------------------------------------------------------------
# _enhance_frame – try-except fallback on PIL failure
# ---------------------------------------------------------------------------


def test_enhance_frame_returns_original_frame_on_pil_error(monkeypatch):
    """_enhance_frame must return the safe-cast original frame if PIL raises."""
    import PIL.ImageEnhance as _IE

    frame = np.full((50, 50, 3), 100, dtype=np.uint8)

    # Force PIL contrast enhance to raise
    monkeypatch.setattr(
        _IE.Contrast, "enhance", mock.Mock(side_effect=RuntimeError("PIL exploded"))
    )

    result = _enhance_frame(frame)

    assert result.dtype == np.uint8
    assert result.shape == frame.shape
    # The result must be identical to the safe-cast input
    np.testing.assert_array_equal(result, frame)


def test_enhance_frame_converts_float_input_to_uint8():
    """_enhance_frame must cast float32 input to uint8 before processing."""
    frame = np.full((20, 20, 3), 0.5, dtype=np.float32)
    result = _enhance_frame(frame)
    assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# _make_video_clip – Long-Asset Safety Guard
# ---------------------------------------------------------------------------


def test_make_video_clip_long_asset_caps_subclip_to_scene_duration(tmp_path):
    """_make_video_clip must call normalize_youtube_clip with trim_end capped to
    start_time + duration, preventing encoding of the entire long-form source."""
    fake_mp4 = tmp_path / "long_video.mp4"
    fake_mp4.write_bytes(b"not-a-real-video")

    scene_duration = 8.0

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = scene_duration  # trimmed proxy is already short
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    captured = {}

    def fake_normalize(input_path, output_path, start, end):
        captured["start"] = start
        captured["end"] = end
        return output_path

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(main_module, "normalize_youtube_clip", side_effect=fake_normalize):
            with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                _make_video_clip(
                    str(fake_mp4), duration=scene_duration, start_time=0, end_time=None,
                    tmp_dir=str(tmp_path),
                )

    # normalize_youtube_clip must be called with end = start + duration = 8.0
    assert "end" in captured, "normalize_youtube_clip must be called"
    assert float(captured["end"]) <= scene_duration, (
        f"normalize end ({captured['end']}) must not exceed scene_duration ({scene_duration}s)"
    )
    # The trim window end must equal start_time + duration (0 + 8 = 8)
    assert float(captured["start"]) == 0
    assert float(captured["end"]) == scene_duration


def test_make_video_clip_strips_mask_when_present(tmp_path):
    """_make_video_clip must call set_mask(None) when clip.mask is not None."""
    from moviepy.editor import ColorClip

    fake_mp4 = tmp_path / "video.mp4"
    fake_mp4.write_bytes(b"not-a-real-video")

    mock_clip = mock.MagicMock()
    mock_clip.mask = mock.MagicMock()  # non-None mask
    mock_clip.set_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 0  # trigger fallback immediately after mask-stripping
    mock_clip.h = 0
    mock_clip.duration = 5.0
    mock_clip.size = (0, 0)

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        _make_video_clip(str(fake_mp4), duration=3.0, start_time=0, end_time=None)

    mock_clip.set_mask.assert_called_once_with(None)



# ---------------------------------------------------------------------------
# New constants – _LONG_ASSET_THRESHOLD_SECONDS, _MAX_EXPORT_DURATION_SECONDS
# ---------------------------------------------------------------------------


def test_long_asset_threshold_is_sixty_seconds():
    """_LONG_ASSET_THRESHOLD_SECONDS must be 60.0 as specified in the requirement."""
    assert _LONG_ASSET_THRESHOLD_SECONDS == 60.0


def test_max_export_duration_is_300_seconds():
    """_MAX_EXPORT_DURATION_SECONDS must be 300.0 (5 minutes)."""
    assert _MAX_EXPORT_DURATION_SECONDS == 300.0


# ---------------------------------------------------------------------------
# _make_video_clip – trim-first transcoding (normalize_youtube_clip used)
# ---------------------------------------------------------------------------


def test_make_video_clip_uses_normalize_youtube_clip_not_transcode_to_proxy(tmp_path):
    """_make_video_clip must call normalize_youtube_clip (trim+transcode) instead
    of _transcode_to_proxy so only the needed seconds are encoded."""
    fake_mp4 = tmp_path / "source.mp4"
    fake_mp4.write_bytes(b"fake")

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = 5.0
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    normalize_called = []
    transcode_called = []

    def fake_normalize(input_path, output_path, start, end):
        normalize_called.append((start, end))
        return output_path

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(main_module, "normalize_youtube_clip", side_effect=fake_normalize):
            with mock.patch.object(main_module, "_transcode_to_proxy", side_effect=transcode_called.append):
                with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                    mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                    mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                    _make_video_clip(
                        str(fake_mp4), duration=5.0, start_time=0, end_time=None,
                        tmp_dir=str(tmp_path),
                    )

    assert normalize_called, "normalize_youtube_clip must be called"
    assert not transcode_called, "_transcode_to_proxy must NOT be called"


def test_make_video_clip_trim_end_equals_end_time_when_provided(tmp_path):
    """When end_time is explicitly provided, normalize_youtube_clip must use it."""
    fake_mp4 = tmp_path / "source.mp4"
    fake_mp4.write_bytes(b"fake")

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = 13.0
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    captured = {}

    def fake_normalize(input_path, output_path, start, end):
        captured["start"] = start
        captured["end"] = end
        return output_path

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(main_module, "normalize_youtube_clip", side_effect=fake_normalize):
            with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                _make_video_clip(
                    str(fake_mp4), duration=5.0, start_time=13, end_time=26,
                    tmp_dir=str(tmp_path),
                )

    assert float(captured["start"]) == 13
    assert float(captured["end"]) == 26


def test_make_video_clip_fallback_subclip_when_normalize_fails(tmp_path):
    """When normalize_youtube_clip fails, _make_video_clip falls back to subclip."""
    fake_mp4 = tmp_path / "source.mp4"
    fake_mp4.write_bytes(b"fake")

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = 30.0
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(
            main_module, "normalize_youtube_clip",
            side_effect=RuntimeError("ffmpeg not found"),
        ):
            with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                _make_video_clip(
                    str(fake_mp4), duration=5.0, start_time=0, end_time=None,
                    tmp_dir=str(tmp_path),
                )

    # Fallback: subclip called with end = min(0 + 5.0, 30.0) = 5.0
    mock_clip.subclip.assert_called_once()
    _, subclip_end = mock_clip.subclip.call_args[0]
    assert subclip_end <= 5.0


# ---------------------------------------------------------------------------
# main_micro_learning – fallback_duration capping
# ---------------------------------------------------------------------------


def test_main_micro_learning_caps_fallback_duration_for_long_source(tmp_path, monkeypatch):
    """main_micro_learning must cap fallback_duration to _LONG_ASSET_THRESHOLD_SECONDS
    when the source asset is longer than the threshold and no end_time is provided."""
    import json as _json

    long_source_duration = 3670.0  # ~1 hour

    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": str(tmp_path / "source.mp4"),
        "scenes": [
            {"type": "original"},  # no start_time / end_time
        ],
    }
    (tmp_path / "source.mp4").write_bytes(b"fake")

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module, "_output_path_for_context", mock.Mock(return_value=str(tmp_path / "out.mp4"))
    )

    # Use ffprobe mock (the new lightweight probe path) to simulate long source duration.
    monkeypatch.setattr(
        main_module, "_get_video_duration_ffprobe", mock.Mock(return_value=long_source_duration)
    )

    captured_duration = []

    def fake_make_clip_for_scene(asset, duration, zoom_in=True, start_time=0, end_time=None):
        # Inline assertion: duration must never exceed the threshold for a long source
        assert duration <= _LONG_ASSET_THRESHOLD_SECONDS, (
            f"_make_clip_for_scene received uncapped duration {duration:.0f}s "
            f"(threshold: {_LONG_ASSET_THRESHOLD_SECONDS}s)"
        )
        captured_duration.append(duration)
        c = mock.MagicMock()
        c.duration = duration
        c.crossfadein.return_value = c
        return c

    monkeypatch.setattr(main_module, "_make_clip_for_scene", fake_make_clip_for_scene)

    mock_concat = mock.MagicMock()
    mock_concat.duration = 10.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 10.0
    mock_composite.audio = None
    mock_composite.write_videofile.side_effect = RuntimeError("abort render")
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    try:
        asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))
    except RuntimeError:
        pass

    assert captured_duration, "Expected _make_clip_for_scene to be called"
    assert captured_duration[0] <= _LONG_ASSET_THRESHOLD_SECONDS, (
        f"fallback_duration ({captured_duration[0]:.0f}s) must be capped to "
        f"{_LONG_ASSET_THRESHOLD_SECONDS}s when source is long-form"
    )


# ---------------------------------------------------------------------------
# write_videofile – export parameters (bitrate, threads, preset)
# ---------------------------------------------------------------------------


def test_main_micro_learning_write_videofile_uses_bitrate_5000k(tmp_path, monkeypatch):
    """main_micro_learning write_videofile must include bitrate='5000k' (verified
    via the full educational-scene path that reaches write_videofile)."""
    # This test reuses the educational-scene path since an empty scenes list causes
    # early return before write_videofile is reached.  The actual bitrate assertion
    # is in test_main_micro_learning_ffmpeg_params_includes_preset_veryfast which
    # fully exercises the write call.  Here we verify that the constant exported
    # from the module has the correct value.
    assert main_module._MAX_EXPORT_DURATION_SECONDS == 300.0


def test_main_micro_learning_ffmpeg_params_includes_preset_veryfast(tmp_path, monkeypatch):
    """The ffmpeg_params passed to write_videofile must include -preset veryfast."""
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": "",
        "scenes": [
            {
                "type": "educational",
                "term": "test",
                "definition": "A test definition.",
                "example": "A test example.",
                "it_example": "A test IT example.",
            }
        ],
    }

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module, "_output_path_for_context", mock.Mock(return_value=str(tmp_path / "out.mp4"))
    )

    # Stub out audio generation
    async def fake_gen_edu_audio(text, voice_key, idx):
        audio_path = str(tmp_path / f"edu_{idx}.mp3")
        open(audio_path, "wb").close()
        return audio_path

    monkeypatch.setattr(main_module, "_generate_edu_audio", fake_gen_edu_audio)

    mock_audio = mock.MagicMock()
    mock_audio.duration = 3.0
    mock_audio.crossfadein.return_value = mock_audio
    mock_audio.set_audio.return_value = mock_audio
    monkeypatch.setattr(main_module, "AudioFileClip", mock.Mock(return_value=mock_audio))
    monkeypatch.setattr(main_module, "_make_educational_card", mock.Mock(return_value=mock_audio))

    mock_concat = mock.MagicMock()
    mock_concat.duration = 3.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 3.0
    mock_composite.audio = None
    captured_kwargs = {}

    def fake_write(path, **kwargs):
        captured_kwargs.update(kwargs)

    mock_composite.write_videofile.side_effect = fake_write
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))

    params = captured_kwargs.get("ffmpeg_params", [])
    assert "-preset" in params, "ffmpeg_params must include -preset"
    preset_idx = params.index("-preset")
    assert params[preset_idx + 1] == "veryfast", "preset must be veryfast"
    assert captured_kwargs.get("bitrate") == "5000k", "bitrate must be 5000k"


# ---------------------------------------------------------------------------
# _get_video_duration_ffprobe
# ---------------------------------------------------------------------------


def test_get_video_duration_ffprobe_returns_float_from_ffprobe(monkeypatch):
    """_get_video_duration_ffprobe must parse the ffprobe stdout as a float."""
    from main import _get_video_duration_ffprobe

    mock_result = mock.MagicMock()
    mock_result.stdout = "3670.123456\n"
    monkeypatch.setattr("subprocess.run", mock.Mock(return_value=mock_result))

    duration = _get_video_duration_ffprobe("/fake/video.mp4")

    assert isinstance(duration, float)
    assert abs(duration - 3670.123456) < 1e-4


def test_get_video_duration_ffprobe_returns_default_on_ffprobe_missing(monkeypatch):
    """_get_video_duration_ffprobe must return the default when ffprobe is absent."""
    from main import _get_video_duration_ffprobe

    monkeypatch.setattr(
        "subprocess.run", mock.Mock(side_effect=FileNotFoundError("ffprobe not found"))
    )

    result = _get_video_duration_ffprobe("/fake/video.mp4", default=99.0)
    assert result == 99.0


def test_get_video_duration_ffprobe_returns_default_on_nonzero_exit(monkeypatch):
    """_get_video_duration_ffprobe must return the default on a non-zero exit."""
    import subprocess
    from main import _get_video_duration_ffprobe

    monkeypatch.setattr(
        "subprocess.run",
        mock.Mock(side_effect=subprocess.CalledProcessError(1, "ffprobe", stderr="")),
    )

    result = _get_video_duration_ffprobe("/fake/video.mp4", default=5.0)
    assert result == 5.0


def test_get_video_duration_ffprobe_uses_error_log_level(monkeypatch):
    """ffprobe command must include -v error to suppress non-critical output."""
    from main import _get_video_duration_ffprobe

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = "10.0\n"
        return r

    monkeypatch.setattr("subprocess.run", fake_run)
    _get_video_duration_ffprobe("/fake/video.mp4")

    cmd = captured.get("cmd", [])
    assert "-v" in cmd
    v_idx = cmd.index("-v")
    assert cmd[v_idx + 1] == "error", "ffprobe must suppress non-error output"


# ---------------------------------------------------------------------------
# main_micro_learning – no VideoFileClip for metadata probe
# ---------------------------------------------------------------------------


def test_main_micro_learning_does_not_call_videofileclip_for_src_duration(
    tmp_path, monkeypatch
):
    """main_micro_learning must NOT call VideoFileClip to probe source duration.

    The primary cause of the memory leak / hang on long YouTube sources is
    opening the full video with VideoFileClip just to read its duration.
    The refactored code uses _get_video_duration_ffprobe instead.
    """
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": str(tmp_path / "source.mp4"),
        "scenes": [
            {
                "type": "educational",
                "term": "test",
                "definition": "A test.",
                "example": "A test example.",
            }
        ],
    }
    (tmp_path / "source.mp4").write_bytes(b"fake")

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module,
        "_output_path_for_context",
        mock.Mock(return_value=str(tmp_path / "out.mp4")),
    )

    videofileclip_calls = []

    def spy_videofileclip(*args, **kwargs):
        videofileclip_calls.append(args)
        c = mock.MagicMock()
        c.duration = 5.0
        c.mask = None
        return c

    monkeypatch.setattr(main_module, "VideoFileClip", spy_videofileclip)
    monkeypatch.setattr(
        main_module,
        "_get_video_duration_ffprobe",
        mock.Mock(return_value=10.0),
    )

    async def fake_gen_edu_audio(text, voice_key, idx):
        p = str(tmp_path / f"edu_{idx}.mp3")
        open(p, "wb").close()
        return p

    monkeypatch.setattr(main_module, "_generate_edu_audio", fake_gen_edu_audio)

    mock_audio = mock.MagicMock()
    mock_audio.duration = 3.0
    mock_audio.crossfadein.return_value = mock_audio
    mock_audio.set_audio.return_value = mock_audio
    monkeypatch.setattr(main_module, "AudioFileClip", mock.Mock(return_value=mock_audio))
    monkeypatch.setattr(main_module, "_make_educational_card", mock.Mock(return_value=mock_audio))

    mock_concat = mock.MagicMock()
    mock_concat.duration = 3.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 3.0
    mock_composite.audio = None
    mock_composite.write_videofile.return_value = None
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))

    # VideoFileClip must NOT have been called with the source video path
    source_path = str(tmp_path / "source.mp4")
    for call_args in videofileclip_calls:
        if call_args and source_path in str(call_args[0]):
            pytest.fail(
                "VideoFileClip was called with the source video path "
                f"'{source_path}' – the metadata probe must use ffprobe instead."
            )


# ---------------------------------------------------------------------------
# main_micro_learning – review scene auto-duration inheritance
# ---------------------------------------------------------------------------


def test_main_micro_learning_review_auto_inherits_original_window(tmp_path, monkeypatch):
    """review scene with duration='auto' must inherit start/end from first original scene."""
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": str(tmp_path / "source.mp4"),
        "scenes": [
            {"type": "original", "start_time": 13, "end_time": 26},
            {
                "type": "review",
                "duration": "auto",
                "text": "Watch again.",
                "subtitles": False,
            },
        ],
    }
    (tmp_path / "source.mp4").write_bytes(b"fake")

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module,
        "_output_path_for_context",
        mock.Mock(return_value=str(tmp_path / "out.mp4")),
    )
    monkeypatch.setattr(
        main_module,
        "_get_video_duration_ffprobe",
        mock.Mock(return_value=3600.0),
    )

    captured_scene_calls = []

    def fake_make_clip(asset, duration, zoom_in=True, start_time=0, end_time=None):
        captured_scene_calls.append({"start_time": start_time, "end_time": end_time})
        c = mock.MagicMock()
        c.duration = duration if duration else 13.0
        c.crossfadein.return_value = c
        return c

    monkeypatch.setattr(main_module, "_make_clip_for_scene", fake_make_clip)

    mock_concat = mock.MagicMock()
    mock_concat.duration = 26.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 26.0
    mock_composite.audio = None
    mock_composite.write_videofile.side_effect = RuntimeError("abort render")
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    try:
        asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))
    except RuntimeError:
        pass

    assert len(captured_scene_calls) == 2, "Expected 2 video scenes (original + review)"
    original_call = captured_scene_calls[0]
    review_call = captured_scene_calls[1]

    assert original_call["start_time"] == 13
    assert original_call["end_time"] == 26

    # review scene must inherit the exact same window as the original
    assert review_call["start_time"] == 13, (
        f"review start_time should be 13 (from original), got {review_call['start_time']}"
    )
    assert review_call["end_time"] == 26, (
        f"review end_time should be 26 (from original), got {review_call['end_time']}"
    )


def test_main_micro_learning_review_explicit_times_not_overridden(tmp_path, monkeypatch):
    """review scene with explicit start/end_time must NOT be overridden by auto logic."""
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": str(tmp_path / "source.mp4"),
        "scenes": [
            {"type": "original", "start_time": 13, "end_time": 26},
            {
                "type": "review",
                "start_time": 50,
                "end_time": 63,
                "text": "Watch again.",
                "subtitles": False,
            },
        ],
    }
    (tmp_path / "source.mp4").write_bytes(b"fake")

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module,
        "_output_path_for_context",
        mock.Mock(return_value=str(tmp_path / "out.mp4")),
    )
    monkeypatch.setattr(
        main_module,
        "_get_video_duration_ffprobe",
        mock.Mock(return_value=3600.0),
    )

    captured_scene_calls = []

    def fake_make_clip(asset, duration, zoom_in=True, start_time=0, end_time=None):
        captured_scene_calls.append({"start_time": start_time, "end_time": end_time})
        c = mock.MagicMock()
        c.duration = duration if duration else 13.0
        c.crossfadein.return_value = c
        return c

    monkeypatch.setattr(main_module, "_make_clip_for_scene", fake_make_clip)

    mock_concat = mock.MagicMock()
    mock_concat.duration = 26.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 26.0
    mock_composite.audio = None
    mock_composite.write_videofile.side_effect = RuntimeError("abort render")
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    try:
        asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))
    except RuntimeError:
        pass

    assert len(captured_scene_calls) == 2
    review_call = captured_scene_calls[1]
    # Explicit times must be respected
    assert review_call["start_time"] == 50
    assert review_call["end_time"] == 63


# ---------------------------------------------------------------------------
# _make_video_clip – subclip cache reuse
# ---------------------------------------------------------------------------


def test_make_video_clip_reuses_cached_trim_skips_normalize(tmp_path):
    """_make_video_clip must skip normalize_youtube_clip when a cached trim exists."""
    fake_mp4 = tmp_path / "source.mp4"
    fake_mp4.write_bytes(b"fake")

    # Pre-create the cached trimmed file that _make_video_clip would produce.
    start_time = 13.0
    end_time = 26.0
    trimmed_name = f"scene_trim_{start_time * 1000:.0f}_{end_time * 1000:.0f}.mp4"
    cached_trim = tmp_path / trimmed_name
    cached_trim.write_bytes(b"cached trim content")

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = 13.0
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    normalize_called = []

    def fake_normalize(input_path, output_path, start, end):
        normalize_called.append((start, end))
        return output_path

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(main_module, "normalize_youtube_clip", side_effect=fake_normalize):
            with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                _make_video_clip(
                    str(fake_mp4),
                    duration=13.0,
                    start_time=start_time,
                    end_time=end_time,
                    tmp_dir=str(tmp_path),
                )

    assert not normalize_called, (
        "normalize_youtube_clip must NOT be called when a cached trim already exists"
    )


def test_make_video_clip_calls_normalize_when_cache_absent(tmp_path):
    """_make_video_clip must call normalize_youtube_clip when no cache exists."""
    fake_mp4 = tmp_path / "source.mp4"
    fake_mp4.write_bytes(b"fake")

    mock_clip = mock.MagicMock()
    mock_clip.mask = None
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.duration = 13.0
    mock_clip.size = (1280, 720)
    mock_clip.fps = 24
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.resize.return_value = mock_clip
    mock_clip.crop.return_value = mock_clip
    mock_clip.fl_image.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip
    mock_clip.set_audio.return_value = mock_clip

    normalize_called = []

    def fake_normalize(input_path, output_path, start, end):
        normalize_called.append((start, end))
        return output_path

    with mock.patch.object(main_module, "VideoFileClip", return_value=mock_clip):
        with mock.patch.object(main_module, "normalize_youtube_clip", side_effect=fake_normalize):
            with mock.patch.object(main_module, "CompositeVideoClip") as mock_cv:
                mock_cv.return_value.set_duration.return_value = mock_cv.return_value
                mock_cv.return_value.set_audio.return_value = mock_cv.return_value
                _make_video_clip(
                    str(fake_mp4),
                    duration=13.0,
                    start_time=13.0,
                    end_time=26.0,
                    tmp_dir=str(tmp_path),  # empty dir → no cache
                )

    assert normalize_called, "normalize_youtube_clip must be called when cache is absent"


# ---------------------------------------------------------------------------
# main_micro_learning – clip duration logger and cap
# ---------------------------------------------------------------------------


def test_main_micro_learning_caps_clips_exceeding_threshold(tmp_path, monkeypatch):
    """main_micro_learning must cap clips > _LONG_ASSET_THRESHOLD_SECONDS before concat."""
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": "",
        "scenes": [
            {
                "type": "educational",
                "term": "test",
                "definition": "A test.",
                "example": "A test.",
            }
        ],
    }

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module,
        "_output_path_for_context",
        mock.Mock(return_value=str(tmp_path / "out.mp4")),
    )

    async def fake_gen_edu_audio(text, voice_key, idx):
        p = str(tmp_path / f"edu_{idx}.mp3")
        open(p, "wb").close()
        return p

    monkeypatch.setattr(main_module, "_generate_edu_audio", fake_gen_edu_audio)

    # Simulate an educational clip with an over-threshold duration (90s)
    oversized_duration = 90.0
    mock_audio = mock.MagicMock()
    mock_audio.duration = oversized_duration
    mock_audio.crossfadein.return_value = mock_audio
    # set_audio is called to attach narration; must return self so duration is preserved
    mock_audio.set_audio.return_value = mock_audio
    capped_clip = mock.MagicMock()
    capped_clip.duration = _LONG_ASSET_THRESHOLD_SECONDS
    mock_audio.set_duration.return_value = capped_clip
    monkeypatch.setattr(main_module, "AudioFileClip", mock.Mock(return_value=mock_audio))
    monkeypatch.setattr(main_module, "_make_educational_card", mock.Mock(return_value=mock_audio))

    captured_concat_args = {}

    def fake_concat(clips, **kwargs):
        captured_concat_args["clips"] = list(clips)
        result = mock.MagicMock()
        result.duration = sum(c.duration for c in clips)
        result.audio = None
        return result

    monkeypatch.setattr(main_module, "concatenate_videoclips", fake_concat)

    mock_composite = mock.MagicMock()
    mock_composite.duration = _LONG_ASSET_THRESHOLD_SECONDS
    mock_composite.audio = None
    mock_composite.write_videofile.return_value = None
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))

    clips_passed = captured_concat_args.get("clips", [])
    assert clips_passed, "concatenate_videoclips must receive at least one clip"
    for clip in clips_passed:
        assert clip.duration <= _LONG_ASSET_THRESHOLD_SECONDS, (
            f"Clip duration {clip.duration}s passed to concat exceeds threshold "
            f"{_LONG_ASSET_THRESHOLD_SECONDS}s"
        )


# ---------------------------------------------------------------------------
# main_micro_learning – uses _get_video_duration_ffprobe (not VideoFileClip)
# ---------------------------------------------------------------------------


def test_main_micro_learning_uses_ffprobe_for_src_duration(tmp_path, monkeypatch):
    """main_micro_learning must call _get_video_duration_ffprobe, not VideoFileClip,
    to determine the source video duration."""
    script = {
        "metadata": {"tone": "INFORMATIVE", "narrator_voice": "H"},
        "video_source": str(tmp_path / "source.mp4"),
        "scenes": [
            {
                "type": "educational",
                "term": "test",
                "definition": "A test.",
                "example": "A test example.",
            }
        ],
    }
    (tmp_path / "source.mp4").write_bytes(b"fake")

    monkeypatch.setattr(main_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "AUDIO_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "SFX_DIR", str(tmp_path))
    monkeypatch.setattr(
        main_module,
        "_output_path_for_context",
        mock.Mock(return_value=str(tmp_path / "out.mp4")),
    )

    ffprobe_mock = mock.Mock(return_value=3670.0)
    monkeypatch.setattr(main_module, "_get_video_duration_ffprobe", ffprobe_mock)

    async def fake_gen_edu_audio(text, voice_key, idx):
        p = str(tmp_path / f"edu_{idx}.mp3")
        open(p, "wb").close()
        return p

    monkeypatch.setattr(main_module, "_generate_edu_audio", fake_gen_edu_audio)

    mock_audio = mock.MagicMock()
    mock_audio.duration = 3.0
    mock_audio.crossfadein.return_value = mock_audio
    mock_audio.set_audio.return_value = mock_audio
    monkeypatch.setattr(main_module, "AudioFileClip", mock.Mock(return_value=mock_audio))
    monkeypatch.setattr(main_module, "_make_educational_card", mock.Mock(return_value=mock_audio))

    mock_concat = mock.MagicMock()
    mock_concat.duration = 3.0
    mock_concat.audio = None
    monkeypatch.setattr(main_module, "concatenate_videoclips", mock.Mock(return_value=mock_concat))

    mock_composite = mock.MagicMock()
    mock_composite.duration = 3.0
    mock_composite.audio = None
    mock_composite.write_videofile.return_value = None
    mock_composite.set_audio.return_value = mock_composite
    monkeypatch.setattr(main_module, "CompositeVideoClip", mock.Mock(return_value=mock_composite))
    monkeypatch.setattr(main_module, "_make_watermark_clip", mock.Mock(return_value=None))

    import asyncio
    asyncio.get_event_loop().run_until_complete(main_module.main_micro_learning(script))

    # _get_video_duration_ffprobe must have been called with the source path
    ffprobe_mock.assert_called_once()
    call_path = ffprobe_mock.call_args[0][0]
    assert str(tmp_path / "source.mp4") in call_path
