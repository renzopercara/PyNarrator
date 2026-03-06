"""tests/test_video_engine.py – Unit tests for src/video_engine.py."""

import os
import sys
import types
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_engine import (
    _FFMPEG_PARAMS,
    _TARGET_RESOLUTION,
    _YTDLP_FORMAT,
    build_source_clip,
    download_video,
    render_video,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_ytdlp_format_contains_mp4():
    assert "mp4" in _YTDLP_FORMAT


def test_ytdlp_format_prefers_bestvideo_bestaudio():
    assert "bestvideo" in _YTDLP_FORMAT
    assert "bestaudio" in _YTDLP_FORMAT


def test_target_resolution_is_tuple_of_two_ints():
    assert isinstance(_TARGET_RESOLUTION, tuple)
    assert len(_TARGET_RESOLUTION) == 2
    assert all(isinstance(v, int) for v in _TARGET_RESOLUTION)


def test_ffmpeg_params_includes_pix_fmt_yuv420p():
    assert "-pix_fmt" in _FFMPEG_PARAMS
    idx = _FFMPEG_PARAMS.index("-pix_fmt")
    assert _FFMPEG_PARAMS[idx + 1] == "yuv420p"


def test_ffmpeg_params_includes_faststart():
    assert "-movflags" in _FFMPEG_PARAMS
    idx = _FFMPEG_PARAMS.index("-movflags")
    assert "+faststart" in _FFMPEG_PARAMS[idx + 1]


# ---------------------------------------------------------------------------
# download_video
# ---------------------------------------------------------------------------


def test_download_video_raises_on_empty_url(tmp_path):
    with pytest.raises(ValueError, match="url must not be empty"):
        download_video("", output_dir=str(tmp_path))


def test_download_video_raises_when_ytdlp_missing(tmp_path):
    """RuntimeError when yt-dlp binary is not found."""
    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="yt-dlp is not installed"):
            download_video("https://example.com/video", output_dir=str(tmp_path))


def test_download_video_raises_on_nonzero_exit(tmp_path):
    """RuntimeError when yt-dlp exits with non-zero status."""
    import subprocess

    err = subprocess.CalledProcessError(1, "yt-dlp", stderr="network error")
    with mock.patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="yt-dlp failed"):
            download_video("https://example.com/video", output_dir=str(tmp_path))


def test_download_video_passes_correct_format(tmp_path):
    """The yt-dlp command must include the correct --format argument."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        # Also mock the returned path check – download_video returns the path
        # without verifying its existence.
        download_video("https://example.com/video", output_dir=str(tmp_path))

    cmd = captured["cmd"]
    assert "--format" in cmd
    fmt_idx = cmd.index("--format")
    assert cmd[fmt_idx + 1] == _YTDLP_FORMAT


def test_download_video_creates_output_dir(tmp_path):
    """download_video creates output_dir if it does not exist."""
    new_dir = str(tmp_path / "new" / "nested")

    def fake_run(cmd, **kwargs):
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        download_video("https://example.com/video", output_dir=new_dir)

    assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# build_source_clip
# ---------------------------------------------------------------------------


def test_build_source_clip_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        build_source_clip("/nonexistent/path/video.mp4")


def test_build_source_clip_raises_on_negative_start(tmp_path):
    # Create a dummy file so the existence check passes
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")
    with pytest.raises(ValueError, match="start_time must be non-negative"):
        build_source_clip(str(dummy), start_time=-1)


def test_build_source_clip_raises_when_end_lte_start(tmp_path):
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")
    with pytest.raises(ValueError, match="end_time.*must be greater than start_time"):
        build_source_clip(str(dummy), start_time=10, end_time=5)


def test_build_source_clip_calls_without_mask_and_set_opacity(tmp_path):
    """build_source_clip must call .without_mask().set_opacity(1) on the clip."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 30.0
    mock_clip.w = 1280
    mock_clip.h = 720

    # Chain: without_mask() returns something, set_opacity() returns something,
    # subclip() returns something.
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)

    with mock.patch("src.video_engine.VideoFileClip", mock_vfc_cls, create=True):
        # Patch the import inside the function
        import src.video_engine as ve
        original = getattr(ve, "VideoFileClip", None)
        ve_module = sys.modules.get("moviepy.editor")
        # Use importlib mock
        with mock.patch.dict(
            "sys.modules",
            {"moviepy.editor": types.ModuleType("moviepy.editor")},
        ):
            sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
            result = build_source_clip(str(dummy), start_time=13, end_time=26)

    mock_clip.without_mask.assert_called_once()
    mock_clip.set_opacity.assert_called_once_with(1)


def test_build_source_clip_passes_target_resolution(tmp_path):
    """VideoFileClip must be instantiated with target_resolution."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 30.0
    mock_clip.w = 720
    mock_clip.h = 1280
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    custom_res = (480, 640)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
        build_source_clip(str(dummy), target_resolution=custom_res)

    mock_vfc_cls.assert_called_once_with(str(dummy), target_resolution=custom_res)


# ---------------------------------------------------------------------------
# render_video
# ---------------------------------------------------------------------------


def test_render_video_raises_on_empty_clips():
    with pytest.raises(ValueError, match="clips must not be empty"):
        render_video([], output_path="output/test.mp4")


def test_render_video_uses_compose_method(tmp_path):
    """concatenate_videoclips must be called with method='compose'."""
    output_path = str(tmp_path / "out.mp4")

    mock_final = mock.MagicMock()
    mock_concat = mock.MagicMock(return_value=mock_final)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].concatenate_videoclips = mock_concat  # type: ignore[attr-defined]
        render_video([mock.MagicMock()], output_path=output_path)

    mock_concat.assert_called_once()
    _, kwargs = mock_concat.call_args
    assert kwargs.get("method") == "compose"


def test_render_video_write_includes_pix_fmt(tmp_path):
    """write_videofile ffmpeg_params must include -pix_fmt yuv420p."""
    output_path = str(tmp_path / "out.mp4")

    mock_final = mock.MagicMock()
    mock_concat = mock.MagicMock(return_value=mock_final)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].concatenate_videoclips = mock_concat  # type: ignore[attr-defined]
        render_video([mock.MagicMock()], output_path=output_path)

    mock_final.write_videofile.assert_called_once()
    _, kwargs = mock_final.write_videofile.call_args
    params = kwargs.get("ffmpeg_params", [])
    assert "-pix_fmt" in params
    pix_idx = params.index("-pix_fmt")
    assert params[pix_idx + 1] == "yuv420p"


def test_render_video_creates_output_dir(tmp_path):
    """render_video must create output directory if it does not exist."""
    nested = tmp_path / "a" / "b" / "c"
    output_path = str(nested / "out.mp4")

    mock_final = mock.MagicMock()
    mock_concat = mock.MagicMock(return_value=mock_final)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].concatenate_videoclips = mock_concat  # type: ignore[attr-defined]
        render_video([mock.MagicMock()], output_path=output_path)

    assert os.path.isdir(str(nested))


def test_render_video_custom_ffmpeg_params(tmp_path):
    """Custom ffmpeg_params must override the default params."""
    output_path = str(tmp_path / "out.mp4")
    custom_params = ["-an"]

    mock_final = mock.MagicMock()
    mock_concat = mock.MagicMock(return_value=mock_final)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].concatenate_videoclips = mock_concat  # type: ignore[attr-defined]
        render_video(
            [mock.MagicMock()],
            output_path=output_path,
            ffmpeg_params=custom_params,
        )

    _, kwargs = mock_final.write_videofile.call_args
    assert kwargs["ffmpeg_params"] == custom_params


# ---------------------------------------------------------------------------
# script.json validation
# ---------------------------------------------------------------------------


def test_script_json_start_end_time():
    """script.json original scene must have start_time=13 and end_time=26."""
    import json

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "script.json",
    )
    with open(script_path, encoding="utf-8") as f:
        script = json.load(f)

    original_scenes = [s for s in script["scenes"] if s.get("type") == "original"]
    assert original_scenes, "script.json must have at least one 'original' scene"
    scene = original_scenes[0]
    assert scene["start_time"] == 13
    assert scene["end_time"] == 26


def test_script_json_keywords_are_proposal_merge_branch():
    """script.json highlighted scene must use keywords proposal, merge, branch."""
    import json

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "script.json",
    )
    with open(script_path, encoding="utf-8") as f:
        script = json.load(f)

    highlighted = [s for s in script["scenes"] if s.get("type") == "highlighted"]
    assert highlighted, "script.json must have a 'highlighted' scene"
    kws = highlighted[0]["keywords"]
    assert "proposal" in kws
    assert "merge" in kws
    assert "branch" in kws
    assert "collaborate" not in kws


def test_script_json_educational_scenes_use_ava_voice():
    """All educational scenes in script.json must use en-US-AvaNeural."""
    import json

    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "script.json",
    )
    with open(script_path, encoding="utf-8") as f:
        script = json.load(f)

    edu_scenes = [s for s in script["scenes"] if s.get("type") == "educational"]
    assert edu_scenes, "script.json must have at least one 'educational' scene"
    for scene in edu_scenes:
        assert scene.get("narrator_voice") == "en-US-AvaNeural", (
            f"Scene {scene.get('scene_id')} must use en-US-AvaNeural, "
            f"got {scene.get('narrator_voice')!r}"
        )
