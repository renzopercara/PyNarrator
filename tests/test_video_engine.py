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
    _TMP_DIR,
    _YTDLP_FORMAT,
    _transcode_to_proxy,
    build_source_clip,
    download_video,
    normalize_youtube_clip,
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
    expected_path = str(tmp_path / "source.mp4")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate yt-dlp writing the file so the post-download existence
        # check in download_video passes without error.
        open(expected_path, "wb").close()
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        download_video("https://example.com/video", output_dir=str(tmp_path))

    cmd = captured["cmd"]
    assert "--format" in cmd
    fmt_idx = cmd.index("--format")
    assert cmd[fmt_idx + 1] == _YTDLP_FORMAT


def test_download_video_creates_output_dir(tmp_path):
    """download_video creates output_dir if it does not exist."""
    new_dir = str(tmp_path / "new" / "nested")
    expected_path = os.path.join(new_dir, "source.mp4")

    def fake_run(cmd, **kwargs):
        # Simulate yt-dlp writing the file so the post-download check passes.
        open(expected_path, "wb").close()
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        download_video("https://example.com/video", output_dir=new_dir)

    assert os.path.isdir(new_dir)


def test_download_video_returns_absolute_path(tmp_path):
    """download_video must return an absolute path regardless of input."""
    expected_path = str(tmp_path / "source.mp4")

    def fake_run(cmd, **kwargs):
        open(expected_path, "wb").close()
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        returned = download_video("https://example.com/video", output_dir=str(tmp_path))

    assert os.path.isabs(returned), "download_video must return an absolute path"


def test_download_video_falls_back_to_newest_mp4_when_expected_path_missing(tmp_path):
    """When the expected output path is not found, download_video should fall
    back to the newest .mp4 in the output directory (yt-dlp renamed the file)."""
    alternate_path = str(tmp_path / "vHBrMxIEEwY.mp4")

    def fake_run(cmd, **kwargs):
        # yt-dlp wrote to a different name than requested
        open(alternate_path, "wb").close()
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        returned = download_video(
            "https://example.com/video",
            output_dir=str(tmp_path),
            filename="source.mp4",   # expected name that won't be created
        )

    assert returned == alternate_path


def test_download_video_raises_when_no_mp4_found_after_download(tmp_path):
    """RuntimeError is raised when no .mp4 exists in output_dir after download."""
    def fake_run(cmd, **kwargs):
        # yt-dlp runs but writes nothing
        result = mock.MagicMock()
        result.stdout = ""
        return result

    with mock.patch("subprocess.run", side_effect=fake_run):
        with pytest.raises(RuntimeError, match="Downloaded file not found"):
            download_video("https://example.com/video", output_dir=str(tmp_path))


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

    # Chain: without_mask() → set_opacity() → set_fps() → subclip() → mock_clip
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

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
            sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
            # transcode_proxy=False avoids calling ffmpeg in unit tests
            result = build_source_clip(str(dummy), start_time=13, end_time=26, transcode_proxy=False)

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
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    custom_res = (480, 640)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
        # transcode_proxy=False avoids calling ffmpeg in unit tests
        build_source_clip(str(dummy), target_resolution=custom_res, transcode_proxy=False)

    mock_vfc_cls.assert_called_once_with(str(dummy), target_resolution=custom_res)


def test_build_source_clip_calls_set_fps(tmp_path):
    """build_source_clip must call .set_fps(fps) on the clip."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 30.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
        build_source_clip(str(dummy), fps=24, transcode_proxy=False)

    mock_clip.set_fps.assert_called_once_with(24)


def test_build_source_clip_uses_composite_with_black_background(tmp_path):
    """build_source_clip must wrap the clip in CompositeVideoClip over a black ColorClip."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 13.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    mock_color_cls = mock.MagicMock()
    mock_bg = mock.MagicMock()
    mock_color_cls.return_value = mock_bg
    mock_bg.set_duration.return_value = mock_bg
    mock_composite_cls = mock.MagicMock()

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].ColorClip = mock_color_cls  # type: ignore[attr-defined]
        sys.modules["moviepy.editor"].CompositeVideoClip = mock_composite_cls  # type: ignore[attr-defined]
        build_source_clip(str(dummy), transcode_proxy=False)

    # ColorClip must be constructed with a black colour
    mock_color_cls.assert_called_once()
    color_kwargs = mock_color_cls.call_args[1]
    assert color_kwargs.get("color") == (0, 0, 0), "Background must be black (0,0,0)"

    # CompositeVideoClip must be called with [bg, clip] layers
    mock_composite_cls.assert_called_once()
    layers = mock_composite_cls.call_args[0][0]
    assert layers[0] is mock_bg, "First layer must be the black background"


def test_build_source_clip_transcode_proxy_calls_ffmpeg(tmp_path):
    """When transcode_proxy=True, build_source_clip must call ffmpeg via subprocess."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    proxy = str(tmp_path / "proxy.mp4")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 13.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        with mock.patch("tempfile.NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.__enter__.return_value.name = proxy
            mock_ntf.return_value.__exit__ = mock.MagicMock(return_value=False)
            with mock.patch.dict(
                "sys.modules",
                {"moviepy.editor": types.ModuleType("moviepy.editor")},
            ):
                sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
                sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
                sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
                build_source_clip(str(dummy), transcode_proxy=True)

    assert "ffmpeg" in captured["cmd"], "ffmpeg must be called for proxy transcoding"
    assert "-profile:v" in captured["cmd"]
    idx = captured["cmd"].index("-profile:v")
    assert captured["cmd"][idx + 1] == "baseline"


def test_transcode_to_proxy_raises_on_missing_ffmpeg(tmp_path):
    """_transcode_to_proxy must raise RuntimeError when ffmpeg is not found."""
    dummy = str(tmp_path / "video.mp4")
    (tmp_path / "video.mp4").write_bytes(b"")

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            _transcode_to_proxy(dummy)


def test_transcode_to_proxy_raises_on_nonzero_exit(tmp_path):
    """_transcode_to_proxy must raise RuntimeError when ffmpeg exits non-zero."""
    import subprocess as _sp

    dummy = str(tmp_path / "video.mp4")
    (tmp_path / "video.mp4").write_bytes(b"")

    err = _sp.CalledProcessError(1, "ffmpeg", stderr="codec error")
    with mock.patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="ffmpeg proxy transcoding failed"):
            _transcode_to_proxy(dummy)


def test_transcode_to_proxy_ffmpeg_cmd_uses_libx264_baseline(tmp_path):
    """_transcode_to_proxy must call ffmpeg with libx264/baseline/yuv420p."""
    dummy = str(tmp_path / "video.mp4")
    (tmp_path / "video.mp4").write_bytes(b"")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        _transcode_to_proxy(dummy)

    cmd = captured["cmd"]
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-profile:v" in cmd and cmd[cmd.index("-profile:v") + 1] == "baseline"
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"


# ---------------------------------------------------------------------------
# _TMP_DIR constant
# ---------------------------------------------------------------------------


def test_tmp_dir_constant_points_to_assets_tmp():
    assert os.path.normpath(_TMP_DIR) == os.path.normpath(os.path.join("assets", "tmp"))


# ---------------------------------------------------------------------------
# normalize_youtube_clip
# ---------------------------------------------------------------------------


def test_normalize_youtube_clip_raises_on_missing_ffmpeg(tmp_path):
    """normalize_youtube_clip must raise RuntimeError when ffmpeg is not found."""
    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    output_path = str(tmp_path / "normalized.mp4")

    with mock.patch("subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            normalize_youtube_clip(input_path, output_path, "13", "26")


def test_normalize_youtube_clip_raises_on_nonzero_exit(tmp_path):
    """normalize_youtube_clip must raise RuntimeError when ffmpeg exits non-zero."""
    import subprocess as _sp

    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    output_path = str(tmp_path / "normalized.mp4")

    err = _sp.CalledProcessError(1, "ffmpeg", stderr="codec error")
    with mock.patch("subprocess.run", side_effect=err):
        with pytest.raises(RuntimeError, match="ffmpeg normalize_youtube_clip failed"):
            normalize_youtube_clip(input_path, output_path, "13", "26")


def test_normalize_youtube_clip_ffmpeg_cmd_includes_ss_and_to(tmp_path):
    """normalize_youtube_clip must pass -ss and -to to ffmpeg."""
    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    output_path = str(tmp_path / "normalized.mp4")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        normalize_youtube_clip(input_path, output_path, "13", "26")

    cmd = captured["cmd"]
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "13"
    assert "-to" in cmd and cmd[cmd.index("-to") + 1] == "26"


def test_normalize_youtube_clip_ffmpeg_cmd_uses_libx264_yuv420p_aac(tmp_path):
    """normalize_youtube_clip must encode with libx264, yuv420p, aac."""
    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    output_path = str(tmp_path / "normalized.mp4")

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        normalize_youtube_clip(input_path, output_path, "13", "26")

    cmd = captured["cmd"]
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "libx264"
    assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "aac"


def test_normalize_youtube_clip_creates_output_dir(tmp_path):
    """normalize_youtube_clip must create the output directory if absent."""
    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    nested_dir = tmp_path / "assets" / "tmp"
    output_path = str(nested_dir / "youtube_normalized.mp4")

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        normalize_youtube_clip(input_path, output_path, "0", "13")

    assert os.path.isdir(str(nested_dir))


def test_normalize_youtube_clip_returns_output_path(tmp_path):
    """normalize_youtube_clip must return output_path for call-chain convenience."""
    input_path = str(tmp_path / "raw.mp4")
    (tmp_path / "raw.mp4").write_bytes(b"")
    output_path = str(tmp_path / "normalized.mp4")

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = normalize_youtube_clip(input_path, output_path, "13", "26")

    assert result == output_path


# ---------------------------------------------------------------------------
# build_source_clip – normalize_youtube_clip integration
# ---------------------------------------------------------------------------


def test_build_source_clip_uses_normalize_youtube_clip_when_end_time_given(tmp_path):
    """When transcode_proxy=True and end_time is set, build_source_clip must use
    normalize_youtube_clip (one-pass FFmpeg trim + normalise) instead of
    _transcode_to_proxy."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 13.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        with mock.patch.dict(
            "sys.modules",
            {"moviepy.editor": types.ModuleType("moviepy.editor")},
        ):
            sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
            build_source_clip(
                str(dummy), start_time=13, end_time=26,
                transcode_proxy=True, tmp_dir=str(tmp_path),
            )

    # Must have called ffmpeg with -ss and -to (normalize_youtube_clip behaviour)
    cmd = captured.get("cmd", [])
    assert "ffmpeg" in cmd
    assert "-ss" in cmd
    assert "-to" in cmd
    # Must NOT include -profile:v baseline (that is _transcode_to_proxy behaviour)
    assert "-profile:v" not in cmd


def test_build_source_clip_uses_transcode_proxy_when_no_end_time(tmp_path):
    """When transcode_proxy=True but end_time is None, _transcode_to_proxy is used."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 30.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.subclip.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        with mock.patch("tempfile.NamedTemporaryFile") as mock_ntf:
            mock_ntf.return_value.__enter__.return_value.name = str(tmp_path / "proxy.mp4")
            mock_ntf.return_value.__exit__ = mock.MagicMock(return_value=False)
            with mock.patch.dict(
                "sys.modules",
                {"moviepy.editor": types.ModuleType("moviepy.editor")},
            ):
                sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
                sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
                sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
                build_source_clip(str(dummy), transcode_proxy=True)

    # Must use _transcode_to_proxy path: includes -profile:v baseline
    cmd = captured.get("cmd", [])
    assert "ffmpeg" in cmd
    assert "-profile:v" in cmd
    assert cmd[cmd.index("-profile:v") + 1] == "baseline"


def test_build_source_clip_sets_explicit_duration_when_pretrimmed(tmp_path):
    """When normalize_youtube_clip pre-trims the clip, set_duration must be called."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 13.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        with mock.patch.dict(
            "sys.modules",
            {"moviepy.editor": types.ModuleType("moviepy.editor")},
        ):
            sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].CompositeVideoClip = mock.MagicMock()  # type: ignore[attr-defined]
            build_source_clip(
                str(dummy), start_time=13, end_time=26,
                transcode_proxy=True, tmp_dir=str(tmp_path),
            )

    # set_duration must be called when clip is pre-trimmed (not subclip)
    mock_clip.set_duration.assert_called_once()


def test_build_source_clip_assigns_audio_to_composite(tmp_path):
    """build_source_clip must assign clip.audio to the composite for guaranteed sync."""
    dummy = tmp_path / "video.mp4"
    dummy.write_bytes(b"")

    mock_clip = mock.MagicMock()
    mock_clip.duration = 13.0
    mock_clip.w = 1280
    mock_clip.h = 720
    mock_clip.without_mask.return_value = mock_clip
    mock_clip.set_opacity.return_value = mock_clip
    mock_clip.set_fps.return_value = mock_clip
    mock_clip.set_duration.return_value = mock_clip
    mock_clip.set_position.return_value = mock_clip

    mock_audio = mock.MagicMock()
    mock_clip.audio = mock_audio

    mock_vfc_cls = mock.MagicMock(return_value=mock_clip)
    mock_composite_inst = mock.MagicMock()
    mock_composite_cls = mock.MagicMock(return_value=mock_composite_inst)

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = ""
        return r

    with mock.patch("subprocess.run", side_effect=fake_run):
        with mock.patch.dict(
            "sys.modules",
            {"moviepy.editor": types.ModuleType("moviepy.editor")},
        ):
            sys.modules["moviepy.editor"].VideoFileClip = mock_vfc_cls  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].ColorClip = mock.MagicMock()  # type: ignore[attr-defined]
            sys.modules["moviepy.editor"].CompositeVideoClip = mock_composite_cls  # type: ignore[attr-defined]
            build_source_clip(
                str(dummy), start_time=13, end_time=26,
                transcode_proxy=True, tmp_dir=str(tmp_path),
            )

    # composite.audio must be set to clip.audio
    assert mock_composite_inst.audio is mock_audio


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


def test_render_video_anchors_first_clip_at_zero(tmp_path):
    """render_video must call .set_start(0) on the first clip before concatenation."""
    output_path = str(tmp_path / "out.mp4")

    mock_clip_1 = mock.MagicMock()
    mock_clip_anchored = mock.MagicMock()
    mock_clip_1.set_start.return_value = mock_clip_anchored

    mock_clip_2 = mock.MagicMock()

    mock_final = mock.MagicMock()
    mock_concat = mock.MagicMock(return_value=mock_final)

    with mock.patch.dict(
        "sys.modules",
        {"moviepy.editor": types.ModuleType("moviepy.editor")},
    ):
        sys.modules["moviepy.editor"].concatenate_videoclips = mock_concat  # type: ignore[attr-defined]
        render_video([mock_clip_1, mock_clip_2], output_path=output_path)

    mock_clip_1.set_start.assert_called_once_with(0)
    # The anchored clip (result of set_start) must appear as the first element
    # passed to concatenate_videoclips.
    args, _ = mock_concat.call_args
    passed_clips = args[0]
    assert passed_clips[0] is mock_clip_anchored
    assert passed_clips[1] is mock_clip_2


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
