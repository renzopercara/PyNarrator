import os
import whisper
from moviepy.editor import TextClip
from src.config import AUDIO_DIR, VIDEO_RES

_AUDIO_PATH = os.path.join(AUDIO_DIR, "final_voice.mp3")
_VIDEO_WIDTH, _VIDEO_HEIGHT = VIDEO_RES  # width x height (1080 x 1920)

_FONT = "Arial-Bold"
_FONTSIZE = 70
_COLOR = "#FFFF00"
_STROKE_COLOR = "black"
_STROKE_WIDTH = 2
_Y_POSITION = int(_VIDEO_HEIGHT * 0.80)


def generate_subtitles(audio_path: str = _AUDIO_PATH) -> list:
    """Generate word-level subtitle clips from *audio_path* using Whisper.

    Loads the Whisper 'base' model, transcribes the audio with
    ``word_timestamps=True``, and converts each word into a styled
    :class:`moviepy.editor.TextClip` ready to be overlaid on the final
    video composition.

    Args:
        audio_path: Path to the MP3 audio file to transcribe.
                    Defaults to ``assets/audio/final_voice.mp3``.

    Returns:
        A list of :class:`moviepy.editor.TextClip` objects, one per word,
        with start time, duration, and position already set.
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)

    clips: list[TextClip] = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            start = word_info.get("start", 0.0)
            end = word_info.get("end", 0.0)
            duration = max(end - start, 0.05)  # minimum 50 ms so the clip is visible

            if not word:
                continue

            clip = (
                TextClip(
                    word,
                    fontsize=_FONTSIZE,
                    color=_COLOR,
                    font=_FONT,
                    stroke_color=_STROKE_COLOR,
                    stroke_width=_STROKE_WIDTH,
                )
                .set_start(start)
                .set_duration(duration)
                .set_position(("center", _Y_POSITION))
            )
            clips.append(clip)

    return clips
