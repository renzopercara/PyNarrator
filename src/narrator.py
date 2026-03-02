import asyncio
import edge_tts
from moviepy.editor import concatenate_audioclips, AudioFileClip
import os
from src.config import VOICES, AUDIO_DIR

# Voice rate adjustments per tone (edge-tts rate string format)
_TONE_VOICE_RATE: dict[str, str] = {
    "ENERGICO": "+20%",     # faster, higher energy
    "INFORMATIVO": "+0%",   # neutral default
    "RELAJADO": "-15%",     # slower, calmer
}

# Voice pitch adjustments per tone
_TONE_VOICE_PITCH: dict[str, str] = {
    "ENERGICO": "+5Hz",
    "INFORMATIVO": "+0Hz",
    "RELAJADO": "-5Hz",
}


class ArgentineNarrator:
    def __init__(self):
        os.makedirs(AUDIO_DIR, exist_ok=True)

    async def generate_voice_overs(self, script_data, tone: str = "INFORMATIVO"):
        """Generate voice-overs for each item in script_data using edge-tts.

        For every element the method picks the Argentine voice (H = male,
        M = female) defined in VOICES, synthesises an MP3 fragment using the
        rate and pitch corresponding to *tone*, then concatenates all fragments
        into a single final_voice.mp3.

        Args:
            script_data: List of scene dicts from ``script.json``.
            tone:        Overall script tone (``"ENERGICO"``, ``"INFORMATIVO"``,
                         or ``"RELAJADO"``).  Drives voice rate and pitch.

        Returns:
            list[dict]: Each dict contains:
                - texto       (str)   original text
                - archivo_audio (str) path to the final concatenated audio file
                - duracion    (float) duration of the fragment in seconds
                - start_time  (float) offset in the concatenated audio (seconds)
        """
        if not script_data:
            return []

        temp_files = []
        try:
            # --- 1. Synthesise one MP3 per line ---------------------------------
            rate = _TONE_VOICE_RATE.get(tone, "+0%")
            pitch = _TONE_VOICE_PITCH.get(tone, "+0Hz")
            for idx, item in enumerate(script_data):
                text = item.get("texto", "").strip()
                if not text:
                    raise ValueError(
                        f"Item at index {idx} has empty 'texto'; cannot synthesise audio."
                    )
                voice_key = item.get("voz", "H")
                voice_name = VOICES.get(voice_key, VOICES["H"])
                temp_path = os.path.join(AUDIO_DIR, f"fragment_{idx:04d}.mp3")

                communicate = edge_tts.Communicate(text=text, voice=voice_name, rate=rate, pitch=pitch)
                await communicate.save(temp_path)
                temp_files.append(temp_path)

            # --- 2. Load clips and compute start times --------------------------
            clips = []
            for path in temp_files:
                clips.append(AudioFileClip(path))

            # --- 3. Concatenate into a single file ------------------------------
            final_path = os.path.join(AUDIO_DIR, "final_voice.mp3")
            final_clip = concatenate_audioclips(clips)
            final_clip.write_audiofile(final_path)

            results = []
            current_time = 0.0
            for idx, (item, clip) in enumerate(zip(script_data, clips)):
                results.append(
                    {
                        "texto": item.get("texto", ""),
                        "archivo_audio": final_path,
                        "duracion": clip.duration,
                        "start_time": current_time,
                    }
                )
                current_time += clip.duration

            # Close individual clips to release file handles
            for clip in clips:
                clip.close()
            final_clip.close()

            return results

        except (ValueError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(f"Unexpected error in generate_voice_overs: {exc}") from exc
        finally:
            # --- 4. Clean up temporary fragment files ---------------------------
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass