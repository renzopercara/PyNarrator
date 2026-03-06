import json
import asyncio
import logging
import os
import random
import textwrap
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageEnhance

# Parche para compatibilidad de Pillow con MoviePy
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from src.narrator import ArgentineNarrator
from src.image_manager import get_visual_assets
from src.subtitle_generator import generate_subtitles
from src.sentiment_analyzer import analyze_tone
from src.copy_generator import generate_social_copy
from src.vocabulary_annotator import annotate_story
from src.config import (
    OUTPUT_DIR, AUDIO_DIR,
    MUSIC_DIR, SFX_DIR, VIDEO_RES, LOGO_PATH,
    WATERMARK_PATH, WATERMARK_OPACITY, WATERMARK_WIDTH_PERCENT,
    VOICES, ESL_VOICE_RATE, ESL_VOICE_PITCH,
)
from src.context import (
    detect_context as _detect_context,
    output_path_for_context as _output_path_for_context,
)

from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, ColorClip,
    CompositeVideoClip, CompositeAudioClip, concatenate_videoclips, vfx,
)
import moviepy.audio.fx.all as afx

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = VIDEO_RES

# --- PLATFORM RESOLUTIONS ---
# Maps a "target platform" name to its (width, height) render resolution.
_PLATFORM_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "Instagram": (1080, 1920),
    "LinkedIn": (1080, 1080),
}

# Current target resolution – updated at the start of main() based on the
# chosen platform.  Helper functions that need the render size read from
# this variable so the whole pipeline stays consistent.
_TARGET_RESOLUTION: tuple[int, int] = VIDEO_RES

# Smartbuild brand colour used as a fallback background when visual assets
# are unavailable (deep-navy blue).
_BRAND_COLOR: tuple[int, int, int] = (15, 20, 50)

# --- CONSTANTES DE ESTILO OPTIMIZADAS ---
_ZOOM_RATE_NORMAL = 0.02
_CONTRAST_BOOST = 1.15   
_SATURATION_BOOST = 1.25 
_GAMMA_CORRECTION = 0.80 

def _enhance_frame(frame):
    """Mejora visual: Brillo, Contraste y Saturación."""
    img = PIL.Image.fromarray(frame.astype(np.uint8))
    img = PIL.ImageEnhance.Contrast(img).enhance(_CONTRAST_BOOST)
    img = PIL.ImageEnhance.Color(img).enhance(_SATURATION_BOOST)
    img = PIL.ImageEnhance.Brightness(img).enhance(1.1) 
    
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.power(arr, _GAMMA_CORRECTION) 
    return np.clip(arr * 255, 0, 255).astype(np.uint8)

def _load_hook_font(size: int):
    candidates = ["C:/Windows/Fonts/montserrat-extrabold.ttf", "C:/Windows/Fonts/arialbd.ttf"]
    for path in candidates:
        if os.path.exists(path): return PIL.ImageFont.truetype(path, size)
    return PIL.ImageFont.load_default()

def _make_hook_clip(topic: str, duration: float = 2.0) -> ImageClip:
    font = _load_hook_font(75)
    text = topic.upper()
    img = PIL.Image.new("RGBA", (VIDEO_W, 300), (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((VIDEO_W - tw) // 2, (300 - th) // 2), text, font=font, 
              fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    return ImageClip(np.array(img)).set_duration(duration).set_position(("center", 200)).crossfadeout(0.5)

def _make_watermark_clip(video_duration: float) -> ImageClip:
    if not os.path.exists(WATERMARK_PATH): return None
    watermark_w = int(VIDEO_W * WATERMARK_WIDTH_PERCENT)
    clip = ImageClip(WATERMARK_PATH).resize(width=watermark_w)
    return (clip.set_opacity(WATERMARK_OPACITY).set_duration(video_duration)
            .set_position(("center", VIDEO_H - clip.size[1] - 50)))

def _make_clip_for_scene(asset_path, duration, zoom_in=True, start_time=0, end_time=None):
    if not asset_path or not os.path.exists(asset_path):
        return ColorClip(size=VIDEO_RES, color=(240, 240, 240), duration=duration)

    ext = os.path.splitext(asset_path)[1].lower()
    
    if ext == ".mp4":
        clip = VideoFileClip(asset_path)
        # Normalise: strip any alpha mask, enforce full opacity and consistent FPS
        # to prevent white/blank frames on YouTube and social-media players.
        clip = clip.without_mask().set_opacity(1).resize(height=720).set_fps(24)
        # Trim to the requested time window (supports start_time / end_time in seconds)
        clip_start = start_time
        clip_end = min(end_time, clip.duration) if end_time is not None else clip.duration
        clip = clip.subclip(clip_start, clip_end)
        duration = clip.duration
        # Crop-Fill: escalar para cubrir el eje menor y recortar el eje mayor.
        # Esto garantiza que no queden bordes de 1 píxel ni transparencias.
        vid_aspect = clip.w / clip.h
        target_aspect = VIDEO_W / VIDEO_H
        if vid_aspect > target_aspect:
            # Video más ancho que 9:16: ajustar por altura y recortar ancho sobrante
            clip = clip.resize(height=VIDEO_H)
        else:
            # Video más alto o igual: ajustar por ancho y recortar alto sobrante
            clip = clip.resize(width=VIDEO_W)
        clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=VIDEO_W, height=VIDEO_H)
        base = clip.fl_image(_enhance_frame)
        # Fondo negro sólido para eliminar cualquier transparencia residual
        bg = ColorClip(VIDEO_RES, color=(0, 0, 0)).set_duration(duration)
        return CompositeVideoClip([bg, base.set_position("center")], size=VIDEO_RES).set_duration(duration)

    # --- Lógica para Imágenes ---
    img_clip = ImageClip(asset_path).set_duration(duration)
    
    # Redimensionamos la imagen para que CUBRA todo el canvas (evita bordes)
    w, h = img_clip.size
    aspect_ratio_target = VIDEO_W / VIDEO_H
    if w / h > aspect_ratio_target:
        img_clip = img_clip.resize(height=VIDEO_H)
    else:
        img_clip = img_clip.resize(width=VIDEO_W)

    # Aplicamos el Zoom
    def _resize_fn(t):
        # Zoom sutil: de 1.0 a 1.1 o de 1.1 a 1.0
        base = 1.0 + 0.1 * t/duration if zoom_in else 1.1 - 0.1 * t/duration
        return base

    img_clip = img_clip.resize(_resize_fn).set_position("center")
    
    # CLAVE: Forzamos el CompositeVideoClip a VIDEO_RES y le ponemos un fondo negro sólido
    # Esto elimina cualquier transparencia que Instagram pueda detectar como "error de aspecto"
    return (CompositeVideoClip([ColorClip(VIDEO_RES, color=(0,0,0)).set_duration(duration), img_clip], size=VIDEO_RES)
            .set_duration(duration)
            .fl_image(_enhance_frame))

# ---------------------------------------------------------------------------
# Micro-Learning helpers
# ---------------------------------------------------------------------------

# Card colour palette
_EDU_BG_COLOR = (15, 20, 50)        # Deep navy background
_EDU_TERM_COLOR = (255, 215, 0)     # Gold for the keyword term
_EDU_LABEL_COLOR = (120, 180, 255)  # Light blue for section labels
_EDU_BODY_COLOR = (240, 240, 240)   # Near-white for body text

_EDU_CARD_PADDING = 90              # px from canvas edge
_EDU_LINE_SPACING = 14              # extra px between wrapped lines


def _wrap_text_to_lines(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap *text* into lines that fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def _make_educational_card(term: str, definition: str, example: str, duration: float) -> ImageClip:
    """Create a vocabulary educational card as a MoviePy :class:`ImageClip`.

    The card shows:
      - The keyword term (gold, large)
      - A "DEFINITION" section with the plain-English explanation
      - An "EXAMPLE" section with a real IT usage example

    Args:
        term:       The technical keyword.
        definition: Plain-English definition of the term.
        example:    IT usage example sentence.
        duration:   Duration of the resulting clip in seconds.

    Returns:
        A :class:`~moviepy.editor.ImageClip` of size :data:`VIDEO_RES`.
    """
    font_term = _load_hook_font(88)
    font_label = _load_hook_font(48)
    font_body = _load_hook_font(52)

    img = PIL.Image.new("RGB", VIDEO_RES, _EDU_BG_COLOR)
    draw = PIL.ImageDraw.Draw(img)

    max_text_w = VIDEO_W - 2 * _EDU_CARD_PADDING
    y = int(VIDEO_H * 0.12)

    # ── Term ────────────────────────────────────────────────────────────────
    term_upper = term.upper()
    bbox = draw.textbbox((0, 0), term_upper, font=font_term)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(
        ((VIDEO_W - tw) // 2, y),
        term_upper, font=font_term, fill=_EDU_TERM_COLOR,
        stroke_width=4, stroke_fill=(0, 0, 0),
    )
    y += th + 60

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line(
        [(_EDU_CARD_PADDING, y), (VIDEO_W - _EDU_CARD_PADDING, y)],
        fill=_EDU_TERM_COLOR, width=4,
    )
    y += 50

    # ── Definition label ────────────────────────────────────────────────────
    draw.text((_EDU_CARD_PADDING, y), "DEFINITION", font=font_label, fill=_EDU_LABEL_COLOR)
    bbox = draw.textbbox((0, 0), "DEFINITION", font=font_label)
    y += (bbox[3] - bbox[1]) + 28

    # ── Definition body ─────────────────────────────────────────────────────
    for line in _wrap_text_to_lines(draw, definition, font_body, max_text_w):
        draw.text((_EDU_CARD_PADDING, y), line, font=font_body, fill=_EDU_BODY_COLOR)
        bbox = draw.textbbox((0, 0), line, font=font_body)
        y += (bbox[3] - bbox[1]) + _EDU_LINE_SPACING
    y += 55

    # ── Divider ─────────────────────────────────────────────────────────────
    draw.line(
        [(_EDU_CARD_PADDING, y), (VIDEO_W - _EDU_CARD_PADDING, y)],
        fill=_EDU_LABEL_COLOR, width=2,
    )
    y += 50

    # ── Example label ───────────────────────────────────────────────────────
    draw.text((_EDU_CARD_PADDING, y), "EXAMPLE", font=font_label, fill=_EDU_LABEL_COLOR)
    bbox = draw.textbbox((0, 0), "EXAMPLE", font=font_label)
    y += (bbox[3] - bbox[1]) + 28

    # ── Example body ─────────────────────────────────────────────────────────
    for line in _wrap_text_to_lines(draw, example, font_body, max_text_w):
        draw.text((_EDU_CARD_PADDING, y), line, font=font_body, fill=_EDU_BODY_COLOR)
        bbox = draw.textbbox((0, 0), line, font=font_body)
        y += (bbox[3] - bbox[1]) + _EDU_LINE_SPACING

    return ImageClip(np.array(img)).set_duration(duration)


def _is_micro_learning_script(script) -> bool:
    """Return True if *script* follows the micro-learning JSON format.

    The micro-learning format is a :class:`dict` with both ``"metadata"`` and
    ``"scenes"`` keys, as opposed to the legacy flat list format.
    """
    return isinstance(script, dict) and "metadata" in script and "scenes" in script


async def _generate_edu_audio(text: str, voice_key: str, idx: int) -> str:
    """Generate a TTS audio file for an educational scene narration.

    Uses the ESL rate (``-5%``) and pitch (``+0Hz``) defined in
    :mod:`src.config` for maximum educational clarity.

    Args:
        text:      The text to synthesise (definition + example).
        voice_key: ``"H"`` (male ``en-US-AndrewNeural``) or ``"M"`` (female
                   ``en-US-AvaNeural``).
        idx:       Scene index used to derive a unique filename.

    Returns:
        Absolute path to the saved ``.mp3`` file.
    """
    import edge_tts

    os.makedirs(AUDIO_DIR, exist_ok=True)
    voice_name = VOICES.get(voice_key, VOICES["H"])
    path = os.path.join(AUDIO_DIR, f"edu_{idx:04d}.mp3")
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice_name,
        rate=ESL_VOICE_RATE,
        pitch=ESL_VOICE_PITCH,
    )
    await communicate.save(path)
    return path


async def main_micro_learning(script: dict) -> None:
    """Process and render a micro-learning video from a structured script dict.

    Handles the four scene types defined by the micro-learning workflow:

    * ``"original"``    – Loads ``video_source`` as a clip (no new TTS).
    * ``"highlighted"`` – Same as ``"original"``; the ``keywords`` field is
                          stored in the clip's metadata for potential downstream
                          subtitle highlighting.
    * ``"educational"`` – Renders a vocabulary card image with TTS narration
                          voiced by the *opposite* gender from the original
                          narrator (SLA best practice).
    * ``"review"``      – Repeats the ``video_source`` clip without distractions.

    Args:
        script: Micro-learning script dict (as produced by
                :func:`src.micro_learning_generator.generate_micro_learning_script`).
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata = script.get("metadata", {})
    video_source = script.get("video_source", "")
    scenes = script.get("scenes", [])

    tone = metadata.get("tone", "INFORMATIVE")
    narrator_voice = metadata.get("narrator_voice", "H")

    # Determine duration of the source video clip
    src_duration = 5.0
    src_video_clip = None
    if video_source and os.path.exists(video_source):
        src_video_clip = VideoFileClip(video_source)
        src_duration = src_video_clip.duration
        src_video_clip.close()
        src_video_clip = None

    output_path = _output_path_for_context("esl")
    logger.info("🎬 Micro-Learning script detected → %s", output_path)

    scene_clips = []
    edu_audio_paths = []

    try:
        for i, scene in enumerate(scenes):
            scene_type = scene.get("type", "")

            if scene_type in ("original", "highlighted", "review"):
                # Load the video source using start_time/end_time when present
                scene_start = scene.get("start_time") or 0
                scene_end = scene.get("end_time") if isinstance(scene.get("end_time"), (int, float)) else None
                # Fallback duration is used only for non-MP4 assets (e.g. ColorClip)
                fallback_duration = (
                    (scene_end - scene_start) if scene_end is not None else src_duration
                )
                clip = _make_clip_for_scene(
                    video_source, fallback_duration, zoom_in=False,
                    start_time=scene_start, end_time=scene_end,
                )
                if scene_clips:
                    clip = clip.crossfadein(0.4)
                scene_clips.append(clip)
                logger.info(
                    "▶ Scene %d (%s): loaded video_source [%ss–%ss] (%.1fs)",
                    i, scene_type, scene_start,
                    scene_end if scene_end is not None else "end", clip.duration,
                )

            elif scene_type == "educational":
                term = scene.get("term", "")
                definition = scene.get("definition", "")
                # Support both "it_example" (Language Coach) and "example" (basic)
                example = scene.get("it_example") or scene.get("example", "")
                edu_voice = scene.get("narrator_voice", "M" if narrator_voice == "H" else "H")

                # Compose the narration text: "<term>. <definition> For example: <example>"
                narration = f"{term}. {definition} For example: {example}"
                audio_path = await _generate_edu_audio(narration, edu_voice, i)
                edu_audio_paths.append(audio_path)

                audio_clip = AudioFileClip(audio_path)
                edu_duration = audio_clip.duration
                audio_clip.close()

                card = _make_educational_card(term, definition, example, edu_duration)
                card = card.set_audio(AudioFileClip(audio_path))
                if scene_clips:
                    card = card.crossfadein(0.4)
                scene_clips.append(card)
                logger.info(
                    "📚 Scene %d (educational): '%s' voiced by %s (%.1fs)",
                    i, term, edu_voice, edu_duration,
                )

            else:
                logger.warning("⚠️ Unknown scene type '%s' at index %d – skipping.", scene_type, i)

        if not scene_clips:
            logger.error("❌ No scene clips were generated. Aborting.")
            return

        final_video = concatenate_videoclips(scene_clips, method="compose")

        # Optional watermark
        watermark = _make_watermark_clip(final_video.duration)
        layers = [final_video]
        if watermark:
            layers.append(watermark)
        final_video = CompositeVideoClip(layers, size=VIDEO_RES)

        # Background music (optional)
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = (
            [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))]
            if os.path.isdir(tone_music_dir) else []
        )
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        if m_files:
            bg_music = (
                AudioFileClip(random.choice(m_files))
                .fx(afx.audio_loop, duration=final_video.duration)
                .volumex(0.08)
                .audio_fadeout(2.0)
            )
            existing_audio = final_video.audio
            if existing_audio is not None:
                final_video = final_video.set_audio(
                    CompositeAudioClip([existing_audio, bg_music])
                )
            else:
                final_video = final_video.set_audio(bg_music)

        logger.info("💾 Exporting micro-learning to: %s", output_path)
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(OUTPUT_DIR, "temp-audio.m4a"),
            remove_temp=True,
            threads=4,
            logger=None,
            verbose=False,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-vf", "setsar=1:1",
                "-movflags", "+faststart",
                "-profile:v", "high",
                "-level", "4.0",
            ],
        )
        logger.info("✅ Micro-learning complete!")

    finally:
        try:
            final_video.close()
        except Exception:
            pass
        for clip in scene_clips:
            try:
                clip.close()
            except Exception:
                pass
        for path in edu_audio_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


async def main(target_platform: str = "Instagram") -> None:
    """Render the video described by ``script.json``.

    Args:
        target_platform: Output resolution preset.  Supported values are
            ``"Instagram"`` (1080×1920 vertical) and ``"LinkedIn"``
            (1080×1080 square).  Defaults to ``"Instagram"``.
    """
    global _TARGET_RESOLUTION
    _TARGET_RESOLUTION = _PLATFORM_RESOLUTIONS.get(target_platform, VIDEO_RES)
    logger.info("🎯 Target platform: %s → resolution %s", target_platform, _TARGET_RESOLUTION)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # All AudioFileClip / VideoFileClip objects created during this run are
    # collected here so they can be closed in the finally block even if an
    # error occurs mid-render.
    media_clips: list = []
    scene_clips: list = []
    base_video = None
    final_video = None

    try:
        with open("script.json", "r", encoding="utf-8") as f:
            script = json.load(f)

        # Dispatch: micro-learning format (dict with metadata + scenes) vs. legacy list
        if _is_micro_learning_script(script):
            await main_micro_learning(script)
            # Generate social copy from micro-learning scenes.
            scenes_as_list = [
                {"keyword": s.get("term", ""), "texto": s.get("definition", "")}
                for s in script.get("scenes", [])
                if s.get("term") or s.get("definition")
            ]
            _save_social_post(generate_social_copy(scenes_as_list))
            return

        context = _detect_context(script)
        output_path = _output_path_for_context(context)
        logger.info("🎬 Contexto detectado: %s → %s", context, output_path)

        tone = analyze_tone(script)
        narrator = ArgentineNarrator()
        voice_data = await narrator.generate_voice_overs(script, tone=tone)

        # --- Asset error handling: fall back to brand-colour ColorClips ----
        try:
            visual_assets = get_visual_assets(script, tone=tone)
            if not visual_assets:
                raise ValueError(
                    "get_visual_assets returned an empty list; falling back to brand color clips"
                )
        except Exception as exc:
            logger.warning(
                "⚠️ get_visual_assets failed (%s) – using brand ColorClips as fallback.", exc
            )
            visual_assets = [None] * len(voice_data)

        # 1. Video Base
        zoom_in = True
        for data, asset in zip(voice_data, visual_assets):
            if asset is None:
                clip = ColorClip(
                    size=_TARGET_RESOLUTION,
                    color=_BRAND_COLOR,
                    duration=data["duracion"],
                )
            else:
                clip = _make_clip_for_scene(asset, data["duracion"], zoom_in)
            if scene_clips:
                clip = clip.crossfadein(0.4)
            scene_clips.append(clip)
            zoom_in = not zoom_in

        base_video = concatenate_videoclips(scene_clips, method="compose")
        audio_path = os.path.join(AUDIO_DIR, "final_voice.mp3")

        # 2. Subtítulos y Branding
        subtitle_clips, _, sentence_start_times = generate_subtitles(
            audio_path, script_data=script, return_segment_times=True, tone=tone
        )
        hook_clip = _make_hook_clip((script[0].get("keyword") or context).strip())
        watermark = _make_watermark_clip(base_video.duration)

        layers = [base_video]
        if watermark:
            layers.append(watermark)

        final_video = CompositeVideoClip(layers + subtitle_clips + [hook_clip], size=_TARGET_RESOLUTION)

        # 3. Mezcla de Audio (CORREGIDO)
        logger.info("🔊 Mezclando capas de audio...")
        voice_audio = AudioFileClip(audio_path).volumex(1.4)
        media_clips.append(voice_audio)
        audio_layers = [voice_audio]

        # Música de Fondo
        tone_music_dir = os.path.join(MUSIC_DIR, tone.lower())
        m_files = [os.path.join(tone_music_dir, f) for f in os.listdir(tone_music_dir) if f.endswith((".mp3", ".wav"))] if os.path.isdir(tone_music_dir) else []
        if not m_files:
            m_files = [os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".wav"))]

        if m_files:
            bg_music = (AudioFileClip(random.choice(m_files))
                        .fx(afx.audio_loop, duration=final_video.duration)
                        .volumex(0.08).audio_fadeout(2.0))
            media_clips.append(bg_music)
            audio_layers.append(bg_music)

        # Efectos (Ambiente y Pops)
        a_path = os.path.join(SFX_DIR, "ambience_construction.mp3")
        if os.path.exists(a_path):
            amb_clip = AudioFileClip(a_path).fx(afx.audio_loop, duration=final_video.duration).volumex(0.03)
            media_clips.append(amb_clip)
            audio_layers.append(amb_clip)

        p_path = os.path.join(SFX_DIR, "pop.mp3")
        if os.path.exists(p_path):
            for t in sentence_start_times:
                pop_clip = AudioFileClip(p_path).volumex(0.15).set_start(t)
                media_clips.append(pop_clip)
                audio_layers.append(pop_clip)

        final_video = final_video.set_audio(CompositeAudioClip(audio_layers))

        # 4. Render Final
        logger.info("💾 Exportando a: %s", output_path)
        final_video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(OUTPUT_DIR, "temp-audio.m4a"),
            remove_temp=True,
            threads=4,
            logger=None,
            verbose=False,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",      # Color compatible con todos los celulares
                "-vf", "setsar=1:1",        # Asegura que cada píxel sea un cuadrado perfecto
                "-movflags", "+faststart",  # Mueve la metadata al principio del archivo (CLAVE para redes sociales)
                "-profile:v", "high",       # Perfil de compresión que prefiere Instagram
                "-level", "4.0"             # Nivel de compatibilidad estándar
            ]
        )
        logger.info("✅ ¡Proceso completado!")

        # 5. Social copy
        _save_social_post(generate_social_copy(script))

    finally:
        # Explicit close of every AudioFileClip / VideoFileClip to prevent
        # file-handle leaks even when an error occurs mid-render.
        for obj in media_clips:
            try:
                obj.close()
            except Exception:
                pass
        if final_video is not None:
            try:
                final_video.close()
            except Exception:
                pass
        if base_video is not None:
            try:
                base_video.close()
            except Exception:
                pass
        for c in scene_clips:
            try:
                c.close()
            except Exception:
                pass


def _save_social_post(info_file_path: str) -> None:
    """Copy the generated social copy to ``output/social_post.txt``.

    ``generate_social_copy`` already writes to ``output/INFO_POSTEO.txt``.
    This helper additionally saves the same content as ``social_post.txt``
    so users can find it alongside the rendered video without hunting for
    the original file.

    Args:
        info_file_path: Path returned by :func:`~src.copy_generator.generate_social_copy`.
    """
    social_post_path = os.path.join(OUTPUT_DIR, "social_post.txt")
    try:
        with open(info_file_path, "r", encoding="utf-8") as src_fh:
            content = src_fh.read()
        with open(social_post_path, "w", encoding="utf-8") as dst_fh:
            dst_fh.write(content)
        logger.info("📋 Social post saved to: %s", social_post_path)
    except Exception as exc:
        logger.warning("⚠️ No se pudo guardar social_post.txt: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
