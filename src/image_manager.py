import logging
import os
import requests
from PIL import Image
from src.config import IMAGES_DIR, PEXELS_API_KEY, VIDEO_RES

logger = logging.getLogger(__name__)

_PEXELS_HEADERS = {"Authorization": PEXELS_API_KEY} if PEXELS_API_KEY else {}
_TARGET_W, _TARGET_H = VIDEO_RES  # 1080 x 1920

_CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}


def _resolve_url_extension(url: str) -> str:
    """Return the file extension for *url*, falling back to a HEAD request.

    First tries to derive the extension from the URL path. If the path has
    no recognisable extension, a lightweight HEAD request is made to read the
    ``Content-Type`` header.  Defaults to ``.jpg`` when nothing else works.
    """
    url_path = url.split("?")[0]
    ext = os.path.splitext(url_path)[1].lower()
    if ext in _CONTENT_TYPE_TO_EXT.values():
        return ext
    # No extension in URL path – ask the server
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        return _CONTENT_TYPE_TO_EXT.get(content_type, ".jpg")
    except requests.RequestException:
        return ".jpg"


def _search_pexels_video(keyword: str) -> str | None:
    """Return the URL of the first portrait/square short-video clip or None."""
    try:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=_PEXELS_HEADERS,
            params={"query": keyword, "orientation": "portrait", "per_page": 5},
            timeout=10,
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
        for video in videos:
            for file in video.get("video_files", []):
                if file.get("quality") in ("hd", "sd") and file.get("link"):
                    return file["link"]
    except requests.RequestException:
        pass
    return None


def _search_pexels_image(keyword: str) -> str | None:
    """Return the URL of the first high-resolution portrait photo or None."""
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers=_PEXELS_HEADERS,
            params={"query": keyword, "orientation": "portrait", "per_page": 5},
            timeout=10,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        for photo in photos:
            url = photo.get("src", {}).get("large2x") or photo.get("src", {}).get("original")
            if url:
                return url
    except requests.RequestException:
        pass
    return None


def _download_file(url: str, dest_path: str) -> bool:
    """Download *url* to *dest_path*. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    fh.write(chunk)
        return True
    except requests.RequestException:
        return False


def process_visual_asset(asset_path: str, output_path: str) -> str:
    """Resize/crop *asset_path* to 1080×1920 and save it to *output_path*.

    - Videos are resized via MoviePy (keeping audio if present).
    - Images are center-cropped with Pillow so they fill the vertical frame
      without distortion.

    Returns the *output_path*.
    """
    ext = os.path.splitext(asset_path)[1].lower()

    if ext == ".mp4":
        # Import here to avoid hard dependency when only images are used
        from moviepy.editor import VideoFileClip  # type: ignore

        clip = VideoFileClip(asset_path)
        try:
            clip_w, clip_h = clip.size
            scale = max(_TARGET_W / clip_w, _TARGET_H / clip_h)
            new_w = int(clip_w * scale)
            new_h = int(clip_h * scale)
            clip_resized = clip.resize((new_w, new_h))
            try:
                x_center = new_w / 2
                y_center = new_h / 2
                clip_cropped = clip_resized.crop(
                    x_center=x_center,
                    y_center=y_center,
                    width=_TARGET_W,
                    height=_TARGET_H,
                )
                try:
                    clip_cropped.write_videofile(
                        output_path, codec="libx264", audio_codec="aac", logger=None
                    )
                finally:
                    clip_cropped.close()
            finally:
                clip_resized.close()
        finally:
            clip.close()
    else:
        # Image processing with Pillow
        with Image.open(asset_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size
            scale = max(_TARGET_W / img_w, _TARGET_H / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - _TARGET_W) // 2
            top = (new_h - _TARGET_H) // 2
            img = img.crop((left, top, left + _TARGET_W, top + _TARGET_H))
            img.save(output_path, quality=95)

    return output_path


def get_visual_assets(script_data: list[dict]) -> list[str]:
    """Download and process one visual asset per item in *script_data*.

    Each item may optionally contain a ``"source"`` key:

    - **URL** (starts with ``http://`` or ``https://``): the file is downloaded
      from that address and resized/cropped to 1080×1920.
    - **Local path** (e.g. ``assets/mifoto.jpg``): that file is used directly
      and resized/cropped to 1080×1920.
    - **Absent / empty**: the existing Pexels keyword-search logic is used
      (tries a portrait video first, then a photo).

    Files are saved under *IMAGES_DIR* as ``01_visual.mp4`` / ``01_visual.jpg``
    (numbered in script order).

    Returns a list of absolute paths to the processed files.
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)
    results: list[str] = []

    for idx, item in enumerate(script_data, start=1):
        prefix = f"{idx:02d}_visual"
        source = item.get("source", "").strip()

        # --- Explicit source (URL or local path) --------------------------------
        if source:
            if source.startswith("http://") or source.startswith("https://"):
                # Derive extension from URL path; fall back to Content-Type header
                ext = _resolve_url_extension(source)
                raw_path = os.path.join(IMAGES_DIR, f"{prefix}_raw{ext}")
                out_path = os.path.join(IMAGES_DIR, f"{prefix}{ext}")
                if _download_file(source, raw_path):
                    try:
                        process_visual_asset(raw_path, out_path)
                        os.remove(raw_path)
                        results.append(out_path)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("processing failed for source URL '%s': %s", source, exc)
                        if os.path.exists(raw_path):
                            os.remove(raw_path)
                else:
                    logger.warning("failed to download source URL: %s", source)
            else:
                # Local file path
                if os.path.exists(source):
                    ext = os.path.splitext(source)[1].lower()
                    out_path = os.path.join(IMAGES_DIR, f"{prefix}{ext}")
                    try:
                        process_visual_asset(source, out_path)
                        results.append(out_path)
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("processing failed for local source '%s': %s", source, exc)
                else:
                    logger.info("local source file not found: '%s'. Falling back to keyword search.", source)
            # If source was provided but failed, fall through to keyword search

        # --- Keyword-based Pexels search (used when source is absent) -----------
        keyword = item.get("keyword", "").strip()
        if not keyword:
            results.append("")
            continue

        # --- Try video first ---------------------------------------------------
        video_url = _search_pexels_video(keyword)
        if video_url:
            raw_path = os.path.join(IMAGES_DIR, f"{prefix}_raw.mp4")
            out_path = os.path.join(IMAGES_DIR, f"{prefix}.mp4")
            if _download_file(video_url, raw_path):
                try:
                    process_visual_asset(raw_path, out_path)
                    os.remove(raw_path)
                    results.append(out_path)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("video processing failed for '%s': %s", keyword, exc)
                    if os.path.exists(raw_path):
                        os.remove(raw_path)

        # --- Fallback: photo ---------------------------------------------------
        image_url = _search_pexels_image(keyword)
        if image_url:
            raw_path = os.path.join(IMAGES_DIR, f"{prefix}_raw.jpg")
            out_path = os.path.join(IMAGES_DIR, f"{prefix}.jpg")
            if _download_file(image_url, raw_path):
                try:
                    process_visual_asset(raw_path, out_path)
                    os.remove(raw_path)
                    results.append(out_path)
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning("image processing failed for '%s': %s", keyword, exc)
                    if os.path.exists(raw_path):
                        os.remove(raw_path)

        # --- Nothing found or download failed ----------------------------------
        results.append("")

    return results


# ---------------------------------------------------------------------------
# Backward-compatible aliases for code that still uses the old names
# ---------------------------------------------------------------------------
def download_images(keywords: list[str]) -> list[str]:
    """Thin wrapper kept for backward compatibility."""
    script_data = [{"keyword": kw} for kw in keywords]
    return get_visual_assets(script_data)


def process_image_for_video(image_path: str) -> str:
    """Thin wrapper kept for backward compatibility."""
    out_path = os.path.splitext(image_path)[0] + "_processed.jpg"
    return process_visual_asset(image_path, out_path)
