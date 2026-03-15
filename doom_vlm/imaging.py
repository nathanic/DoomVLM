"""Image processing: grid overlay, encoding, debug screenshots."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Font loader (cached)
_font_cache: dict[int, ImageFont.ImageFont] = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ]
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            _font_cache[size] = font
            return font
        except (OSError, IOError):
            continue
    font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def screen_to_pil(screen_buffer: np.ndarray) -> Image.Image:
    """Convert a VizDoom screen buffer (CHW or HWC) to a PIL Image."""
    if screen_buffer.ndim == 3 and screen_buffer.shape[0] in (1, 3, 4):
        img_array = np.transpose(screen_buffer, (1, 2, 0))
    else:
        img_array = screen_buffer
    if img_array.ndim == 2:
        return Image.fromarray(img_array, mode="L")
    if img_array.shape[2] == 1:
        return Image.fromarray(img_array[:, :, 0], mode="L")
    return Image.fromarray(img_array)


def draw_grid_overlay(img: Image.Image, grid_cols: int) -> Image.Image:
    """Draw a numbered grid overlay on the image."""
    overlay = img.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    col_width = w // grid_cols
    font = _load_font(24)

    for i in range(grid_cols):
        x = i * col_width
        if i > 0:
            draw.line([(x, 0), (x, h)], fill=(255, 255, 0), width=2)
        label = str(i + 1)
        cx = x + col_width // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((cx - tw // 2 + dx, 5 + dy), label, fill=(0, 0, 0), font=font)
        draw.text((cx - tw // 2, 5), label, fill=(255, 255, 0), font=font)

    return overlay


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        test = current + " " + word
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def encode_frame(img: Image.Image, max_dim: int = 320) -> str:
    """Resize + JPEG encode + base64."""
    resized = img.copy()
    resized.thumbnail((max_dim, max_dim))
    if resized.mode not in ("RGB", "L"):
        resized = resized.convert("RGB")
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_debug_screenshot(
    img_with_grid: Image.Image,
    player_name: str,
    episode: int,
    step: int,
    parsed: dict[str, str],
    raw_response: str,
    action_desc: str,
    reward: float,
    health: float,
    ammo: float,
    latency: float,
    screenshot_dir: Path,
) -> None:
    """Save a debug image with game screenshot and VLM response."""
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    game_img = img_with_grid.copy().convert("RGB")
    gw, gh = game_img.size
    font = _load_font(14)

    padding = 8
    line_height = 18
    max_text_width = gw - padding * 2

    header = f"{player_name} | Episode {episode}  Step {step}  |  HP={health:.0f}  AMMO={ammo:.0f}  |  Latency={latency:.1f}s"
    raw_lines = []
    for rl in raw_response.split("\n"):
        raw_lines.extend(_wrap_text(rl, font, max_text_width) if rl.strip() else [""])
    action_line = f"Game: {action_desc}"

    all_lines = (
        _wrap_text(header, font, max_text_width)
        + [""]
        + raw_lines
        + [""]
        + _wrap_text(action_line, font, max_text_width)
    )

    text_height = len(all_lines) * line_height + padding * 2
    combined = Image.new("RGB", (gw, gh + text_height), (30, 30, 30))
    combined.paste(game_img, (0, 0))

    draw = ImageDraw.Draw(combined)
    y = gh + padding
    for line in all_lines:
        draw.text((padding, y), line, fill=(220, 220, 220), font=font)
        y += line_height

    path = screenshot_dir / f"{player_name}_ep{episode:02d}_step{step:03d}.png"
    combined.save(path)
