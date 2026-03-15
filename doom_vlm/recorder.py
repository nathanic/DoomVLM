"""EpisodeRecorder — records per-tic frames, assembles GIF or MP4."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from doom_vlm.imaging import _load_font, draw_grid_overlay, screen_to_pil

logger = logging.getLogger("doom_dm")


class EpisodeRecorder:
    """Records per-tic frames with overlay, assembles GIF or MP4."""

    def __init__(
        self,
        episode: int,
        scenario: str,
        fmt: str,
        grid_cols: int,
        player_name: str = "agent",
        fps: int = 12,
        game_type: str = "dm",
        results_dir: Path | None = None,
    ):
        self.episode = episode
        self.scenario = scenario
        self.fmt = fmt
        self.grid_cols = grid_cols
        self.player_name = player_name
        self.fps = fps
        self.game_type = game_type
        self.results_dir = results_dir or Path("results")
        self._frames: list[Image.Image] = []
        self._tmp_dir: Path | None = None
        self._frame_count = 0
        self._step_ctx: dict[str, Any] = {}
        self._prev_health: float = 100.0

        if fmt == "mp4":
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="doom_rec_"))

    def set_step_context(
        self, step: int, health: float, ammo: float, frags: float,
        parsed: dict[str, str], action_desc: str, reward: float, latency: float,
    ) -> None:
        hp_dropped = health < self._prev_health
        self._prev_health = health
        self._step_ctx = {
            "step": step, "health": health, "ammo": ammo, "frags": frags,
            "reason": parsed.get("reason", ""), "action_desc": action_desc,
            "reward": reward, "latency": latency,
            "hp_dropped": hp_dropped, "dead": health <= 0,
        }

    def capture_tic(self, screen_buffer: np.ndarray, tic_idx: int, total_tics: int) -> None:
        img = screen_to_pil(screen_buffer)
        img = draw_grid_overlay(img, self.grid_cols)
        img = self._draw_overlay(img, tic_idx, total_tics)
        img = img.convert("RGB")

        if self.fmt == "gif":
            self._frames.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT))
        else:
            path = self._tmp_dir / f"frame_{self._frame_count:05d}.png"
            img.save(path)

        self._frame_count += 1

    def _draw_overlay(self, img: Image.Image, tic_idx: int, total_tics: int) -> Image.Image:
        img = img.copy().convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = img.size
        ctx = self._step_ctx
        font = _load_font(13)
        font_sm = _load_font(11)

        # Top bar
        draw.rectangle([(0, 30), (w, 56)], fill=(0, 0, 0, 160))
        step = ctx.get("step", 0)
        health = ctx.get("health", 0)
        ammo = ctx.get("ammo", 0)
        frags = ctx.get("frags", 0)
        hp_color = (255, 80, 80) if ctx.get("hp_dropped") else (80, 255, 80)
        top_text = f"Step {step} | Tic {tic_idx + 1}/{total_tics} | "
        draw.text((6, 34), top_text, fill=(255, 255, 255), font=font)
        hp_x = 6 + font.getbbox(top_text)[2]
        hp_str = f"HP={health:.0f}"
        draw.text((hp_x, 34), hp_str, fill=hp_color, font=font)
        ammo_x = hp_x + font.getbbox(hp_str)[2] + 8
        ammo_str = f"AMMO={ammo:.0f}"
        draw.text((ammo_x, 34), ammo_str, fill=(255, 255, 255), font=font)
        frags_x = ammo_x + font.getbbox(ammo_str)[2] + 8
        frag_label = "KILLS" if self.game_type == "solo" else "FRAGS"
        draw.text((frags_x, 34), f"{frag_label}={frags:.0f}", fill=(255, 200, 0), font=font)

        # Bottom bar
        draw.rectangle([(0, h - 24), (w, h)], fill=(0, 0, 0, 160))
        reason = ctx.get("reason", "")
        action_desc = ctx.get("action_desc", "")
        latency = ctx.get("latency", 0)
        bot_text = f"VLM: {reason} | {action_desc} | {latency:.1f}s"
        max_chars = w // 6
        if len(bot_text) > max_chars:
            bot_text = bot_text[:max_chars - 1] + "\u2026"
        draw.text((6, h - 22), bot_text, fill=(200, 200, 200), font=font_sm)

        img = Image.alpha_composite(img, overlay)
        result = img.convert("RGB")
        draw_rgb = ImageDraw.Draw(result)

        if tic_idx == 0:
            for i in range(3):
                draw_rgb.rectangle([(i, i), (w - 1 - i, h - 1 - i)], outline=(0, 255, 0))

        if ctx.get("dead"):
            red_overlay = Image.new("RGB", result.size, (180, 0, 0))
            result = Image.blend(result, red_overlay, alpha=0.4)

        return result

    def finalize(self) -> Path | None:
        if self._frame_count == 0:
            return None
        self.results_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.player_name}_episode_{self.episode}_{self.scenario}"
        if self.fmt == "gif":
            return self._finalize_gif(name)
        return self._finalize_mp4(name)

    def _finalize_gif(self, name: str) -> Path:
        out = self.results_dir / f"{name}.gif"
        duration_ms = 1000 // self.fps
        self._frames[0].save(
            out, save_all=True, append_images=self._frames[1:],
            duration=duration_ms, loop=0, optimize=False,
        )
        size_mb = out.stat().st_size / 1_000_000
        if size_mb > 50:
            tmp = Path(tempfile.mkdtemp(prefix="doom_gif2mp4_"))
            for i, f in enumerate(self._frames):
                f.convert("RGB").save(tmp / f"frame_{i:05d}.png")
            self._tmp_dir = tmp
            self._finalize_mp4(name)
            out.unlink(missing_ok=True)
            return self.results_dir / f"{name}.mp4"
        return out

    def _finalize_mp4(self, name: str) -> Path | None:
        out = self.results_dir / f"{name}.mp4"
        cmd = [
            "ffmpeg", "-y", "-framerate", str(self.fps),
            "-i", str(self._tmp_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(out),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except FileNotFoundError:
            logger.error("ffmpeg not found — install ffmpeg for MP4 recording")
            return None
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed: %s", e.stderr.decode(errors="replace"))
            return None
        finally:
            if self._tmp_dir and self._tmp_dir.exists():
                shutil.rmtree(self._tmp_dir)
        return out
