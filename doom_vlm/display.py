"""Terminal display using Rich — replaces ipywidgets GameDisplay."""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

logger = logging.getLogger("doom_dm")


class _DevNull:
    """File-like object that swallows all writes."""
    def write(self, *a, **kw): pass
    def flush(self, *a, **kw): pass
    def fileno(self): raise io.UnsupportedOperation("no fileno")


class TerminalDisplay:
    """Live terminal scoreboard + log using Rich.

    Same interface as the notebook GameDisplay: show(), update_agent(), log(), stop().

    Redirects stdout/stderr so stray prints (VizDoom, failed VLM calls, etc.)
    don't corrupt the Live region.
    """

    def __init__(self, agent_names: list[str], agent_colors: list[str], game_type: str = "dm"):
        self._agent_names = agent_names
        self._agent_colors = {n: c for n, c in zip(agent_names, agent_colors)}
        self._game_type = game_type
        self._status: dict[str, dict[str, Any]] = {}
        self._log_lines: list[str] = []
        self._lock = threading.Lock()
        self._console = Console()
        self._live: Live | None = None
        self._saved_stdout: Any = None
        self._saved_stderr: Any = None
        self._saved_fd_out: int | None = None
        self._saved_fd_err: int | None = None

    def show(self) -> None:
        # Suppress stdout/stderr at the FD level so VizDoom's C-level prints
        # and Python-level warnings don't corrupt the Rich Live region.
        try:
            self._saved_fd_out = os.dup(1)
            self._saved_fd_err = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            pass
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr
        sys.stdout = _DevNull()
        sys.stderr = _DevNull()

        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

        # Restore stdout/stderr
        if self._saved_stdout:
            sys.stdout = self._saved_stdout
            self._saved_stdout = None
        if self._saved_stderr:
            sys.stderr = self._saved_stderr
            self._saved_stderr = None
        try:
            if self._saved_fd_out is not None:
                os.dup2(self._saved_fd_out, 1)
                os.close(self._saved_fd_out)
                self._saved_fd_out = None
            if self._saved_fd_err is not None:
                os.dup2(self._saved_fd_err, 2)
                os.close(self._saved_fd_err)
                self._saved_fd_err = None
        except OSError:
            pass

    def update_agent(self, name: str, step_data: dict) -> None:
        with self._lock:
            self._status[name] = step_data
        if self._live:
            self._live.update(self._render())

    def log(self, msg: str) -> None:
        with self._lock:
            self._log_lines.append(msg)
            if len(self._log_lines) > 20:
                self._log_lines = self._log_lines[-20:]
        logger.info(msg)
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Group:
        return Group(self._render_scoreboard(), self._render_log())

    def _render_scoreboard(self) -> Table:
        if self._game_type == "solo":
            return self._render_solo_scoreboard()
        return self._render_dm_scoreboard()

    def _render_solo_scoreboard(self) -> Table:
        table = Table(title="Solo Scoreboard", expand=True)
        table.add_column("Agent", style="bold")
        table.add_column("Kills", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("HP", justify="right")
        table.add_column("Ammo", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Action")

        with self._lock:
            for name in self._agent_names:
                s = self._status.get(name, {})
                color = self._agent_colors.get(name, "white")
                table.add_row(
                    Text(name, style=f"bold {_css_to_rich(color)}"),
                    str(s.get("kills", 0)),
                    f"{s.get('reward', 0):.1f}",
                    f"{s.get('health', 0):.0f}",
                    f"{s.get('ammo', 0):.0f}",
                    f"{s.get('latency', 0):.1f}s",
                    s.get("action", "-"),
                )
        return table

    def _render_dm_scoreboard(self) -> Table:
        table = Table(title="Deathmatch Scoreboard", expand=True)
        table.add_column("Agent", style="bold")
        table.add_column("Frags", justify="right")
        table.add_column("Deaths", justify="right")
        table.add_column("K/D", justify="right")
        table.add_column("HP", justify="right")
        table.add_column("Ammo", justify="right")
        table.add_column("Latency", justify="right")
        table.add_column("Action")

        with self._lock:
            for name in self._agent_names:
                s = self._status.get(name, {})
                color = self._agent_colors.get(name, "white")
                frags = s.get("frags", 0)
                deaths = s.get("deaths", 0)
                kd = frags / max(deaths, 1)
                table.add_row(
                    Text(name, style=f"bold {_css_to_rich(color)}"),
                    f"{frags:.0f}",
                    f"{deaths:.0f}",
                    f"{kd:.2f}",
                    f"{s.get('health', 0):.0f}",
                    f"{s.get('ammo', 0):.0f}",
                    f"{s.get('latency', 0):.1f}s",
                    s.get("action", "-"),
                )
        return table

    def _render_log(self) -> Table:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        with self._lock:
            for line in self._log_lines:
                table.add_row(Text(line, style="dim"))
        return table


class NullDisplay:
    """No-op display for --no-display / headless mode. Logs to file only."""

    def __init__(self, agent_names: list[str], agent_colors: list[str], game_type: str = "dm"):
        pass

    def show(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def update_agent(self, name: str, step_data: dict) -> None:
        pass

    def log(self, msg: str) -> None:
        logger.info(msg)


def _css_to_rich(css_color: str) -> str:
    """Convert CSS hex color to Rich color name (best-effort)."""
    mapping = {
        "#00cc00": "green",
        "#cc0000": "red",
        "#0066cc": "blue",
        "#cccc00": "yellow",
    }
    return mapping.get(css_color, "white")
