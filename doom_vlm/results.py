"""Results display and ZIP packaging."""

from __future__ import annotations

import zipfile
from pathlib import Path

from rich.console import Console
from rich.table import Table


def print_solo_results(results: list[dict], console: Console | None = None) -> None:
    """Print solo benchmark results as Rich tables."""
    console = console or Console()

    for agent_result in results:
        agent_name = agent_result["agent"]
        model = agent_result["model"]
        episodes = agent_result.get("episodes", [])

        table = Table(title=f"{agent_name} ({model})")
        table.add_column("Episode", justify="right")
        table.add_column("Kills", justify="right")
        table.add_column("Reward", justify="right")
        table.add_column("Steps", justify="right")
        table.add_column("Avg Latency", justify="right")

        for ep in episodes:
            table.add_row(
                str(ep.get("episode", "?")),
                str(ep.get("kills", 0)),
                f"{ep.get('reward', 0):.1f}",
                str(ep.get("steps", 0)),
                f"{ep.get('avg_latency', 0):.1f}s",
            )

        if episodes:
            n = len(episodes)
            avg_kills = sum(e.get("kills", 0) for e in episodes) / n
            avg_reward = sum(e.get("reward", 0) for e in episodes) / n
            avg_steps = sum(e.get("steps", 0) for e in episodes) / n
            avg_lat = sum(e.get("avg_latency", 0) for e in episodes) / n
            table.add_section()
            table.add_row(
                "AVG",
                f"{avg_kills:.1f}",
                f"{avg_reward:.1f}",
                f"{avg_steps:.0f}",
                f"{avg_lat:.1f}s",
                style="bold",
            )

        console.print(table)
        console.print()


def print_dm_benchmark_results(results: list[dict], console: Console | None = None) -> None:
    """Print DM benchmark results as Rich tables."""
    console = console or Console()

    for agent_result in results:
        agent_name = agent_result["agent"]
        model = agent_result["model"]
        episodes = agent_result.get("episodes", [])

        table = Table(title=f"{agent_name} ({model})")
        table.add_column("Episode", justify="right")
        table.add_column("Frags", justify="right")
        table.add_column("Deaths", justify="right")
        table.add_column("K/D", justify="right")
        table.add_column("Avg Latency", justify="right")

        for ep in episodes:
            frags = ep.get("frags", 0)
            deaths = ep.get("deaths", 0)
            kd = frags / max(deaths, 1)
            table.add_row(
                str(ep.get("episode", "?")),
                f"{frags:.0f}",
                f"{deaths:.0f}",
                f"{kd:.2f}",
                f"{ep.get('avg_latency', 0):.1f}s",
            )

        if episodes:
            n = len(episodes)
            avg_frags = sum(e.get("frags", 0) for e in episodes) / n
            avg_deaths = sum(e.get("deaths", 0) for e in episodes) / n
            avg_kd = avg_frags / max(avg_deaths, 1)
            avg_lat = sum(e.get("avg_latency", 0) for e in episodes) / n
            table.add_section()
            table.add_row("AVG", f"{avg_frags:.1f}", f"{avg_deaths:.1f}",
                          f"{avg_kd:.2f}", f"{avg_lat:.1f}s", style="bold")

        console.print(table)
        console.print()


def print_arena_results(results: list[dict], console: Console | None = None) -> None:
    """Print arena scoreboard sorted by frags."""
    console = console or Console()

    arena_rows = []
    for r in results:
        result = r.get("result", {})
        arena_rows.append({
            "agent": r["agent"],
            "frags": result.get("frags", 0),
            "deaths": result.get("deaths", 0),
        })
    arena_rows.sort(key=lambda x: x["frags"], reverse=True)

    table = Table(title="Arena Scoreboard")
    table.add_column("#", justify="right")
    table.add_column("Player")
    table.add_column("Frags", justify="right")
    table.add_column("Deaths", justify="right")
    table.add_column("K/D", justify="right")

    for i, r in enumerate(arena_rows, 1):
        kd = r["frags"] / max(r["deaths"], 1)
        table.add_row(str(i), r["agent"], f"{r['frags']:.0f}",
                      f"{r['deaths']:.0f}", f"{kd:.2f}")

    console.print(table)
    console.print()


def package_zip(workspace_root: Path, output_path: Path) -> Path | None:
    """ZIP-package the entire workspace directory."""
    if not workspace_root.exists():
        return None

    file_count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(workspace_root.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f"workspace/{f.relative_to(workspace_root)}")
                file_count += 1

    if file_count > 0:
        return output_path
    output_path.unlink(missing_ok=True)
    return None
