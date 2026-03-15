"""CLI entry point — replaces Jupyter notebook UI."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from doom_vlm.config import (
    AGENT_COLORS, SCENARIO_CATALOG, AgentConfig, GameSettings,
    _agent_dict, load_config,
)
from doom_vlm.display import NullDisplay, TerminalDisplay
from doom_vlm.results import (
    package_zip, print_arena_results, print_dm_benchmark_results,
    print_solo_results,
)
from doom_vlm.runners import run_arena, run_benchmark, run_solo_benchmark

logger = logging.getLogger("doom_dm")
console = Console()


def _list_scenarios() -> None:
    table = Table(title="Available Scenarios")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Config")
    table.add_column("Description")

    for name, meta in SCENARIO_CATALOG.items():
        table.add_row(
            name,
            meta["game_type"],
            meta["cfg"],
            meta["desc"],
        )
    console.print(table)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doom-vlm",
        description="Run VLM agents in DOOM via VizDoom",
    )
    p.add_argument("config", nargs="?", help="Path to TOML config file")
    p.add_argument("--scenario", help="Override scenario name")
    p.add_argument("--episodes", type=int, help="Override number of episodes")
    p.add_argument("--record", choices=["gif", "mp4", "none"], help="Override recording format")
    p.add_argument("--no-display", action="store_true", help="Headless mode (log to file only)")
    p.add_argument("--zip", action="store_true", help="Package results as ZIP after run")
    p.add_argument("--list-scenarios", action="store_true", help="List available scenarios and exit")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    agents, gs = load_config(config_path)

    # CLI overrides
    if args.scenario:
        gs.scenario = args.scenario
    if args.episodes is not None:
        gs.episodes = args.episodes
    if args.record:
        gs.record = args.record

    # Validate scenario
    if gs.scenario not in SCENARIO_CATALOG:
        console.print(f"[red]Unknown scenario: {gs.scenario}[/red]")
        console.print("Use --list-scenarios to see available options.")
        sys.exit(1)

    scenario_meta = SCENARIO_CATALOG[gs.scenario]

    # Validate game type matches scenario
    expected_type = "dm" if gs.type == "deathmatch" else "solo"
    if scenario_meta["game_type"] != expected_type:
        console.print(
            f"[red]Scenario '{gs.scenario}' is {scenario_meta['game_type']}, "
            f"but game type is '{gs.type}'[/red]"
        )
        sys.exit(1)

    # Set SDL_VIDEODRIVER before any VizDoom imports in subprocesses
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # multiprocessing fork guard for arena mode
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass

    # Create per-run workspace
    base_dir = Path.cwd()
    workspace_root = base_dir / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.name for d in workspace_root.iterdir() if d.is_dir())
    seq = int(existing[-1].split("_")[0]) + 1 if existing else 1
    run_id = f"{seq:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = workspace_root / run_id
    results_dir = run_dir / "results"
    screenshot_dir = run_dir / "screenshots"
    for d in (results_dir, screenshot_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Setup logging — file only; TerminalDisplay handles all terminal output
    # via Rich Live, so no StreamHandler (it would corrupt the display).
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(run_dir / "game.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Log run configuration
    logger.info("=" * 60)
    logger.info("Run: %s", run_id)
    logger.info("Game type: %s | Scenario: %s (%s)", gs.type, gs.scenario, scenario_meta["cfg"])
    logger.info("Grid: %d cols | Image: %dpx | Tics/action: %d", gs.grid_cols, gs.image_size, gs.tics_per_action)
    if gs.record != "none":
        logger.info("Recording: %s", gs.record)
    for i, ac in enumerate(agents):
        logger.info("--- Agent %d: %s ---", i + 1, ac.name)
        logger.info("  Model: %s @ %s", ac.model, ac.api_url)
        logger.info("  Temp=%.1f top_p=%.2f pres_pen=%.1f max_tok=%d",
                     ac.temperature, ac.top_p, ac.presence_penalty, ac.max_tokens)
    logger.info("=" * 60)

    # Convert AgentConfigs to dicts
    agent_dicts = [_agent_dict(ac) for ac in agents]
    agent_names = [a["name"] for a in agent_dicts]
    agent_css_colors = [a["color_css"] for a in agent_dicts]

    # Ctrl+C handling: first press = graceful stop, second = hard kill
    stop_event = threading.Event()
    _sigint_count = 0

    def _sigint_handler(sig, frame):
        nonlocal _sigint_count
        _sigint_count += 1
        if _sigint_count == 1:
            console.print("\n[yellow]Stopping... (Ctrl+C again to force quit)[/yellow]")
            stop_event.set()
        else:
            console.print("\n[red]Force quit.[/red]")
            os._exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Create display
    game_type_display = "solo" if gs.type == "solo" else "dm"
    if args.no_display:
        display_mgr = NullDisplay(agent_names, agent_css_colors, game_type=game_type_display)
    else:
        display_mgr = TerminalDisplay(agent_names, agent_css_colors, game_type=game_type_display)

    record_fmt = gs.record if gs.record != "none" else None

    console.print(f"[bold]Workspace:[/bold] {run_id}")
    console.print(f"[bold]Scenario:[/bold] {gs.scenario} ({scenario_meta['desc']})")
    agents_str = ", ".join(f"{a.name} ({a.model})" for a in agents)
    console.print(f"[bold]Agents:[/bold] {agents_str}")
    console.print()

    display_mgr.show()
    dm_results = []

    try:
        if gs.type == "solo":
            game_settings = {
                "cfg": scenario_meta["cfg"],
                "scenario_label": gs.scenario,
                "tics_per_action": gs.tics_per_action,
                "grid_cols": gs.grid_cols,
                "max_dim": gs.image_size,
                "benchmark_episodes": gs.episodes,
                "record_fmt": record_fmt,
                "results_dir": str(results_dir),
                "screenshot_dir": str(screenshot_dir),
            }

            display_mgr.log(f"Game type: Solo | Scenario: {gs.scenario}")
            display_mgr.log(f"Episodes: {gs.episodes}")

            dm_results = run_solo_benchmark(
                agent_dicts, game_settings, display_mgr, stop_event,
            )
            display_mgr.log("=== Solo benchmark complete ===")

        else:
            # Deathmatch
            game_settings = {
                "mode": gs.mode,
                "dm_mode": gs.timing,
                "scenario": scenario_meta["cfg"],
                "map_name": scenario_meta.get("map", "map01"),
                "num_bots": gs.bots,
                "timelimit": gs.time_limit,
                "tics_per_action": gs.tics_per_action,
                "grid_cols": gs.grid_cols,
                "max_dim": gs.image_size,
                "benchmark_episodes": gs.episodes,
                "record_fmt": record_fmt,
                "results_dir": str(results_dir),
                "screenshot_dir": str(screenshot_dir),
            }

            display_mgr.log(f"Mode: {gs.mode} | Scenario: {gs.scenario}")
            display_mgr.log(f"Bots: {gs.bots}, Time limit: {gs.time_limit} min, Timing: {gs.timing}")

            if gs.mode == "arena":
                dm_results = run_arena(agent_dicts, game_settings, display_mgr, stop_event)
            else:
                dm_results = run_benchmark(agent_dicts, game_settings, display_mgr, stop_event)

            display_mgr.log("=== Deathmatch complete ===")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Game run failed")
    finally:
        display_mgr.stop()

    # Print results
    if dm_results:
        console.print()
        if gs.type == "solo":
            print_solo_results(dm_results, console)
        elif gs.mode == "arena":
            print_arena_results(dm_results, console)
        else:
            print_dm_benchmark_results(dm_results, console)

    # ZIP packaging
    if args.zip:
        zip_path = base_dir / "doom_deathmatch_results.zip"
        result = package_zip(workspace_root, zip_path)
        if result:
            size_mb = result.stat().st_size / 1_000_000
            console.print(f"[green]Packed results into {result} ({size_mb:.1f} MB)[/green]")
        else:
            console.print("[yellow]Nothing to pack.[/yellow]")

    # Log final summary
    for r in dm_results:
        if "episodes" in r:
            for ep in r["episodes"]:
                if gs.type == "solo":
                    logger.info("RESULT %s ep%d: kills=%s reward=%s steps=%s avg_lat=%.1fs",
                                r["agent"], ep.get("episode", 0), ep.get("kills", "?"),
                                ep.get("reward", "?"), ep.get("steps", "?"), ep.get("avg_latency", 0))
                else:
                    logger.info("RESULT %s ep%d: frags=%s deaths=%s steps=%s avg_lat=%.1fs",
                                r["agent"], ep.get("episode", 0), ep.get("frags", "?"),
                                ep.get("deaths", "?"), ep.get("steps", "?"), ep.get("avg_latency", 0))
        elif "result" in r:
            res = r["result"]
            logger.info("RESULT %s: frags=%s deaths=%s",
                        r["agent"], res.get("frags", "?"), res.get("deaths", "?"))

    console.print(f"\n[dim]Logs: {run_dir / 'game.log'}[/dim]")
