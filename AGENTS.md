# AGENTS.md

## Project Overview

DoomVLM is a CLI app (`doom-vlm`) that lets Vision Language Models play Doom via the ViZDoom engine. The AI sees game screenshots, decides actions through OpenAI-compatible tool calling API (`shoot(column)` / `move(direction)`), and the app translates these into game inputs.

**Game modes:** Solo scenarios (11 classic ViZDoom challenges), Deathmatch Benchmark (sequential fair comparison vs bots), Deathmatch Arena (multiprocessing PvP, all agents simultaneously).

**Backend:** Any OpenAI-compatible API with vision + tool calling. Recommended: LM Studio with Qwen 3.5 models.

## Dev Environment Setup

```bash
# Install and run (uv handles dependencies automatically)
uv run doom-vlm example_config.toml

# Or install with pip
pip install -e .
doom-vlm example_config.toml

# Start LM Studio server (install from https://lmstudio.ai/download)
lms server start
lms get qwen-3.5-0.8b
lms load qwen3.5-0.8b --context-length 4096
```

System dependencies for recording:
- macOS: `brew install ffmpeg`
- Linux: `apt-get install ffmpeg fonts-dejavu-core libsdl2-dev zstd`

## Architecture

```
doom_vlm/
  __init__.py        # version string
  __main__.py        # python -m doom_vlm
  cli.py             # argparse + main orchestration, workspace setup, signal handling
  config.py          # constants, SCENARIO_CATALOG, AgentConfig, GameSettings, TOML loading
  imaging.py         # draw_grid_overlay, encode_frame, screen_to_pil, save_debug_screenshot
  vlm.py             # call_vlm, parse_vlm_response, build_action, make_vlm_tools
  engine.py          # setup_solo_game, setup_dm_host/join, run_solo_loop, run_dm_loop
  recorder.py        # EpisodeRecorder (GIF/MP4 with stat overlays)
  runners.py         # run_solo_benchmark, run_benchmark, run_arena
  display.py         # TerminalDisplay (Rich Live) + NullDisplay
  results.py         # Rich result tables, ZIP packaging
pyproject.toml       # project metadata, dependencies, entry point
example_config.toml  # sample TOML config
```

## Key Functions and Classes

**Config (config.py):**
- `AgentConfig` — dataclass with all per-agent settings
- `GameSettings` — dataclass for game configuration
- `load_config(path)` — parses TOML into `(list[AgentConfig], GameSettings)`
- `SCENARIO_CATALOG` — dict of all available scenarios
- `format_prompt(template, **kwargs)` — safe prompt formatting (unknown tags left as-is)

**VLM Integration (vlm.py):**
- `call_vlm(b64_image, user_text, system_prompt, tools, ...)` — sends base64 screenshot + prompt to API, returns `(response_dict, latency)`, retries up to 3 times
- `parse_vlm_response(response, grid_cols)` — extracts tool calls, returns `dict(shoot, cell, move, reason)`, fallback to `move="forward"` on failure
- `build_action(parsed, turn_deltas)` — converts parsed response to 7-element ViZDoom action vector
- `make_vlm_tools(grid_cols, ...)` — generates OpenAI tool definitions

**Game Engine (engine.py):**
- `setup_solo_game(cfg)` — initializes ViZDoom for solo scenarios
- `setup_dm_host(...)` / `setup_dm_join(...)` — multiplayer setup
- `setup_benchmark_game(...)` — single-player deathmatch vs bots
- `run_solo_loop(agent_cfg, game_settings, status_queue, stop_event)` — solo game loop (runs in thread)
- `run_dm_loop(...)` — deathmatch game loop (runs in thread or subprocess)

**Display (display.py):**
- `TerminalDisplay` — Rich Live scoreboard + scrolling log, redirects stdout/stderr to prevent corruption
- `NullDisplay` — no-op for `--no-display` mode

**Runners (runners.py):**
- `run_solo_benchmark(agents, settings, display, stop_event)` — sequential solo episodes
- `run_benchmark(...)` — sequential DM episodes per agent
- `run_arena(...)` — multiprocessing arena, all agents in one game

**CLI (cli.py):**
- Parses args, loads TOML config, creates workspace, sets up logging
- Routes to appropriate runner based on game type/mode
- Two-stage Ctrl+C: first = graceful stop, second = `os._exit(1)`

## Known Issues

1. **macOS multiprocessing:** `multiprocessing.set_start_method("fork")` can deadlock on macOS. Arena mode uses fork. Benchmark mode (threads) is safer.

2. **VLM response parsing:** Some models emit `<think>...</think>` tags or special tokens (`<|im_end|>`, `<|eot_id|>`). These are stripped before parsing. Empty responses fallback to `move="forward"`.

3. **VizDoom C-level prints:** VizDoom prints to stdout/stderr from C code. `TerminalDisplay` redirects file descriptors 1 and 2 to `/dev/null` while the Live display is active to prevent corruption.

## Conventions

- All game logic takes explicit directory parameters (`results_dir`, `screenshot_dir`) — no globals
- Agent configs are converted to plain dicts for pickling in arena mode (`_agent_dict()`)
- `tool_choice: "required"` in API calls to force tool call responses
- Paths use `Path.cwd()` — all relative, no hardcoded user paths
- Agent colors: green (0), red (1), blue (2), yellow (3) — assigned by index
