# DoomVLM

### AI plays Doom — Vision Language Models vs demons and each other

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ViZDoom](https://img.shields.io/badge/Engine-ViZDoom-red)](https://vizdoom.farama.org/)
[![LM Studio](https://img.shields.io/badge/Backend-LM%20Studio-purple)](https://lmstudio.ai/)

<p align="center">
  <img src="media/banner.png" alt="DoomVLM Banner" width="700">
</p>

A CLI app that lets AI vision models play classic Doom. The AI sees the game screen, decides where to shoot or move, and you watch the live scoreboard in your terminal. Pit up to 4 different models against each other in deathmatch — or test them solo on 11 built-in scenarios.

**How it works:** the app takes a screenshot of the game, draws a numbered grid on it, sends it to a vision model, and the model calls `shoot(column)` or `move(direction)`. That's it — the model plays Doom through two simple tools.

```
Screenshot → Grid Overlay → VLM API (tool calling) → shoot / move → Game
```

<p align="center">
  <img src="media/gameplay.gif" alt="DoomVLM Gameplay" width="500">
</p>

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Game Modes](#game-modes)
- [LM Studio Setup](#lm-studio-setup)
- [Alternative Backends](#alternative-backends)
- [RunPod GPU Cloud](#runpod-gpu-cloud)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **11 Solo Scenarios** — classic ViZDoom challenges: shoot monsters, dodge fireballs, navigate mazes, gather health
- **4 Deathmatch Maps** — small and large arenas for multiplayer combat
- **Benchmark Mode** — each agent plays solo vs bots sequentially for fair comparison
- **Arena Mode** — all agents fight at once via multiprocessing, direct PvP
- **1-4 Agents** — each with its own model, API endpoint, prompts, and parameters
- **Live Terminal Scoreboard** — real-time stats via Rich
- **Recording** — save episodes as GIF or MP4 with stat overlays (HP, ammo, kills, VLM reasoning)
- **Tool Use API** — models play through `shoot(column)` and `move(direction)` function calls
- **Customizable Prompts** — template variables `{health}`, `{ammo}`, `{grid_cols}`, etc.
- **TOML Config** — all settings in a single file, no GUI required
- **Any OpenAI-compatible API** — works with LM Studio, Ollama, vLLM, OpenRouter, OpenAI, and more

---

## Quick Start

### 1. Install LM Studio and load a model

Download [LM Studio](https://lmstudio.ai/download), then:

```bash
lms server start
lms get qwen-3.5-0.8b        # interactive — pick lmstudio-community, Q8_0
lms load qwen3.5-0.8b --context-length 4096
```

### 2. Install DoomVLM

```bash
git clone https://github.com/nathanic/DoomVLM.git
cd DoomVLM
uv run doom-vlm --list-scenarios   # installs deps automatically on first run
```

Or with pip:

```bash
pip install -e .
doom-vlm --list-scenarios
```

### 3. Run

```bash
uv run doom-vlm example_config.toml
```

That's it — the agent plays Basic (a single stationary monster) using your local LM Studio server.

---

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| **OS** | macOS, Linux | macOS, Linux |
| **Python** | 3.11 | 3.12+ |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 2 GB (for smallest model) | 10+ GB |

> Windows is supported through WSL.

**Linux system dependencies** (for headless rendering and recording):

```bash
sudo apt-get install ffmpeg fonts-dejavu-core libsdl2-dev zstd
```

**macOS** (for MP4 recording):

```bash
brew install ffmpeg
```

### Model sizes

| Model | VRAM / RAM | Quality |
|---|---|---|
| Qwen3.5-0.8B | ~1 GB | Basic — good for testing |
| Qwen3.5-2B | ~3 GB | Good |
| Qwen3.5-4B | ~5 GB | Better |
| Qwen3.5-9B | ~10 GB | Best |

### Performance reference

| Hardware | Model | Inference per step |
|---|---|---|
| MacBook M1 Pro 16 GB (CPU/MLX) | Qwen3.5-0.8B | ~10 sec |
| [RunPod](https://runpod.io) L40S (GGUF Q8) | Qwen3.5-0.8B | ~0.5 sec |

---

## Configuration

DoomVLM uses TOML config files. The config has two sections: `[game]` for game settings and `[[agents]]` for agent definitions.

### Minimal config

```toml
[game]
type = "solo"
scenario = "Basic"

[[agents]]
name = "Agent-1"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"
```

### Full game settings

```toml
[game]
type = "solo"              # "solo" or "deathmatch"
scenario = "Basic"         # scenario name (use --list-scenarios to see all)
episodes = 3               # number of episodes per agent
grid_cols = 5              # screen divisions for aiming (3-10, more = finer aim)
tics_per_action = 4        # game ticks between VLM decisions (1-8, fewer = faster)
image_size = 512           # screenshot resolution sent to VLM (128-1024)
record = "mp4"             # "none", "gif", or "mp4"

# Deathmatch-only settings (ignored for solo):
mode = "benchmark"         # "benchmark" (sequential) or "arena" (simultaneous)
timing = "realtime"        # "realtime" or "sync"
bots = 4                   # built-in AI opponents (0-7)
time_limit = 5.0           # game duration in minutes
```

### Agent settings

Each `[[agents]]` block defines one agent (up to 4). Colors are assigned automatically: green, red, blue, yellow.

```toml
[[agents]]
name = "Agent-1"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"
api_key = ""                       # required for cloud APIs, leave empty for local
temperature = 0.7
top_p = 0.8
presence_penalty = 1.5
max_tokens = 200
history_len = 0                    # previous turns to include (0 = no memory)
history_images = false             # include screenshots in history (uses more VRAM)
system_prompt = "You are playing DOOM."
user_prompt = "HP={health} AMMO={ammo}"
shoot_desc = "Shoot at the enemy. Call when you see an enemy on screen."
move_desc = "Move around when no enemy is visible."
column_desc = "Enemy Column number 1-{grid_cols}"
direction_desc = "Direction to move"
```

### Multiple agents

Add more `[[agents]]` blocks — TOML's double-bracket syntax creates an array:

```toml
[[agents]]
name = "small"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"

[[agents]]
name = "large"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-4b"
```

### Using cloud APIs

```toml
[[agents]]
name = "GPT-4o"
api_url = "https://api.openai.com/v1/chat/completions"
model = "gpt-4o"
api_key = "sk-..."

[[agents]]
name = "Local"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"
```

### Prompt template variables

Use these in `system_prompt`, `user_prompt`, and tool descriptions — they're replaced with live game values:

| Variable | Available In | Description |
|---|---|---|
| `{grid_cols}` | System, User, Tools | Number of grid columns on screen |
| `{health}` | User prompt | Current HP |
| `{ammo}` | User prompt | Current ammo count |
| `{frags}` | User prompt | Frag count (deathmatch) |
| `{deaths}` | User prompt | Death count (deathmatch) |
| `{kills}` | User prompt | Kill count (solo) |
| `{reward}` | User prompt | Cumulative reward (solo) |
| `{step}` | User prompt | Current game step number |

### Conversation history

Set `history_len` to give the agent memory of previous turns:

```toml
[[agents]]
name = "Agent-1"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-4b"
history_len = 5              # remember last 5 turns
history_images = false       # text-only history (cheaper)
```

With `history_images = true`, each history turn includes the screenshot — more context but significantly more tokens/VRAM. Only useful with larger context windows (8k+).

---

## CLI Reference

```
doom-vlm [config.toml] [options]
```

| Option | Description |
|---|---|
| `config.toml` | Path to TOML config file |
| `--scenario NAME` | Override scenario from config |
| `--episodes N` | Override episode count |
| `--record gif\|mp4\|none` | Override recording format |
| `--no-display` | Headless mode (log to file only, no Rich display) |
| `--zip` | Package all workspace results as ZIP after run |
| `--list-scenarios` | Print scenario table and exit |

Examples:

```bash
# Quick test
doom-vlm example_config.toml

# Override scenario and record
doom-vlm example_config.toml --scenario "Defend the Center" --record mp4

# Headless (CI/server)
doom-vlm config.toml --no-display --episodes 10

# List all scenarios
doom-vlm --list-scenarios
```

Ctrl+C to stop: first press initiates graceful shutdown, second press force-quits.

---

## Game Modes

### Solo Scenarios

Classic ViZDoom training scenarios. The agent plays alone against built-in enemies. Death ends the episode.

| Scenario | Description |
|---|---|
| **Basic** | A single stationary monster. Kill it ASAP. 300 tics. |
| **Simpler Basic** | Monster centered on screen. Simplest possible test. |
| **Rocket Basic** | Like Basic but with a rocket launcher — projectile has travel time. |
| **Defend the Center** | Monsters approach from all sides. Defend 360°. 26 bullets. |
| **Defend the Line** | Monsters attack from the front. Hold your ground. |
| **Deadly Corridor** | Fight through a corridor of enemies to reach the end. Skill 5/5. |
| **Predict Position** | A monster runs along a wall — predict its position and fire a rocket. |
| **Health Gathering** | Toxic floor drains HP. Collect medkits to survive. No enemies. |
| **Health Gathering Supreme** | Harder Health Gathering. Floor drains faster, fewer medkits. |
| **My Way Home** | Navigate a maze to find the exit. No enemies. Pure navigation. |
| **Take Cover** | Dodge incoming fireballs by moving left/right. No weapons. Survival. |

Solo config:

```toml
[game]
type = "solo"
scenario = "Defend the Center"
episodes = 5
record = "mp4"
```

### Deathmatch — Benchmark Mode

Each agent plays **solo vs bots**, sequentially. Identical conditions for every agent — fair comparison.

```toml
[game]
type = "deathmatch"
scenario = "CIG map01 (small)"
mode = "benchmark"
bots = 4
time_limit = 3.0
episodes = 3
```

Results: Frags, Deaths, K/D ratio, average API latency per agent per episode.

### Deathmatch — Arena Mode

All agents play **together in one game** via multiprocessing. Direct PvP — a faster model gets more actions per second in realtime mode.

```toml
[game]
type = "deathmatch"
scenario = "CIG map01 (small)"
mode = "arena"
timing = "realtime"
bots = 0
time_limit = 5.0
```

- **`timing = "realtime"`** — faster models get more turns (realistic competition)
- **`timing = "sync"`** — the game waits for each VLM response before advancing (fair but slow)

### Deathmatch Maps

| Map | Description |
|---|---|
| **CIG map01 (small)** | Small VizDoom CIG competition arena. Fast-paced combat. |
| **CIG map02 (large)** | Large CIG arena. More room to maneuver. |
| **Multi DM** | Standard multiplayer deathmatch map. |
| **Deathmatch** | Classic Doom II deathmatch. Large map with many weapons. |

### Running multiple instances in parallel

Solo and Benchmark modes are safe to run in parallel — each instance spawns its own isolated VizDoom process with no shared state. Arena mode is **not** safe to run in parallel: VizDoom's multiplayer uses a default network port, so two concurrent arenas would have clients joining the wrong game. Run concurrent arenas in separate containers or network namespaces if you need this.

---

## LM Studio Setup

[LM Studio](https://lmstudio.ai/) is the recommended backend — it runs models locally with zero configuration.

### Installation

1. Download from [lmstudio.ai/download](https://lmstudio.ai/download)
2. Install and open the application
3. Search for a Qwen 3.5 model in the Discover tab and download it

### CLI (headless)

```bash
lms server start                              # start API server
lms get qwen-3.5-0.8b                        # download (interactive picker)
lms load qwen3.5-0.8b --context-length 4096  # load into memory
lms ls                                        # list downloaded models
lms ps                                        # list loaded models
lms server stop                               # stop server
```

### Key details

- API endpoint: `http://localhost:1234/v1/chat/completions`
- All models share port 1234 — selected via the `model` field in the request
- Models load automatically on first API request (JIT loading)
- Tool calling (function calling) is supported natively
- `--context-length 4096` is enough for DoomVLM (each step sends one screenshot + short prompt). Use 8192 if you set `history_len > 0`.

### Official documentation

- [LM Studio Docs](https://lmstudio.ai/docs/) — main documentation
- [CLI Reference](https://lmstudio.ai/docs/cli) — all CLI commands
- [Tool Use / Function Calling](https://lmstudio.ai/docs/developer/openai-compat/tools) — how tool calling works

---

## Alternative Backends

DoomVLM works with **any OpenAI-compatible API** that supports vision and tool calling. Change the `api_url` and `model` in the agent config.

| Backend | `api_url` | Notes |
|---|---|---|
| [LM Studio](https://lmstudio.ai/) | `http://localhost:1234/v1/chat/completions` | Recommended. Local, free, easy setup |
| [Ollama](https://ollama.com/) | `http://localhost:11434/v1/chat/completions` | Local, free. Needs vision + tool calling model |
| [vLLM](https://docs.vllm.ai/) | `http://localhost:8000/v1/chat/completions` | Local, free. GPU required |
| [OpenRouter](https://openrouter.ai/) | `https://openrouter.ai/api/v1/chat/completions` | Cloud. Many models. Requires `api_key` |
| [OpenAI](https://platform.openai.com/) | `https://api.openai.com/v1/chat/completions` | Cloud. GPT-4o has vision + tools. Requires `api_key` |

> Cloud APIs work directly from your local machine — no LM Studio or GPU needed.

---

## RunPod GPU Cloud

Don't have a GPU? Run DoomVLM on a cloud GPU with [RunPod](https://runpod.io) — inference drops from ~10 sec/step (MacBook CPU) to ~0.5 sec/step (L40S).

### Setup

1. Sign up at [runpod.io](https://runpod.io)
2. Deploy a Pod with a GPU (L40S recommended)
3. SSH in and install:

```bash
# Install LM Studio
curl -fsSL https://lmstudio.ai/install.sh | bash
lms server start
lms get qwen-3.5-0.8b
lms load qwen3.5-0.8b --context-length 4096

# Clone and run DoomVLM
git clone https://github.com/nathanic/DoomVLM.git
cd DoomVLM
uv run doom-vlm example_config.toml --record mp4
```

> With an L40S you can serve multiple models simultaneously and run 4-agent Arena battles at real-time speed.

---

## How It Works

Each game step:

```
1. ViZDoom renders a frame
2. Screenshot is taken and a numbered grid overlay is drawn
3. Image is resized, JPEG-compressed, and base64-encoded
4. API request sent to the VLM with:
   - System prompt (instructions)
   - User prompt (HP, ammo, etc.) + screenshot
   - Tool definitions: shoot(column) and move(direction)
   - tool_choice: "required"
5. Model responds with a tool call:
   - shoot(column=3) → turn to column 3 and fire
   - move(direction="forward") → move forward
6. Tool call is converted to a ViZDoom action vector
7. Action is applied for N tics (tics_per_action)
8. Repeat
```

### Tools

**`shoot(column)`** — shoot at the specified grid column (1 to N). The app calculates the turn angle needed to aim at that column based on the FOV.

**`move(direction)`** — move in the given direction: `forward`, `backward`, `left`, `right`, `strafe_left`, `strafe_right`.

### Output files

Each run creates a timestamped directory inside `workspace/`:

```
workspace/0001_20260311_221500/
├── game.log          # Detailed log (every VLM call, action, stats)
├── results/          # GIF/MP4 recordings with stat overlays
└── screenshots/      # Per-step debug images (screenshot + VLM response)
```

Use `--zip` to package everything into a single archive.

---

## Examples

### Quick test — single agent on Basic

```bash
doom-vlm example_config.toml
```

The agent should kill the stationary monster within a few steps.

### Model comparison — Benchmark

Create `benchmark.toml`:

```toml
[game]
type = "deathmatch"
scenario = "CIG map01 (small)"
mode = "benchmark"
bots = 3
time_limit = 2.0
episodes = 3

[[agents]]
name = "0.8B"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"

[[agents]]
name = "2B"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-2b"

[[agents]]
name = "4B"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-4b"
```

```bash
doom-vlm benchmark.toml --record mp4
```

Each agent plays 3 episodes against bots. Compare frags, deaths, and K/D ratio in the results table.

### Battle Royale — Arena

Create `arena.toml`:

```toml
[game]
type = "deathmatch"
scenario = "CIG map01 (small)"
mode = "arena"
timing = "realtime"
bots = 0
time_limit = 5.0

[[agents]]
name = "qwen-0.8b"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-0.8b"

[[agents]]
name = "qwen-4b"
api_url = "http://localhost:1234/v1/chat/completions"
model = "qwen3.5-4b"
```

```bash
doom-vlm arena.toml --record mp4
```

### Solo gauntlet — all scenarios

```bash
for s in "Basic" "Defend the Center" "Deadly Corridor" "Health Gathering"; do
  doom-vlm example_config.toml --scenario "$s" --episodes 3 --record mp4
done
```

### Headless CI run

```bash
doom-vlm config.toml --no-display --episodes 10 --zip
```

---

## Troubleshooting

### LM Studio server not responding

```bash
lms server start
lms ps                    # check loaded models
curl http://localhost:1234/v1/models
```

### Model not found

```bash
lms ls                    # list downloaded models
lms get qwen-3.5-0.8b    # download if needed
```

### MP4 recording doesn't work

Install ffmpeg:

```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Empty VLM responses / agent spinning in place

The app has a built-in fallback — on empty responses, the agent moves forward. Try:
- Increasing `max_tokens` (e.g., to 300)
- Lowering `temperature` (e.g., to 0.5)
- Using a larger model

### Arena mode issues on macOS

Arena uses Python multiprocessing with `fork`, which can deadlock on macOS. If Arena freezes, use Benchmark mode instead — it provides the same competitive data via sequential runs.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [Mykyta Roshchenko](https://github.com/Felliks) — created the [original DoomVLM notebook](https://github.com/Felliks/DoomVLM) that this project is based on
- [ViZDoom](https://vizdoom.farama.org/) by [Farama Foundation](https://farama.org/) — the Doom research platform that makes this possible
- [LM Studio](https://lmstudio.ai/) — local LLM inference made easy
- [Qwen](https://github.com/QwenLM/Qwen3) by Alibaba — the recommended vision language models
- [id Software](https://www.idsoftware.com/) — for creating Doom
