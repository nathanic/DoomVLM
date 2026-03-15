"""Constants, scenario catalog, agent config, TOML loading."""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger("doom_dm")

# ── Vision / action constants ──

FOV_DEGREES = 90
SEARCH_TURN_DELTA = 3

# Button indices
BTN_TURN = 0
BTN_ATTACK = 1
BTN_FWD = 2
BTN_BACK = 3
BTN_LEFT = 4
BTN_RIGHT = 5
BTN_USE = 6
NUM_BUTTONS = 7

# Agent colors (up to 4 agents)
AGENT_COLORS = [
    {"colorset": 0, "css": "#00cc00", "name": "green"},
    {"colorset": 4, "css": "#cc0000", "name": "red"},
    {"colorset": 6, "css": "#0066cc", "name": "blue"},
    {"colorset": 2, "css": "#cccc00", "name": "yellow"},
]

# Default tool descriptions
DEFAULT_SHOOT_DESC = "Shoot at the enemy. Call when you see an enemy on screen."
DEFAULT_MOVE_DESC = "Move around when no enemy is visible."
DEFAULT_COLUMN_DESC = "Enemy Column number 1-{grid_cols}"
DEFAULT_DIRECTION_DESC = "Direction to move"

# Default prompt templates
DEFAULT_SYSTEM_PROMPT = "You are playing DOOM."
DEFAULT_USER_PROMPT = "HP={health} AMMO={ammo}"

# ── Scenario catalog ──

SCENARIO_CATALOG = {
    # Solo scenarios
    "Basic": {
        "game_type": "solo", "cfg": "basic.cfg",
        "desc": "A single stationary monster. Kill it ASAP. 300 tics.",
    },
    "Simpler Basic": {
        "game_type": "solo", "cfg": "simpler_basic.cfg",
        "desc": "Monster centered on screen. Simplest possible test.",
    },
    "Rocket Basic": {
        "game_type": "solo", "cfg": "rocket_basic.cfg",
        "desc": "Like Basic but with a rocket launcher — projectile has travel time.",
    },
    "Defend the Center": {
        "game_type": "solo", "cfg": "defend_the_center.cfg",
        "desc": "Monsters approach from all sides. Defend 360°. 26 bullets.",
    },
    "Defend the Line": {
        "game_type": "solo", "cfg": "defend_the_line.cfg",
        "desc": "Monsters attack from the front. Hold your ground.",
    },
    "Deadly Corridor": {
        "game_type": "solo", "cfg": "deadly_corridor.cfg",
        "desc": "Fight through a corridor of enemies to reach the end. Skill 5/5.",
    },
    "Predict Position": {
        "game_type": "solo", "cfg": "predict_position.cfg",
        "desc": "A monster runs along a wall — predict its position and fire a rocket.",
    },
    "Health Gathering": {
        "game_type": "solo", "cfg": "health_gathering.cfg",
        "desc": "Toxic floor drains HP. Collect medkits to survive. No enemies.",
    },
    "Health Gathering Supreme": {
        "game_type": "solo", "cfg": "health_gathering_supreme.cfg",
        "desc": "Harder Health Gathering. Floor drains faster, fewer medkits.",
    },
    "My Way Home": {
        "game_type": "solo", "cfg": "my_way_home.cfg",
        "desc": "Navigate a maze to find the exit. No enemies. Pure navigation.",
    },
    "Take Cover": {
        "game_type": "solo", "cfg": "take_cover.cfg",
        "desc": "Dodge incoming fireballs by moving left/right. No weapons. Survival.",
    },
    # Deathmatch scenarios
    "CIG map01 (small)": {
        "game_type": "dm", "cfg": "cig.cfg", "map": "map01",
        "desc": "Small VizDoom CIG competition arena. Fast-paced combat.",
    },
    "CIG map02 (large)": {
        "game_type": "dm", "cfg": "cig.cfg", "map": "map02",
        "desc": "Large CIG arena. More room to maneuver.",
    },
    "Multi DM": {
        "game_type": "dm", "cfg": "multi.cfg", "map": "map01",
        "desc": "Standard multiplayer deathmatch map.",
    },
    "Deathmatch": {
        "game_type": "dm", "cfg": "deathmatch.cfg", "map": "map01",
        "desc": "Classic Doom II deathmatch. Large map with many weapons.",
    },
}


def get_scenario_meta(name: str) -> dict:
    return SCENARIO_CATALOG[name]


def is_solo_scenario(name: str) -> bool:
    return SCENARIO_CATALOG[name]["game_type"] == "solo"


# ── Safe prompt formatting ──

class _SafeDict(dict):
    """dict subclass that returns '{key}' for missing keys."""
    def __missing__(self, key):
        logger.warning("Unknown prompt tag: %s", key)
        return "{" + key + "}"


def format_prompt(template: str, **kwargs) -> str:
    return template.format_map(_SafeDict(**kwargs))


# ── Data classes ──

@dataclass
class AgentConfig:
    name: str
    api_url: str
    model: str
    api_key: str = ""
    colorset: int = 0
    color_css: str = "#00cc00"
    temperature: float = 0.7
    top_p: float = 0.8
    presence_penalty: float = 1.5
    max_tokens: int = 200
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    history_len: int = 0
    history_images: bool = False
    shoot_desc: str = DEFAULT_SHOOT_DESC
    move_desc: str = DEFAULT_MOVE_DESC
    column_desc: str = DEFAULT_COLUMN_DESC
    direction_desc: str = DEFAULT_DIRECTION_DESC


@dataclass
class GameSettings:
    type: str = "solo"           # "solo" or "deathmatch"
    scenario: str = "Basic"
    mode: str = "benchmark"      # "benchmark" or "arena" (DM only)
    timing: str = "realtime"     # "sync" or "realtime" (DM only)
    episodes: int = 1
    tics_per_action: int = 4
    grid_cols: int = 5
    image_size: int = 512
    record: str = "none"         # "none", "gif", "mp4"
    bots: int = 0
    time_limit: float = 5.0


def _agent_dict(ac: AgentConfig) -> dict[str, Any]:
    """Convert AgentConfig to a plain dict (for pickling in arena mode)."""
    return {f.name: getattr(ac, f.name) for f in fields(ac)}


# ── TOML config loading ──

def load_config(path: Path) -> tuple[list[AgentConfig], GameSettings]:
    """Load agents + game settings from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Game settings
    gs_data = data.get("game", {})
    gs = GameSettings(
        type=gs_data.get("type", "solo"),
        scenario=gs_data.get("scenario", "Basic"),
        mode=gs_data.get("mode", "benchmark"),
        timing=gs_data.get("timing", "realtime"),
        episodes=gs_data.get("episodes", 1),
        tics_per_action=gs_data.get("tics_per_action", 4),
        grid_cols=gs_data.get("grid_cols", 5),
        image_size=gs_data.get("image_size", 512),
        record=gs_data.get("record", "none"),
        bots=gs_data.get("bots", 0),
        time_limit=gs_data.get("time_limit", 5.0),
    )

    # Agents
    agents_data = data.get("agents", [])
    if not agents_data:
        raise ValueError("Config must define at least one [[agents]] section")

    agents = []
    for i, ad in enumerate(agents_data):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agents.append(AgentConfig(
            name=ad.get("name", f"Agent-{i+1}"),
            api_url=ad["api_url"],
            model=ad.get("model", ""),
            api_key=ad.get("api_key", ""),
            colorset=color["colorset"],
            color_css=color["css"],
            temperature=ad.get("temperature", 0.7),
            top_p=ad.get("top_p", 0.8),
            presence_penalty=ad.get("presence_penalty", 1.5),
            max_tokens=ad.get("max_tokens", 200),
            system_prompt=ad.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            user_prompt=ad.get("user_prompt", DEFAULT_USER_PROMPT),
            history_len=ad.get("history_len", 0),
            history_images=ad.get("history_images", False),
            shoot_desc=ad.get("shoot_desc", DEFAULT_SHOOT_DESC),
            move_desc=ad.get("move_desc", DEFAULT_MOVE_DESC),
            column_desc=ad.get("column_desc", DEFAULT_COLUMN_DESC),
            direction_desc=ad.get("direction_desc", DEFAULT_DIRECTION_DESC),
        ))

    return agents, gs
