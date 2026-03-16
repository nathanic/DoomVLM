"""VLM communication: API calls, response parsing, action building."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests

from doom_vlm.config import (
    BTN_ATTACK, BTN_BACK, BTN_FWD, BTN_LEFT, BTN_RIGHT, BTN_TURN, BTN_USE,
    DEFAULT_COLUMN_DESC, DEFAULT_DIRECTION_DESC, DEFAULT_MOVE_DESC,
    DEFAULT_SHOOT_DESC, DEFAULT_SYSTEM_PROMPT, FOV_DEGREES, NUM_BUTTONS,
    SEARCH_TURN_DELTA, format_prompt,
)

logger = logging.getLogger("doom_dm")


def compute_grid_turn_deltas(grid_cols: int, tics_per_action: int = 4) -> dict[int, int]:
    """Compute per-tic turn deltas for each column."""
    deltas = {}
    center = (grid_cols + 1) / 2.0
    col_width_deg = FOV_DEGREES / grid_cols
    for col in range(1, grid_cols + 1):
        offset_deg = (col - center) * col_width_deg
        deltas[col] = round(offset_deg / tics_per_action)
    deltas[0] = SEARCH_TURN_DELTA
    return deltas


def make_vlm_tools(
    grid_cols: int,
    shoot_desc: str = DEFAULT_SHOOT_DESC,
    move_desc: str = DEFAULT_MOVE_DESC,
    column_desc: str = DEFAULT_COLUMN_DESC,
    direction_desc: str = DEFAULT_DIRECTION_DESC,
) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "shoot",
                "description": shoot_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "integer",
                            "description": format_prompt(column_desc, grid_cols=grid_cols),
                            "enum": list(range(1, grid_cols + 1)),
                        },
                    },
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "move",
                "description": move_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "description": direction_desc,
                            "enum": ["forward", "backward", "left", "right",
                                     "strafe_left", "strafe_right"],
                        },
                    },
                    "required": ["direction"],
                },
            },
        },
    ]


def parse_vlm_response(data: dict, grid_cols: int) -> dict[str, str]:
    """Parse VLM response — extract tool call shoot(column) or move(direction)."""
    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls")

    if tool_calls:
        tc = tool_calls[0]
        fn_name = tc["function"]["name"]
        args_raw = tc["function"]["arguments"]
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            return {"shoot": "no", "cell": "0", "move": "forward",
                    "reason": f"malformed tool args: {str(args_raw)[:100]}"}

        if fn_name == "shoot":
            col = int(args["column"])
            if not 1 <= col <= grid_cols:
                raise ValueError(f"shoot column {col} out of range 1-{grid_cols}")
            return {
                "shoot": "yes", "cell": str(col),
                "move": "none", "reason": f"shoot(column={col})",
            }

        if fn_name == "move":
            direction = args["direction"]
            return {
                "shoot": "no", "cell": "0",
                "move": direction, "reason": f"move(direction={direction})",
            }

        raise ValueError(f"Unknown tool call: {fn_name}")

    # No tool call — strip reasoning tags and special tokens, fallback to forward
    content = (message.get("content") or "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    content = re.sub(r"<\|im_end\|>|<\|end\|>|<\|eot_id\|>|</s>", "", content).strip()
    return {
        "shoot": "no", "cell": "0",
        "move": "forward",
        "reason": content or "no tool call, moving forward",
    }


def build_action(parsed: dict[str, str], turn_deltas: dict[int, int]) -> list[float]:
    """Translate parsed VLM response into a VizDoom action vector."""
    action = [0.0] * NUM_BUTTONS

    cell = int(parsed["cell"])
    shoot = parsed["shoot"] == "yes"
    move = parsed["move"]
    search_delta = float(turn_deltas[0])

    if shoot:
        action[BTN_TURN] = float(turn_deltas[cell])
        action[BTN_ATTACK] = 1.0
    elif move != "none":
        if move == "forward":
            action[BTN_FWD] = 1.0
        elif move == "backward":
            action[BTN_BACK] = 1.0
        elif move == "left":
            action[BTN_TURN] = -search_delta
        elif move == "right":
            action[BTN_TURN] = search_delta
        elif move == "strafe_left":
            action[BTN_LEFT] = 1.0
        elif move == "strafe_right":
            action[BTN_RIGHT] = 1.0
    else:
        action[BTN_TURN] = search_delta

    return action


def call_vlm(
    b64_image: str,
    user_text: str,
    system_prompt: str,
    tools: list[dict],
    *,
    api_url: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 0.8,
    presence_penalty: float = 1.5,
    max_tokens: int = 200,
    history: list[dict] | None = None,
    timeout: float = 120,
    api_key: str = "",
    session_id: str = "",
) -> tuple[dict, float]:
    """Send screenshot with grid overlay to VLM. All params are explicit."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            },
        ],
    })
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "required",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
    }
    if session_id:
        payload["litellm_session_id"] = session_id

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code >= 400:
                body = resp.text[:500]
                logger.warning("[%s] VLM attempt %d HTTP %d: %s",
                               model, attempt + 1, resp.status_code, body)
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    logger.error("[%s] VLM failed after 3 attempts: HTTP %d", model, resp.status_code)
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}: {body}", response=resp)
            data = resp.json()
            elapsed = time.perf_counter() - t0
            msg = data.get("choices", [{}])[0].get("message", {})
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls")
            if content:
                logger.info("[%s] VLM %.1fs content: %s", model, elapsed, content)
            if tool_calls:
                tc = tool_calls[0]
                logger.info("[%s] VLM %.1fs tool_call: %s(%s)",
                            model, elapsed, tc["function"]["name"],
                            tc["function"]["arguments"])
            elif not content:
                logger.warning("[%s] VLM %.1fs empty response (no tool_calls, no content)",
                               model, elapsed)
            return data, elapsed
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                wait = 1 * (attempt + 1)
                logger.warning("[%s] VLM attempt %d failed: %s, retrying in %ds",
                               model, attempt + 1, e, wait)
                time.sleep(wait)
            else:
                logger.error("[%s] VLM failed after 3 attempts: %s", model, e)
                raise


def make_dm_system_prompt(grid_cols: int, template: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return format_prompt(template, grid_cols=grid_cols)
