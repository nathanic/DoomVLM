"""Game engine: setup functions and game loops for solo and deathmatch."""

from __future__ import annotations

import collections
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import vizdoom as vzd

from doom_vlm.config import (
    BTN_TURN, DEFAULT_COLUMN_DESC, DEFAULT_DIRECTION_DESC,
    DEFAULT_MOVE_DESC, DEFAULT_SHOOT_DESC, DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT, NUM_BUTTONS, format_prompt,
)
from doom_vlm.imaging import draw_grid_overlay, encode_frame, save_debug_screenshot, screen_to_pil
from doom_vlm.recorder import EpisodeRecorder
from doom_vlm.vlm import (
    build_action, call_vlm, compute_grid_turn_deltas,
    make_dm_system_prompt, make_vlm_tools, parse_vlm_response,
)

logger = logging.getLogger("doom_dm")


# ── Button / variable configuration ──

def _configure_dm_buttons(game: vzd.DoomGame) -> None:
    game.clear_available_buttons()
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 10)
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.USE)


def _configure_dm_variables(game: vzd.DoomGame) -> None:
    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.AMMO2)
    game.add_available_game_variable(vzd.GameVariable.FRAGCOUNT)
    game.add_available_game_variable(vzd.GameVariable.DEATHCOUNT)


def _configure_solo_variables(game: vzd.DoomGame) -> None:
    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.AMMO2)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)


# ── Game setup ──

def setup_dm_host(
    scenario: str, map_name: str, num_players: int,
    timelimit: float, agent_name: str, colorset: int,
    dm_mode: str = "realtime",
) -> vzd.DoomGame:
    agent_name = re.sub(r"[^a-zA-Z0-9_-]", "", agent_name)[:20]
    cfg_path = os.path.join(vzd.scenarios_path, scenario)
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_window_visible(False)

    _configure_dm_buttons(game)
    _configure_dm_variables(game)

    game.add_game_args(f"-host {num_players} -deathmatch")
    game.add_game_args(f"+timelimit {timelimit}")
    game.add_game_args("+sv_forcerespawn 1")
    game.add_game_args("+sv_noautoaim 1")
    game.add_game_args("+sv_respawnprotect 1")
    game.add_game_args("+sv_spawnfarthest 1")
    game.add_game_args("+sv_nocrouch 1")
    game.add_game_args(f"+name {agent_name}")
    game.add_game_args(f"+colorset {colorset}")
    game.add_game_args(f"+map {map_name}")

    game.set_mode(vzd.Mode.PLAYER if dm_mode == "sync" else vzd.Mode.ASYNC_PLAYER)
    game.init()
    return game


def setup_dm_join(
    scenario: str, agent_name: str, colorset: int,
    host_address: str = "127.0.0.1",
    dm_mode: str = "realtime",
) -> vzd.DoomGame:
    agent_name = re.sub(r"[^a-zA-Z0-9_-]", "", agent_name)[:20]
    cfg_path = os.path.join(vzd.scenarios_path, scenario)
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_window_visible(False)

    _configure_dm_buttons(game)
    _configure_dm_variables(game)

    game.add_game_args(f"-join {host_address}")
    game.add_game_args(f"+name {agent_name}")
    game.add_game_args(f"+colorset {colorset}")

    game.set_mode(vzd.Mode.PLAYER if dm_mode == "sync" else vzd.Mode.ASYNC_PLAYER)
    game.init()
    return game


def setup_benchmark_game(
    scenario: str, map_name: str, timelimit: float,
    num_bots: int, agent_name: str, colorset: int,
) -> vzd.DoomGame:
    agent_name = re.sub(r"[^a-zA-Z0-9_-]", "", agent_name)[:20]
    cfg_path = os.path.join(vzd.scenarios_path, scenario)
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_window_visible(False)

    _configure_dm_buttons(game)
    _configure_dm_variables(game)

    game.add_game_args(f"-host 1 -deathmatch")
    game.add_game_args(f"+timelimit {timelimit}")
    game.add_game_args("+sv_forcerespawn 1")
    game.add_game_args("+sv_noautoaim 1")
    game.add_game_args("+sv_respawnprotect 1")
    game.add_game_args("+sv_spawnfarthest 1")
    game.add_game_args("+sv_nocrouch 1")
    game.add_game_args(f"+name {agent_name}")
    game.add_game_args(f"+colorset {colorset}")
    game.add_game_args(f"+map {map_name}")

    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    for _ in range(num_bots):
        game.send_game_command("addbot")

    return game


def setup_solo_game(cfg_name: str) -> vzd.DoomGame:
    cfg_path = os.path.join(vzd.scenarios_path, cfg_name)
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)

    _configure_dm_buttons(game)
    _configure_solo_variables(game)

    game.init()
    return game


# ── Game variable extraction ──

def get_dm_game_vars(state) -> dict[str, float]:
    gv = state.game_variables if state.game_variables is not None else []
    names = ["health", "ammo", "frags", "deaths"]
    if len(gv) < len(names):
        logger.warning("Expected %d game vars, got %d", len(names), len(gv))
    return {name: float(val) for name, val in zip(names, gv)}


def get_solo_game_vars(state) -> dict[str, float]:
    gv = state.game_variables if state.game_variables is not None else []
    names = ["health", "ammo", "kills"]
    if len(gv) < len(names):
        logger.warning("Expected %d game vars, got %d", len(names), len(gv))
    return {name: float(val) for name, val in zip(names, gv)}


# ── Game loops ──

def run_dm_loop(
    agent_cfg: dict,
    game_settings: dict,
    status_queue: Any,
    stop_event: Any,
    is_host: bool = True,
    host_address: str = "127.0.0.1",
) -> None:
    """Main deathmatch loop for a single agent.
    Runs in a subprocess (arena) or thread (benchmark).
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    agent_name = agent_cfg["name"]
    episode = game_settings.get("episode", 1)
    results_dir = Path(game_settings.get("results_dir", "results"))
    screenshot_dir = Path(game_settings.get("screenshot_dir", "screenshots"))
    game = None
    recorder = None

    try:
        mode = game_settings["mode"]
        scenario = game_settings["scenario"]
        map_name = game_settings["map_name"]
        timelimit = game_settings["timelimit"]
        num_bots = game_settings.get("num_bots", 0)
        num_players = game_settings.get("num_players", 1)
        tics_per_action = game_settings["tics_per_action"]
        grid_cols = game_settings["grid_cols"]
        max_dim = game_settings["max_dim"]
        record_fmt = game_settings.get("record_fmt")
        dm_mode = game_settings.get("dm_mode", "realtime")
        include_images = agent_cfg.get("history_images", False)

        if mode == "benchmark":
            game = setup_benchmark_game(
                scenario, map_name, timelimit,
                num_bots, agent_name, agent_cfg["colorset"],
            )
        elif is_host:
            game = setup_dm_host(
                scenario, map_name, num_players,
                timelimit, agent_name, agent_cfg["colorset"],
                dm_mode=dm_mode,
            )
            for _ in range(num_bots):
                game.send_game_command("addbot")
        else:
            game = setup_dm_join(
                scenario, agent_name, agent_cfg["colorset"],
                host_address=host_address,
                dm_mode=dm_mode,
            )

        turn_deltas = compute_grid_turn_deltas(grid_cols, tics_per_action)
        system_prompt = make_dm_system_prompt(
            grid_cols,
            agent_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
        user_prompt_template = agent_cfg.get("user_prompt", DEFAULT_USER_PROMPT)
        tools = make_vlm_tools(
            grid_cols,
            shoot_desc=agent_cfg.get("shoot_desc", DEFAULT_SHOOT_DESC),
            move_desc=agent_cfg.get("move_desc", DEFAULT_MOVE_DESC),
            column_desc=agent_cfg.get("column_desc", DEFAULT_COLUMN_DESC),
            direction_desc=agent_cfg.get("direction_desc", DEFAULT_DIRECTION_DESC),
        )

        history_len = agent_cfg.get("history_len", 0)
        history_buffer = collections.deque(maxlen=history_len) if history_len > 0 else None

        recorder = (
            EpisodeRecorder(
                episode, scenario.replace(".cfg", ""), record_fmt,
                grid_cols, player_name=agent_name, game_type="dm",
                results_dir=results_dir,
            )
            if record_fmt
            else None
        )

        game.new_episode()
        step = 0
        total_latency = 0.0
        respawn_count = 0
        frags = 0.0
        deaths = 0.0

        logger.info("[%s] Episode %d started", agent_name, episode)

        status_queue.put({
            "type": "started",
            "agent": agent_name,
            "episode": episode,
        })

        while not game.is_episode_finished():
            if stop_event.is_set():
                break

            if game.is_player_dead():
                game.respawn_player()
                respawn_count += 1
                continue

            state = game.get_state()
            if state is None:
                game.make_action([0.0] * NUM_BUTTONS, 1)
                continue

            step += 1
            gv = get_dm_game_vars(state)
            health = gv.get("health", 100.0)
            ammo = gv.get("ammo", 0.0)
            frags = gv.get("frags", 0.0)
            deaths = gv.get("deaths", 0.0)

            img = screen_to_pil(state.screen_buffer)
            img_with_grid = draw_grid_overlay(img, grid_cols)
            b64 = encode_frame(img_with_grid, max_dim=max_dim)

            user_text = format_prompt(
                user_prompt_template,
                health=int(health), ammo=int(ammo),
                frags=int(frags), deaths=int(deaths),
                step=step, grid_cols=grid_cols,
            )

            history = [msg for turn in history_buffer for msg in turn] if history_buffer else None

            try:
                vlm_data, latency = call_vlm(
                    b64, user_text, system_prompt, tools,
                    api_url=agent_cfg["api_url"],
                    model=agent_cfg["model"],
                    temperature=agent_cfg["temperature"],
                    top_p=agent_cfg["top_p"],
                    presence_penalty=agent_cfg["presence_penalty"],
                    max_tokens=agent_cfg["max_tokens"],
                    history=history,
                    api_key=agent_cfg.get("api_key", ""),
                )
                parsed = parse_vlm_response(vlm_data, grid_cols)

                if history_buffer is not None:
                    if include_images:
                        user_entry = {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]}
                    else:
                        user_entry = {"role": "user", "content": user_text}
                    assistant_msg = vlm_data["choices"][0]["message"]
                    asst_entry: dict[str, Any] = {"role": "assistant", "content": assistant_msg.get("content")}
                    if assistant_msg.get("tool_calls"):
                        asst_entry["tool_calls"] = assistant_msg["tool_calls"]
                    turn: list[dict] = [user_entry, asst_entry]
                    if assistant_msg.get("tool_calls"):
                        for tc in assistant_msg["tool_calls"]:
                            turn.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})
                    history_buffer.append(turn)

            except Exception as e:
                logger.warning("[%s] step %d VLM error: %s", agent_name, step, e)
                parsed = {"shoot": "no", "cell": "0", "move": "forward", "reason": f"VLM error: {e}"}
                vlm_data = {}
                latency = 0.0

            action_vec = build_action(parsed, turn_deltas)
            total_latency += latency

            cell = int(parsed.get("cell", "0"))
            shoot = parsed.get("shoot", "no") == "yes"
            move_cmd = parsed.get("move", "none")
            act_str = f"SHOOT@{cell}" if shoot else (f"MOVE:{move_cmd}" if move_cmd != "none" else "search")
            turn_delta = action_vec[BTN_TURN]
            action_desc = f"turn={turn_delta:+.0f}\u00b0 act={act_str}"

            logger.debug("[%s] step %d: %s | HP=%.0f AMMO=%.0f F=%.0f D=%.0f",
                         agent_name, step, action_desc, health, ammo, frags, deaths)

            if recorder:
                recorder.set_step_context(
                    step, health, ammo, frags,
                    parsed, action_desc, 0.0, latency,
                )

            reward = 0.0
            for tic in range(tics_per_action):
                if game.is_episode_finished():
                    break
                if game.is_player_dead():
                    break
                if mode == "benchmark":
                    tic_reward = game.make_action(action_vec, 1)
                    reward += tic_reward
                else:
                    game.make_action(action_vec, 1)

                if recorder and not game.is_episode_finished():
                    tic_state = game.get_state()
                    if tic_state:
                        recorder.capture_tic(tic_state.screen_buffer, tic, tics_per_action)

            if not game.is_episode_finished() and not game.is_player_dead():
                post_state = game.get_state()
                if post_state:
                    post_gv = get_dm_game_vars(post_state)
                    frags = post_gv.get("frags", frags)
                    deaths = post_gv.get("deaths", deaths)

            frame_b64 = encode_frame(img_with_grid, max_dim=380)
            status_queue.put({
                "type": "step",
                "agent": agent_name,
                "episode": episode,
                "step": step,
                "frags": frags,
                "deaths": deaths,
                "health": health,
                "ammo": ammo,
                "latency": round(latency, 2),
                "action": act_str,
                "frame_b64": frame_b64,
            })

            save_debug_screenshot(
                img_with_grid, agent_name, episode, step,
                parsed, parsed["reason"], action_desc,
                reward, health, ammo, latency,
                screenshot_dir=screenshot_dir,
            )

        # Episode finished
        try:
            final_frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            final_deaths = game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
        except Exception:
            final_frags = frags
            final_deaths = deaths
        scoreboard = []
        try:
            server = game.get_server_state()
            for i in range(len(server.players_in_game)):
                if server.players_in_game[i]:
                    scoreboard.append({
                        "name": server.players_names[i],
                        "frags": server.players_frags[i],
                    })
        except Exception as e:
            logger.debug("Scoreboard error: %s", e)

        rec_path = None
        if recorder:
            rec_path = recorder.finalize()

        logger.info("[%s] Episode %d done: frags=%.0f deaths=%.0f steps=%d avg_lat=%.1fs",
                    agent_name, episode, final_frags, final_deaths, step,
                    total_latency / max(step, 1))

        status_queue.put({
            "type": "done",
            "agent": agent_name,
            "episode": episode,
            "frags": final_frags,
            "deaths": final_deaths,
            "steps": step,
            "avg_latency": round(total_latency / max(step, 1), 2),
            "scoreboard": scoreboard,
            "recording": str(rec_path) if rec_path else None,
        })

    except Exception as e:
        status_queue.put({
            "type": "error",
            "agent": agent_name,
            "episode": episode,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if recorder:
            try:
                recorder.finalize()
            except Exception:
                pass
        if game is not None:
            try:
                game.close()
            except Exception:
                pass


def run_solo_loop(
    agent_cfg: dict,
    game_settings: dict,
    status_queue: Any,
    stop_event: Any,
) -> None:
    """Solo scenario game loop for a single agent.
    Runs in a thread. Death = episode over (no respawn).
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    agent_name = agent_cfg["name"]
    episode = game_settings.get("episode", 1)
    results_dir = Path(game_settings.get("results_dir", "results"))
    screenshot_dir = Path(game_settings.get("screenshot_dir", "screenshots"))
    game = None
    recorder = None

    try:
        cfg_name = game_settings["cfg"]
        tics_per_action = game_settings["tics_per_action"]
        grid_cols = game_settings["grid_cols"]
        max_dim = game_settings["max_dim"]
        record_fmt = game_settings.get("record_fmt")
        scenario_label = game_settings.get("scenario_label", cfg_name)
        include_images = agent_cfg.get("history_images", False)

        game = setup_solo_game(cfg_name)

        turn_deltas = compute_grid_turn_deltas(grid_cols, tics_per_action)
        system_prompt = make_dm_system_prompt(
            grid_cols,
            agent_cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
        user_prompt_template = agent_cfg.get("user_prompt", DEFAULT_USER_PROMPT)
        tools = make_vlm_tools(
            grid_cols,
            shoot_desc=agent_cfg.get("shoot_desc", DEFAULT_SHOOT_DESC),
            move_desc=agent_cfg.get("move_desc", DEFAULT_MOVE_DESC),
            column_desc=agent_cfg.get("column_desc", DEFAULT_COLUMN_DESC),
            direction_desc=agent_cfg.get("direction_desc", DEFAULT_DIRECTION_DESC),
        )

        history_len = agent_cfg.get("history_len", 0)
        history_buffer = collections.deque(maxlen=history_len) if history_len > 0 else None

        recorder = (
            EpisodeRecorder(
                episode, scenario_label.replace(" ", "_").lower(), record_fmt,
                grid_cols, player_name=agent_name, game_type="solo",
                results_dir=results_dir,
            )
            if record_fmt
            else None
        )

        game.new_episode()
        step = 0
        total_latency = 0.0
        total_reward = 0.0
        kills = 0
        prev_kills = 0.0

        logger.info("[%s] Solo episode %d started — %s", agent_name, episode, scenario_label)

        status_queue.put({
            "type": "started",
            "agent": agent_name,
            "episode": episode,
        })

        while not game.is_episode_finished():
            if stop_event.is_set():
                break

            if game.is_player_dead():
                break

            state = game.get_state()
            if state is None:
                game.make_action([0.0] * NUM_BUTTONS, 1)
                continue

            step += 1
            gv = get_solo_game_vars(state)
            health = gv.get("health", 100.0)
            ammo = gv.get("ammo", 0.0)
            cur_kills = gv.get("kills", 0.0)

            img = screen_to_pil(state.screen_buffer)
            img_with_grid = draw_grid_overlay(img, grid_cols)
            b64 = encode_frame(img_with_grid, max_dim=max_dim)

            user_text = format_prompt(
                user_prompt_template,
                health=int(health), ammo=int(ammo),
                kills=int(cur_kills), reward=round(total_reward, 1),
                step=step, grid_cols=grid_cols,
                frags=int(cur_kills), deaths=0,
            )

            history = [msg for turn in history_buffer for msg in turn] if history_buffer else None

            try:
                vlm_data, latency = call_vlm(
                    b64, user_text, system_prompt, tools,
                    api_url=agent_cfg["api_url"],
                    model=agent_cfg["model"],
                    temperature=agent_cfg["temperature"],
                    top_p=agent_cfg["top_p"],
                    presence_penalty=agent_cfg["presence_penalty"],
                    max_tokens=agent_cfg["max_tokens"],
                    history=history,
                    api_key=agent_cfg.get("api_key", ""),
                )
                parsed = parse_vlm_response(vlm_data, grid_cols)

                if history_buffer is not None:
                    if include_images:
                        user_entry = {"role": "user", "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]}
                    else:
                        user_entry = {"role": "user", "content": user_text}
                    assistant_msg = vlm_data["choices"][0]["message"]
                    asst_entry: dict[str, Any] = {"role": "assistant", "content": assistant_msg.get("content")}
                    if assistant_msg.get("tool_calls"):
                        asst_entry["tool_calls"] = assistant_msg["tool_calls"]
                    turn: list[dict] = [user_entry, asst_entry]
                    if assistant_msg.get("tool_calls"):
                        for tc in assistant_msg["tool_calls"]:
                            turn.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})
                    history_buffer.append(turn)

            except Exception as e:
                logger.warning("[%s] step %d VLM error: %s", agent_name, step, e)
                parsed = {"shoot": "no", "cell": "0", "move": "forward",
                          "reason": f"VLM error: {e}"}
                vlm_data = {}
                latency = 0.0

            action_vec = build_action(parsed, turn_deltas)
            total_latency += latency

            cell = int(parsed.get("cell", "0"))
            shoot = parsed.get("shoot", "no") == "yes"
            move_cmd = parsed.get("move", "none")
            act_str = (f"SHOOT@{cell}" if shoot
                       else (f"MOVE:{move_cmd}" if move_cmd != "none" else "search"))
            turn_delta = action_vec[BTN_TURN]
            action_desc = f"turn={turn_delta:+.0f}\u00b0 act={act_str}"

            if recorder:
                recorder.set_step_context(
                    step, health, ammo, cur_kills,
                    parsed, action_desc, total_reward, latency,
                )

            reward = 0.0
            for tic in range(tics_per_action):
                if game.is_episode_finished() or game.is_player_dead():
                    break
                tic_reward = game.make_action(action_vec, 1)
                reward += tic_reward
                if recorder and not game.is_episode_finished():
                    tic_state = game.get_state()
                    if tic_state:
                        recorder.capture_tic(tic_state.screen_buffer, tic, tics_per_action)

            total_reward += reward

            if not game.is_episode_finished() and not game.is_player_dead():
                post_state = game.get_state()
                if post_state:
                    post_gv = get_solo_game_vars(post_state)
                    post_kills = post_gv.get("kills", cur_kills)
                    if post_kills > prev_kills:
                        kills += int(post_kills - prev_kills)
                    prev_kills = post_kills

            frame_b64 = encode_frame(img_with_grid, max_dim=380)
            status_queue.put({
                "type": "step",
                "agent": agent_name,
                "episode": episode,
                "step": step,
                "kills": kills,
                "reward": round(total_reward, 1),
                "health": health,
                "ammo": ammo,
                "latency": round(latency, 2),
                "action": act_str,
                "frame_b64": frame_b64,
            })

            save_debug_screenshot(
                img_with_grid, agent_name, episode, step,
                parsed, parsed["reason"], action_desc,
                reward, health, ammo, latency,
                screenshot_dir=screenshot_dir,
            )

        # Episode finished
        try:
            final_kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
            if final_kills > prev_kills:
                kills += int(final_kills - prev_kills)
        except Exception:
            pass

        ep_total_reward = game.get_total_reward()

        rec_path = None
        if recorder:
            rec_path = recorder.finalize()

        logger.info("[%s] Solo ep %d done: kills=%d reward=%.1f steps=%d avg_lat=%.1fs",
                    agent_name, episode, kills, ep_total_reward, step,
                    total_latency / max(step, 1))

        status_queue.put({
            "type": "done",
            "agent": agent_name,
            "episode": episode,
            "kills": kills,
            "reward": round(ep_total_reward, 1),
            "steps": step,
            "avg_latency": round(total_latency / max(step, 1), 2),
            "recording": str(rec_path) if rec_path else None,
        })

    except Exception as e:
        status_queue.put({
            "type": "error",
            "agent": agent_name,
            "episode": game_settings.get("episode", 1),
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
    finally:
        if recorder:
            try:
                recorder.finalize()
            except Exception:
                pass
        if game is not None:
            try:
                game.close()
            except Exception:
                pass
