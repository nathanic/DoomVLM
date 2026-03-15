"""Benchmark and arena runners — orchestrate game loops."""

from __future__ import annotations

import logging
import queue
import threading
import time
from multiprocessing import Event as MPEvent, Process, Queue
from typing import Any

from doom_vlm.engine import run_dm_loop, run_solo_loop

logger = logging.getLogger("doom_dm")


def run_benchmark(
    agent_configs: list[dict],
    game_settings: dict,
    display_mgr: Any,
    stop_event: threading.Event,
) -> list[dict]:
    """Benchmark mode: each agent plays N episodes sequentially vs bots."""
    all_results = []
    num_episodes = game_settings.get("benchmark_episodes", 3)

    for agent_cfg in agent_configs:
        agent_name = agent_cfg["name"]
        display_mgr.log(f"--- Starting benchmark for {agent_name} ({agent_cfg['model']}) ---")
        agent_episodes = []

        for ep in range(1, num_episodes + 1):
            if stop_event.is_set():
                display_mgr.log("Stop requested.")
                break

            display_mgr.log(f"{agent_name}: Episode {ep}/{num_episodes}")
            ep_settings = {**game_settings, "episode": ep}

            status_q: queue.Queue = queue.Queue()
            stop_mp = threading.Event()

            t = threading.Thread(
                target=run_dm_loop,
                args=(agent_cfg, ep_settings, status_q, stop_mp),
                kwargs={"is_host": True},
                daemon=True,
            )
            t.start()

            ep_result = None
            while t.is_alive() or not status_q.empty():
                if stop_event.is_set():
                    stop_mp.set()

                try:
                    msg = status_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if msg["type"] == "step":
                    display_mgr.update_agent(agent_name, msg)
                elif msg["type"] == "done":
                    ep_result = msg
                    display_mgr.log(
                        f"{agent_name} ep{ep}: {msg['frags']:.0f} frags, "
                        f"{msg['deaths']:.0f} deaths, {msg['steps']} steps, "
                        f"{msg['avg_latency']:.1f}s avg"
                    )
                elif msg["type"] == "error":
                    display_mgr.log(f"ERROR ({agent_name}): {msg['error']}")
                    ep_result = {"frags": 0, "deaths": 0, "avg_latency": 0, "steps": 0, "error": msg["error"]}

            t.join(timeout=10)
            if t.is_alive():
                logger.warning("[%s] Episode thread still alive, forcing stop", agent_name)
                stop_mp.set()
                t.join(timeout=5)
            if ep_result:
                agent_episodes.append(ep_result)

        all_results.append({
            "agent": agent_name,
            "model": agent_cfg["model"],
            "episodes": agent_episodes,
        })

    return all_results


def run_arena(
    agent_configs: list[dict],
    game_settings: dict,
    display_mgr: Any,
    stop_event: threading.Event,
) -> list[dict]:
    """Arena mode: all agents in one game via multiprocessing."""
    num_agents = len(agent_configs)
    arena_settings = {
        **game_settings,
        "mode": "arena",
        "num_players": num_agents,
        "episode": 1,
    }

    status_q: Queue = Queue()
    mp_stop = MPEvent()

    processes: list[Process] = []
    for i, agent_cfg in enumerate(agent_configs):
        is_host = (i == 0)
        p = Process(
            target=run_dm_loop,
            args=(agent_cfg, arena_settings, status_q, mp_stop),
            kwargs={"is_host": is_host, "host_address": "127.0.0.1"},
            daemon=True,
        )
        processes.append(p)

    display_mgr.log(f"Starting arena with {num_agents} agents + {game_settings.get('num_bots', 0)} bots")
    processes[0].start()
    display_mgr.log(f"Host ({agent_configs[0]['name']}) started")

    for i in range(1, num_agents):
        time.sleep(5)
        processes[i].start()
        display_mgr.log(f"Client ({agent_configs[i]['name']}) started")

    results: dict[str, dict] = {}
    done_count = 0
    arena_start = time.time()
    arena_timeout = game_settings.get("timelimit", 5) * 60 + 180

    while done_count < num_agents:
        if time.time() - arena_start > arena_timeout:
            display_mgr.log("Arena timeout — aborting")
            mp_stop.set()
            break

        if stop_event.is_set():
            mp_stop.set()

        all_dead = all(not p.is_alive() for p in processes)

        try:
            msg = status_q.get(timeout=1.0)
        except queue.Empty:
            if all_dead:
                break
            continue

        if msg["type"] == "step":
            display_mgr.update_agent(msg["agent"], msg)
        elif msg["type"] == "done":
            done_count += 1
            results[msg["agent"]] = msg
            display_mgr.log(
                f"{msg['agent']} finished: {msg['frags']:.0f} frags, "
                f"{msg['deaths']:.0f} deaths"
            )
        elif msg["type"] == "error":
            done_count += 1
            results[msg["agent"]] = msg
            display_mgr.log(f"ERROR ({msg['agent']}): {msg['error']}")
        elif msg["type"] == "started":
            display_mgr.log(f"{msg['agent']} joined the game")

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    try:
        status_q.close()
        status_q.join_thread()
    except Exception:
        pass

    return [{"agent": name, "result": results.get(name, {})} for name in [c["name"] for c in agent_configs]]


def run_solo_benchmark(
    agent_configs: list[dict],
    game_settings: dict,
    display_mgr: Any,
    stop_event: threading.Event,
) -> list[dict]:
    """Solo benchmark: each agent plays N episodes of a solo scenario sequentially."""
    all_results = []
    num_episodes = game_settings.get("benchmark_episodes", 3)

    for agent_cfg in agent_configs:
        agent_name = agent_cfg["name"]
        display_mgr.log(f"--- Starting solo benchmark for {agent_name} ({agent_cfg['model']}) ---")
        agent_episodes = []

        for ep in range(1, num_episodes + 1):
            if stop_event.is_set():
                display_mgr.log("Stop requested.")
                break

            display_mgr.log(f"{agent_name}: Episode {ep}/{num_episodes}")
            ep_settings = {**game_settings, "episode": ep}

            status_q: queue.Queue = queue.Queue()
            stop_th = threading.Event()

            t = threading.Thread(
                target=run_solo_loop,
                args=(agent_cfg, ep_settings, status_q, stop_th),
                daemon=True,
            )
            t.start()

            ep_result = None
            while t.is_alive() or not status_q.empty():
                if stop_event.is_set():
                    stop_th.set()
                try:
                    msg = status_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                if msg["type"] == "step":
                    display_mgr.update_agent(agent_name, msg)
                elif msg["type"] == "done":
                    ep_result = msg
                    display_mgr.log(
                        f"{agent_name} ep{ep}: {msg.get('kills', 0)} kills, "
                        f"reward={msg.get('reward', 0):.1f}, "
                        f"{msg['steps']} steps, {msg['avg_latency']:.1f}s avg"
                    )
                elif msg["type"] == "error":
                    display_mgr.log(f"ERROR ({agent_name}): {msg['error']}")
                    ep_result = {"kills": 0, "reward": 0, "avg_latency": 0,
                                 "steps": 0, "error": msg["error"]}

            t.join(timeout=10)
            if t.is_alive():
                logger.warning("[%s] Episode thread still alive, forcing stop", agent_name)
                stop_th.set()
                t.join(timeout=5)
            if ep_result:
                agent_episodes.append(ep_result)

        all_results.append({
            "agent": agent_name,
            "model": agent_cfg["model"],
            "episodes": agent_episodes,
        })

    return all_results
