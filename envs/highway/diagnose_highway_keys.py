"""Diagnose how keyboard input reaches highway-env (stock highway-v0, default game config).

For every decision step it prints: which physics frames the step covered, every key press and the
frame it arrived on, every action actually applied to the car (with the resulting target lane /
target speed), and what a one-action-per-step recorder would save.

Game config: all highway-v0 defaults (simulation_frequency 15, policy_frequency 1, duration 40 s, ...).
Two NON-game switches are needed so a human can play at all:
  manual_control=True       keyboard drives the car (default False: the action given to step() drives it)
  real_time_rendering=True  show the 15 frames at real speed (default False: as fast as the CPU allows)

Run:  python diagnose_highway_keys.py                 # one episode, Ctrl+C to stop early
      python diagnose_highway_keys.py --no-real-time  # everything default except manual_control
Versions: gymnasium 0.29.1, highway-env 1.9.1 (same as the cluster's GAIL env).
"""
import argparse

import gymnasium as gym
import highway_env  # noqa: F401  registers highway-v0
import pygame
from highway_env.envs.common.graphics import EventHandler

p = argparse.ArgumentParser()
p.add_argument("--no-real-time", action="store_true", help="leave real_time_rendering at its default (False)")
a = p.parse_args()

env = gym.make("highway-v0", render_mode="human")
u = env.unwrapped
u.configure({"manual_control": True, "real_time_rendering": not a.no_real_time})
cfg = u.config
print(f"game config (defaults): simulation_frequency={cfg['simulation_frequency']}  "
      f"policy_frequency={cfg['policy_frequency']}  -> "
      f"{cfg['simulation_frequency'] // cfg['policy_frequency']} physics frames per step, "
      f"duration={cfg['duration']} s")
print(f"display switches: manual_control={cfg['manual_control']}  real_time_rendering={cfg['real_time_rendering']}")

KEYS = {pygame.K_UP: "UP", pygame.K_DOWN: "DOWN", pygame.K_LEFT: "LEFT", pygame.K_RIGHT: "RIGHT"}
log = {"frame": 0, "events": []}  # events of the current step: (frame, text)


def hook_frames():
    """Count physics frames: highway calls road.step() once per simulation frame."""
    road_step = u.road.step

    def step(dt):
        road_step(dt)
        log["frame"] += 1
    u.road.step = step


_handle_event = EventHandler.handle_event.__func__


def handle_event(cls, action_type, event):
    """See every arrow-key press exactly when highway's window receives it."""
    if event.type == pygame.KEYDOWN and event.key in KEYS:
        log["events"].append((log["frame"], f"key {KEYS[event.key]}"))
    _handle_event(cls, action_type, event)


EventHandler.handle_event = classmethod(handle_event)


def hook_actions(applied):
    """See every action actually applied to the car (keyboard or step()), and its effect."""
    at = u.action_type
    names = {i: n for n, i in at.actions_indexes.items()}
    act = at.act

    def logged_act(action):
        act(action)
        v = u.vehicle
        applied.append(int(action))
        log["events"].append((log["frame"], f"APPLIED {names[int(action)]:10s} -> target lane "
                                            f"{v.target_lane_index[2]}, target speed {v.target_speed:.0f}"))
    at.act = logged_act
    return names


try:
    obs, info = env.reset()
    applied = []
    hook_frames()                      # reset() builds a new road and action_type, so hook after it
    names = hook_actions(applied)
    idle = u.action_type.actions_indexes["IDLE"]
    k = 0
    done = False
    while not done:
        start = log["frame"]
        applied.clear()
        log["events"].clear()
        obs, rew, terminated, truncated, info = env.step(idle)   # ignored under manual_control
        done = terminated or truncated
        v = u.vehicle
        n_keys = sum(1 for _, t in log["events"] if t.startswith("key"))
        recorded = names[applied[-1]] if applied else "IDLE"
        print(f"\nstep {k:3d}  frames {start}-{log['frame'] - 1}  step() was given IDLE (ignored: manual_control)")
        for frame, text in log["events"]:
            print(f"    frame {frame:4d}  {text}")
        print(f"    keys: {n_keys}  applied: {len(applied)}  one-per-step recorder saves: {recorded}"
              + ("   <-- more than one action this step: the record loses some" if len(applied) > 1 else ""))
        print(f"    end of step: lane {v.lane_index[2]}  speed {v.speed:.1f}  "
              f"target lane {v.target_lane_index[2]}  target speed {v.target_speed:.0f}"
              + ("  CRASHED" if info.get("crashed") else ""))
        k += 1
except KeyboardInterrupt:
    print("\nstopped")
env.close()
