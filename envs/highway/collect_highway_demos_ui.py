"""OUR EDITION: highway-env demos where the human follows the agent's rule, with a game UI.

(DEFAULT EDITION = collect_highway_demos.py: highway's own manual control, keys act mid-step.)

Game: stock highway-v0, default config (15 physics frames per decision step, 1 decision per second).
Input follows the agent's rules ("latest key wins"): while a step's 15 frames play, each arrow press
overwrites the NEXT action; it is applied on the first frame of the next step, exactly like an
agent's action, and that applied action is what gets recorded. manual_control is NOT used.

Keys:  SPACE start an episode    Q quit (an unfinished episode is not saved)
       Up = lane left (up the screen)   Down = lane right   Right = faster   Left = slower
Saves per finished episode: epNNN.npz (obs, acts, rews, terminal, seed, key_log) + meta.json.
key_log rows = (step, frame within step, action) for EVERY arrow press, including overwritten ones.
Versions: gymnasium 0.29.1, highway-env 1.9.1 (same as the cluster's GAIL env).
"""
import argparse, json, os
from importlib.metadata import version

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  registers highway-v0
import pygame

p = argparse.ArgumentParser()
p.add_argument("--env", default="highway-v0")
p.add_argument("--out", default="highway_demos_ui")
p.add_argument("--zoom", type=float, default=2.0, help="display size only; does not change the game")
a = p.parse_args()

env = gym.make(a.env, render_mode="rgb_array")  # we show the frames ourselves
u = env.unwrapped
# Display setting, not game: in highway-env 1.9.1 offscreen_rendering defaults to False even for
# rgb_array, and then highway opens its own window AND reads the keyboard itself, applying arrow
# keys straight to the car mid-step. Off-screen, it only draws; all keys come to this script.
u.configure({"offscreen_rendering": True})
cfg = u.config
FPS = cfg["simulation_frequency"]                                    # 15 frames per second of game time
FRAMES_PER_STEP = cfg["simulation_frequency"] // cfg["policy_frequency"]  # 15
MAX_STEPS = int(cfg["duration"] * cfg["policy_frequency"])           # 40

ACT = u.action_type.actions_indexes  # {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
NAME = {i: n for n, i in ACT.items()}
KEYMAP = {pygame.K_UP: ACT["LANE_LEFT"], pygame.K_DOWN: ACT["LANE_RIGHT"],
          pygame.K_RIGHT: ACT["FASTER"], pygame.K_LEFT: ACT["SLOWER"]}

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump({"env": a.env, "config": "env defaults, nothing changed",
               "simulation_frequency": cfg["simulation_frequency"], "policy_frequency": cfg["policy_frequency"],
               "input_rule": "latest key wins; applied on the first frame of the next step",
               "gymnasium": version("gymnasium"), "highway_env": version("highway-env")}, f, indent=2)
n_saved = len([f for f in os.listdir(a.out) if f.endswith(".npz")])

# ---- window: the game inside a box, a panel underneath ----
W, H = int(cfg["screen_width"] * a.zoom), int(cfg["screen_height"] * a.zoom)
PAD, PANEL = 20, 150
pygame.init()
screen = pygame.display.set_mode((W + 2 * PAD, H + 2 * PAD + PANEL))
font, small = pygame.font.Font(None, 32), pygame.font.Font(None, 24)
clock = pygame.time.Clock()

ui = {"mode": "waiting", "frame": None, "msg": "Press SPACE to start",
      "step": 0, "f": 0, "ret": 0.0, "now": "IDLE", "next": None, "quit": False}
key_log = []  # (step, frame, action) for every arrow press in the current episode


def handle_events():
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
            ui["quit"] = True
        elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE and ui["mode"] == "waiting":
            ui["mode"] = "start"
        elif e.type == pygame.KEYDOWN and e.key in KEYMAP and ui["mode"] == "running":
            ui["next"] = KEYMAP[e.key]                                  # latest key wins
            key_log.append((ui["step"], ui["f"], KEYMAP[e.key]))


def text(s, x, y, f=font, color=(225, 225, 225)):
    screen.blit(f.render(s, True, color), (x, y))


def draw():
    screen.fill((25, 25, 25))
    if ui["frame"] is not None:
        img = pygame.surfarray.make_surface(ui["frame"].swapaxes(0, 1))
        screen.blit(pygame.transform.smoothscale(img, (W, H)), (PAD, PAD))
    pygame.draw.rect(screen, (140, 140, 140), (PAD - 3, PAD - 3, W + 6, H + 6), 3)  # the box
    y = H + 2 * PAD
    if ui["mode"] == "running":
        text(f"Episode {n_saved}    step {ui['step'] + 1} / {MAX_STEPS}    return {ui['ret']:.1f}", PAD, y)
        text(f"Taking now: {ui['now']}", PAD, y + 40, color=(120, 200, 255))
        nxt = NAME[ui["next"]] if ui["next"] is not None else "IDLE (no key yet)"
        text(f"Next: {nxt}", PAD + W // 2, y + 40, color=(255, 210, 90))
        bar = W // 2 - PAD                                              # countdown to the next decision
        left = max(0.0, 1 - ui["f"] / FRAMES_PER_STEP)
        pygame.draw.rect(screen, (70, 70, 70), (PAD + W // 2, y + 75, bar, 12))
        pygame.draw.rect(screen, (255, 210, 90), (PAD + W // 2, y + 75, int(bar * left), 12))
    else:
        text(ui["msg"], PAD, y + 20, color=(255, 210, 90))
    text("SPACE start    Up lane left    Down lane right    Right faster    Left slower    Q quit",
         PAD, y + 110, small, (150, 150, 150))
    pygame.display.flip()


# highway renders every in-between physics frame through env.render(); show each one in the box
_render = u.render


def render_and_show():
    img = _render()
    ui["frame"] = img
    if ui["mode"] == "running":
        ui["f"] += 1
    handle_events()
    draw()
    clock.tick(FPS)  # real time: 15 frames per second
    return img


u.render = render_and_show

rng = np.random.default_rng()
try:
    while not ui["quit"]:
        seed = int(rng.integers(2**31 - 1))
        obs, info = env.reset(seed=seed)
        ui.update(mode="waiting", step=0, f=0, ret=0.0, now="IDLE", next=None)
        key_log.clear()
        u.render()                       # show the starting road (also re-arms highway's in-between frames)
        pygame.display.set_caption("highway-env demo collection")
        while ui["mode"] == "waiting" and not ui["quit"]:
            handle_events()
            draw()
            clock.tick(30)
        if ui["quit"]:
            break

        ui["mode"] = "running"
        ep_obs, ep_acts, ep_rews = [obs], [], []
        done = False
        while not done and not ui["quit"]:
            action = ACT["IDLE"] if ui["next"] is None else ui["next"]
            ui.update(next=None, now=NAME[action], f=0)
            obs, rew, terminated, truncated, info = env.step(action)  # applied on frame 0, like an agent
            u.render()                   # last frame of the step (+ re-arms the in-between frames)
            ep_obs.append(obs)
            ep_acts.append(action)
            ep_rews.append(rew)
            ui["step"] += 1
            ui["ret"] += rew
            done = terminated or truncated
        if not done:
            print("quit mid-episode: that episode was not saved")
            break

        path = os.path.join(a.out, f"ep{n_saved:03d}.npz")
        np.savez(path, obs=np.array(ep_obs, np.float32), acts=np.array(ep_acts, np.int64),
                 rews=np.array(ep_rews, np.float64), terminal=terminated, seed=seed,
                 key_log=np.array(key_log, np.int64).reshape(-1, 3))
        outcome = "CRASHED" if info.get("crashed") else "time up"
        print(f"saved {path}: {len(ep_acts)} steps, return {sum(ep_rews):.1f}, {outcome}, "
              f"{len(key_log)} key presses")
        n_saved += 1
        ui["msg"] = f"Saved: {outcome}, return {sum(ep_rews):.1f}.  Press SPACE for the next episode"
except KeyboardInterrupt:
    print("\nstopped; the unfinished episode was not saved")
env.close()
pygame.quit()
