"""DEFAULT EDITION: highway-env's own manual control, with a game UI and a rule check.

Game: stock highway-v0 (default config; only --policy-frequency can change it, default 1).
Input: exactly highway-env's default manual control. Every arrow press goes to highway's own
EventHandler and acts on the car immediately, on whatever frame it arrives (mid-step). Several
presses in one step ALL act on the car. One action per step is recorded: the LAST one applied in
that step (IDLE if none). So a step with 2+ actions does not follow the agent's rule (exactly one
action per step, applied on the step's first frame). Every applied action is logged so this can
be measured.

OUR EDITION (collect_highway_demos_ui.py) instead holds the latest press and applies it at the
start of the next step, so it always follows the agent's rule.

Keys:  SPACE start an episode
       Up = lane left (up the screen)   Down = lane right   Right = faster   Left = slower
       after an episode (as in the LunarLander human_demo.py):
         F = save AND FLAG    SPACE = save (keep, not flagged)    X = discard
       Q quit at any time (an unfinished / undecided episode is not saved)
--feedback (what you are shown; the files always record everything, as in LunarLander):
       reward  = running return + final return (default)    outcome = only CRASHED / TIME UP at the end
       both    = return and outcome                          none    = neither
Saves, in --out:
  epNNN.npz            per saved episode: obs, acts, rews, terminal (the demo), seed, flagged,
                       n_applied (actions per step; agent: always 1), applied_log (step, frame, action)
  flags.json           which saved episodes are flagged (the flagged subset is what GAIL trains on)
  all_episodes_log.csv every attempt incl. discarded: status, duration, return, steps, outcome, steps with 2+ actions
  meta.json            env, config, versions, input rule
At quit it prints how many saved (and flagged) episodes broke the one-action-per-step rule.
Versions must match the cluster's GAIL env: gymnasium 0.29.1, highway-env 1.9.1.
"""
import argparse, glob, json, os, time
from importlib.metadata import version

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  registers highway-v0
import pygame
from highway_env.envs.common.graphics import EventHandler

p = argparse.ArgumentParser()
p.add_argument("--env", default="highway-v0")
p.add_argument("--policy-frequency", type=int, default=1, help="decisions/s (env default 1); must match training")
p.add_argument("--out", default="highway_demos")
p.add_argument("--feedback", choices=["reward", "outcome", "both", "none"], default="reward",
               help="what you are shown during/after an episode; the files always record everything")
p.add_argument("--flag-target", type=int, default=50, help="how many flagged demos you are aiming for")
p.add_argument("--zoom", type=float, default=2.0, help="display size only; does not change the game")
a = p.parse_args()
SHOW_RETURN = a.feedback in ("reward", "both")
SHOW_OUTCOME = a.feedback in ("outcome", "both")

config = {"policy_frequency": a.policy_frequency}  # the only game setting we pass
display = {"manual_control": True,       # highway's default manual control: keys drive, step()'s action is ignored
           "offscreen_rendering": True}  # highway only draws; this script shows the frames and hands it the keys
env = gym.make(a.env, render_mode="rgb_array")
u = env.unwrapped
u.configure({**config, **display})
cfg = u.config
FPS = cfg["simulation_frequency"]                                         # 15 frames per second of game time
FRAMES_PER_STEP = cfg["simulation_frequency"] // cfg["policy_frequency"]  # 15 at the default
MAX_STEPS = int(cfg["duration"] * cfg["policy_frequency"])                # 40 at the default

ACT = u.action_type.actions_indexes  # {"LANE_LEFT": 0, "IDLE": 1, "LANE_RIGHT": 2, "FASTER": 3, "SLOWER": 4}
NAME = {i: n for n, i in ACT.items()}
ARROWS = (pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT)

os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump({"env": a.env, "config": config, "display": display, "edition": "default",
               "input_rule": "highway default manual control: keys act immediately (mid-step); "
                             "recorded action = last one applied in the step",
               "feedback": a.feedback, "flag_target": a.flag_target,
               "gymnasium": version("gymnasium"), "highway_env": version("highway-env")}, f, indent=2)


def saved_files():
    return sorted(glob.glob(os.path.join(a.out, "ep*.npz")))


def is_flagged(path):
    d = np.load(path)
    return bool(d["flagged"]) if "flagged" in d else False


n_saved = len(saved_files())
n_flagged = sum(is_flagged(f) for f in saved_files())

# ---- window: the game inside a box, a panel underneath ----
W, H = int(cfg["screen_width"] * a.zoom), int(cfg["screen_height"] * a.zoom)
PAD, PANEL = 20, 190
pygame.init()
screen = pygame.display.set_mode((W + 2 * PAD, H + 2 * PAD + PANEL))
font, small = pygame.font.Font(None, 32), pygame.font.Font(None, 24)
clock = pygame.time.Clock()

ui = {"mode": "waiting", "frame": None, "msg": "Press SPACE to start", "step": 0, "f": 0,
      "ret": 0.0, "now": "IDLE", "speed": 0.0, "multi": 0, "outcome": "", "decision": None, "quit": False}
applied = []      # actions that acted on the car in the current step
applied_log = []  # (step, frame, action) for every action that acted on the car this episode


def hook_actions():
    """Log every action highway applies to the car (reset() builds a new action_type, so re-hook)."""
    at = u.action_type
    act = at.act

    def logged_act(action):
        act(action)
        applied.append(int(action))
        applied_log.append((ui["step"], ui["f"], int(action)))
        ui["now"] = NAME[int(action)]
    at.act = logged_act


def handle_events():
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_q):
            ui["quit"] = True
        elif e.type != pygame.KEYDOWN:
            continue
        elif ui["mode"] == "waiting" and e.key == pygame.K_SPACE:
            ui["mode"] = "start"
        elif ui["mode"] == "running" and e.key in ARROWS:
            EventHandler.handle_event(u.action_type, e)  # highway's own default handling: acts immediately
        elif ui["mode"] == "deciding" and e.key in (pygame.K_f, pygame.K_SPACE, pygame.K_x):
            ui["decision"] = {pygame.K_f: "flagged", pygame.K_SPACE: "saved", pygame.K_x: "discarded"}[e.key]


def text(s, x, y, f=font, color=(225, 225, 225)):
    screen.blit(f.render(s, True, color), (x, y))


OUTCOME_COLOR = {"CRASHED": (255, 90, 90), "TIME UP": (120, 220, 120)}


def draw():
    screen.fill((25, 25, 25))
    if ui["frame"] is not None:
        img = pygame.surfarray.make_surface(ui["frame"].swapaxes(0, 1))
        screen.blit(pygame.transform.smoothscale(img, (W, H)), (PAD, PAD))
    pygame.draw.rect(screen, (140, 140, 140), (PAD - 3, PAD - 3, W + 6, H + 6), 3)  # the box
    y = H + 2 * PAD
    counts = f"Demos saved: {n_saved}    flagged (F): {n_flagged} / {a.flag_target}"
    text(counts, PAD + W // 2, y, color=(255, 255, 0) if n_flagged >= a.flag_target else (120, 220, 120))
    if ui["mode"] == "running":
        head = f"Episode {n_saved}    step {ui['step'] + 1} / {MAX_STEPS}"
        if SHOW_RETURN:
            head += f"    return {ui['ret']:.1f}"
        text(head, PAD, y)
        text(f"Action: {ui['now']}", PAD, y + 40, color=(120, 200, 255))
        text(f"Speed: {ui['speed']:.1f} m/s", PAD + 260, y + 40, color=(120, 200, 255))
        n = len(applied)
        text(f"Actions this step: {n}    steps with 2+: {ui['multi']}", PAD + W // 2, y + 40,
             color=(255, 90, 90) if n > 1 else (200, 200, 200))
        bar = W // 2 - PAD                                              # countdown to the next step
        left = max(0.0, 1 - ui["f"] / FRAMES_PER_STEP)
        pygame.draw.rect(screen, (70, 70, 70), (PAD + W // 2, y + 80, bar, 12))
        pygame.draw.rect(screen, (255, 210, 90), (PAD + W // 2, y + 80, int(bar * left), 12))
    elif ui["mode"] == "deciding":
        x = PAD
        if SHOW_OUTCOME:
            text(ui["outcome"], x, y, color=OUTCOME_COLOR.get(ui["outcome"], (255, 210, 90)))
            x += 180
        if SHOW_RETURN:
            text(f"return {ui['ret']:.1f}", x, y)
        text("F = save & FLAG     SPACE = save (keep)     X = discard     Q = quit", PAD, y + 50,
             color=(255, 210, 90))
    else:
        text(ui["msg"], PAD, y + 50, color=(255, 210, 90))
    text("SPACE start    Up lane left    Down lane right    Right faster    Left slower    Q quit",
         PAD, y + 150, small, (150, 150, 150))
    pygame.display.flip()


# highway renders every in-between physics frame through env.render(); show each one in the box
_render = u.render


def render_and_show():
    img = _render()
    ui["frame"] = img
    if ui["mode"] == "running":
        ui["f"] += 1
    if u.vehicle is not None:
        ui["speed"] = float(u.vehicle.speed)
    handle_events()
    draw()
    clock.tick(FPS)  # real time: 15 frames per second
    return img


u.render = render_and_show


def log_attempt(status, duration, ret, steps, outcome, multi, file=""):
    path = os.path.join(a.out, "all_episodes_log.csv")
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("time,file,status,duration_sec,return,steps,outcome,steps_with_2plus_actions\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{file},{status},{duration:.2f},{ret:.3f},"
                f"{steps},{outcome},{multi}\n")


def write_flags():
    files = saved_files()
    flagged = [os.path.basename(f) for f in files if is_flagged(f)]
    with open(os.path.join(a.out, "flags.json"), "w") as f:
        json.dump({"flagged_files": flagged, "n_total": len(files), "n_flagged": len(flagged),
                   "flag_target": a.flag_target}, f, indent=2)


def rule_check():
    """How many saved episodes in the output folder broke the one-action-per-step rule."""
    rows, old = [], 0
    for f in saved_files():
        d = np.load(f)
        if "n_applied" not in d:
            old += 1                     # saved by an earlier version, no counters
            continue
        n = d["n_applied"]
        rows.append((os.path.basename(f), int((n > 1).sum()), len(n), is_flagged(f)))
    for label, sub in (("all saved", rows), ("flagged", [r for r in rows if r[3]])):
        bad = [r for r in sub if r[1] > 0]
        print(f"Rule check, {label} ({a.out}): {len(bad)} of {len(sub)} episodes have at least one step "
              f"with 2+ actions; {sum(r[1] for r in sub)} of {sum(r[2] for r in sub)} steps in total")
    for name, k, t, flg in rows:
        if k:
            print(f"  {name}{' (flagged)' if flg else ''}: {k} of {t} steps")
    if old:
        print(f"  ({old} older episodes without counters skipped)")


rng = np.random.default_rng()
try:
    while not ui["quit"]:
        seed = int(rng.integers(2**31 - 1))
        obs, info = env.reset(seed=seed)
        hook_actions()
        applied.clear()
        applied_log.clear()
        ui.update(mode="waiting", step=0, f=0, ret=0.0, now="IDLE", multi=0, outcome="", decision=None)
        u.render()                       # show the starting road (also re-arms highway's in-between frames)
        pygame.display.set_caption("highway-env demo collection (default edition)")
        while ui["mode"] == "waiting" and not ui["quit"]:
            handle_events()
            draw()
            clock.tick(30)
        if ui["quit"]:
            break

        ui["mode"] = "running"
        t0 = time.time()
        idle = ACT["IDLE"]
        ep_obs, ep_acts, ep_rews, n_applied = [obs], [], [], []
        done = False
        while not done and not ui["quit"]:
            applied.clear()
            ui["f"] = 0
            obs, rew, terminated, truncated, info = env.step(idle)  # ignored under manual_control
            u.render()                   # last frame of the step (+ re-arms the in-between frames)
            ep_acts.append(applied[-1] if applied else idle)        # recorded: last action of the step
            n_applied.append(len(applied))
            ui["multi"] += len(applied) > 1
            ep_obs.append(obs)
            ep_rews.append(rew)
            ui["step"] += 1
            ui["ret"] += rew
            done = terminated or truncated
        duration = time.time() - t0
        ret = float(sum(ep_rews))
        outcome = "CRASHED" if info.get("crashed") else "TIME UP" if truncated else "ENDED"
        if not done:
            log_attempt("quit_during", duration, ret, len(ep_acts), "", ui["multi"])
            print("quit mid-episode: that episode was not saved")
            break

        # decide, as in LunarLander: F = save & flag, SPACE = save, X = discard (console always gets everything)
        print(f"episode done: {len(ep_acts)} steps, return {ret:.1f}, {outcome}, steps with 2+ actions: "
              f"{ui['multi']}  -- F = save & flag / SPACE = save / X = discard / Q = quit")
        ui.update(mode="deciding", outcome=outcome, decision=None)
        while ui["decision"] is None and not ui["quit"]:
            handle_events()
            draw()
            clock.tick(30)
        if ui["decision"] is None:       # quit before deciding
            log_attempt("quit_undecided", duration, ret, len(ep_acts), outcome, ui["multi"])
            print("quit before deciding: that episode was not saved")
            break

        status = ui["decision"]
        if status == "discarded":
            log_attempt(status, duration, ret, len(ep_acts), outcome, ui["multi"])
            print("  -> discarded")
            ui["msg"] = "Discarded.  Press SPACE to start the next episode"
            continue
        path = os.path.join(a.out, f"ep{n_saved:03d}.npz")
        np.savez(path, obs=np.array(ep_obs, np.float32), acts=np.array(ep_acts, np.int64),
                 rews=np.array(ep_rews, np.float64), terminal=terminated, seed=seed,
                 flagged=(status == "flagged"), n_applied=np.array(n_applied, np.int64),
                 applied_log=np.array(applied_log, np.int64).reshape(-1, 3))
        n_saved += 1
        n_flagged += status == "flagged"
        log_attempt(status, duration, ret, len(ep_acts), outcome, ui["multi"], os.path.basename(path))
        write_flags()
        print(f"  -> {status.upper()}: {path}  (saved {n_saved}, flagged {n_flagged}/{a.flag_target})")
        reached = "  TARGET REACHED." if n_flagged >= a.flag_target else ""
        ui["msg"] = f"{status.capitalize()}.{reached}  Press SPACE to start the next episode"
except KeyboardInterrupt:
    print("\nstopped; the unfinished episode was not saved")
env.close()
pygame.quit()
write_flags()
rule_check()
