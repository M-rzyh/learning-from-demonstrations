#!/usr/bin/env python3
"""
Human demonstration collection for GAIL (IRL3/imitation).

Saves demos as imitation.data.types.TrajectoryWithRew objects via
imitation.data.serialize.save(), directly loadable by GAIL training scripts.

Usage:
    # Web mode (default) — browser UI, works over SSH tunnel, no X11 needed
    python human_demo.py --env LunarLander-v2 --mode web --port 8080

    # Visual mode — renders the game via pygame, needs X11/display
    python human_demo.py --env LunarLander-v2 --mode visual

    # Curses mode — text-only, works over SSH without X11
    python human_demo.py --env LunarLander-v2 --mode curses

    # Text mode — step-by-step prompts
    python human_demo.py --env LunarLander-v2 --mode text

Controls (all modes):
    W / Up    : main engine ON  (discrete: action 2)
    A / Left  : thrust left     (discrete: action 1)
    D / Right : thrust right    (discrete: action 3)
    S / Down  : no-op / coast   (discrete: action 0)
    SPACE     : end current episode (save it)
    X         : discard current episode
    Q         : save all & quit
"""

import collections
import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Action presets for discrete LunarLander-v2
# 0=noop, 1=fire left, 2=fire main, 3=fire right
# ---------------------------------------------------------------------------
PRESETS_DISCRETE = {
    'w': 2,   # main engine
    'a': 1,   # left engine
    'd': 3,   # right engine
    's': 0,   # noop
    'none': 0,
}

# Action presets for continuous envs
PRESETS_CONTINUOUS = {
    'w':     np.array([ 1.0,  0.0], dtype=np.float32),
    'a':     np.array([ 0.0, -1.0], dtype=np.float32),
    'd':     np.array([ 0.0,  1.0], dtype=np.float32),
    's':     np.array([ 0.0,  0.0], dtype=np.float32),
    'wa':    np.array([ 1.0, -1.0], dtype=np.float32),
    'wd':    np.array([ 1.0,  1.0], dtype=np.float32),
    'none':  np.array([ 0.0,  0.0], dtype=np.float32),
}


def _state_string(obs, step, ep_reward, act=None):
    """One-line state summary for LunarLander (obs_dim=8)."""
    lines = []
    if len(obs) == 8:
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+8.2f}")
        lines.append(f"  Pos  x={obs[0]:+.3f}  y={obs[1]:+.3f}")
        lines.append(f"  Vel  vx={obs[2]:+.3f}  vy={obs[3]:+.3f}")
        lines.append(f"  Angle {obs[4]:+.3f} rad ({np.degrees(obs[4]):+.1f} deg)   angvel={obs[5]:+.3f}")
        l, r = 'Y' if obs[6] > 0.5 else 'N', 'Y' if obs[7] > 0.5 else 'N'
        lines.append(f"  Legs  L={l}  R={r}")
        if act is not None:
            if isinstance(act, (int, np.integer)):
                names = {0: 'noop', 1: 'left', 2: 'main', 3: 'right'}
                lines.append(f"  Last action: {act} ({names.get(int(act), '?')})")
            else:
                lines.append(f"  Last action: main={act[0]:+.2f}  lat={act[1]:+.2f}")
    else:
        n = min(8, len(obs))
        vals = ' '.join(f'{v:+.3f}' for v in obs[:n])
        lines.append(f"  Step {step:4d}  |  Ep reward: {ep_reward:+.2f}")
        lines.append(f"  Obs[:{n}]: {vals}{'...' if len(obs)>n else ''}")
        if act is not None:
            lines.append(f"  Action: {act}")
    return '\n'.join(lines)


def _is_discrete(env):
    """Check if action space is discrete."""
    import gymnasium
    return isinstance(env.action_space, gymnasium.spaces.Discrete)


# ---------------------------------------------------------------------------
# Web mode — browser-based UI, streams frames over HTTP
# ---------------------------------------------------------------------------

_WEB_HTML = """<!DOCTYPE html>
<html>
<head>
<title>Human Demo Collection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #111; color: #ddd; font-family: monospace;
         display: flex; flex-direction: column; align-items: center;
         padding: 15px; min-height: 100vh; }
  h2 { margin-bottom: 8px; color: #8cf; }
  #status { font-size: 16px; margin: 8px 0; min-height: 22px; color: #ff0; }
  #flagcount { font-size: 24px; font-weight: bold; color: #4f4; margin: 4px 0; }
  #frame { border: 2px solid #444; background: #000; display: block; }
  #controls { color: #888; margin: 10px 0; font-size: 13px; }
  .k { background: #282828; padding: 2px 7px; border-radius: 3px;
       border: 1px solid #555; color: #ccc; font-weight: bold; }
  #keys { color: #555; margin-top: 4px; font-size: 12px; }
  #alert { color: #f44; font-size: 14px; margin: 6px 0; min-height: 18px; }
</style>
</head>
<body>
  <h2>LunarLander Demo Collection</h2>
  <div id="alert">Click here first to capture keyboard, then press SPACE to start!</div>
  <div id="status">Connecting...</div>
  <!-- Verdict gets its own line so it can be colour-coded and read at a glance; it is blank
       during flight and only fills in when the episode ends. -->
  <div id="outcome" style="font-size:1.6em; font-weight:bold; min-height:1.3em;"></div>
  <div id="flagcount">FLAGGED: 0 / 0</div>
  <canvas id="frame" width="600" height="400" style="width:600px; height:400px;"></canvas>
  <div id="controls">
    <span class="k">W</span>/<span class="k">&uarr;</span> up &nbsp;
    <span class="k">A</span>/<span class="k">&larr;</span> left &nbsp;
    <span class="k">D</span>/<span class="k">&rarr;</span> right &nbsp;
    <span class="k">S</span>/<span class="k">&darr;</span> coast
    &nbsp;|&nbsp;
    <span class="k">SPACE</span> start / keep &nbsp;
    <span class="k">F</span> include (flag) &nbsp;
    <span class="k">X</span> discard &nbsp;
    <span class="k">Q</span> finish
  </div>
  <div id="keys">Keys: none</div>

<script>
const canvas = document.getElementById('frame');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const outcomeEl = document.getElementById('outcome');
const flagEl = document.getElementById('flagcount');
const keysEl = document.getElementById('keys');
const alertEl = document.getElementById('alert');

let hasFocus = false;
window.addEventListener('focus', () => { hasFocus = true; alertEl.textContent = ''; });
window.addEventListener('blur', () => { hasFocus = false; alertEl.textContent = 'Click here to capture keyboard!'; keys.clear(); });
document.addEventListener('click', () => { window.focus(); });

// --- keyboard tracking ---
let keys = new Set();
document.addEventListener('keydown', e => {
  e.preventDefault();
  if (['Space','KeyX','KeyQ','Escape','KeyF'].includes(e.code)) {
    fetch('/special', {method:'POST', body:e.code});
    return;
  }
  keys.add(e.code);
});
document.addEventListener('keyup', e => { e.preventDefault(); keys.delete(e.code); });

// Send held keys at ~30 Hz
setInterval(() => {
  fetch('/keys', {method:'POST', headers:{'Content-Type':'application/json'},
                  body:JSON.stringify([...keys])});
  keysEl.textContent = 'Keys: ' + (keys.size ? [...keys].join(', ') : 'none');
}, 33);

// --- frame polling ---
let busy = false;
setInterval(() => {
  if (busy) return;
  busy = true;
  fetch('/frame?t=' + Date.now())
    .then(r => { if (!r.ok) throw 0; return r.blob(); })
    .then(blob => createImageBitmap(blob))
    .then(bmp => {
      ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
      busy = false;
    })
    .catch(() => { busy = false; });
}, 33);

// --- status polling ---
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d => {
    statusEl.textContent =
      'Ep ' + d.ep + '/' + d.max_ep +
      '  |  Step ' + d.step +
      (d.show_reward ? '  |  Reward: ' + d.reward.toFixed(2) : '') +
      '  |  Saved: ' + d.saved +
      '  |  ' + d.msg;
    // green = landed, red = crashed, amber = ran out of time. Blank during flight.
    outcomeEl.textContent = d.outcome || '';
    outcomeEl.style.color = d.outcome === 'LANDED'  ? '#3f3'
                          : d.outcome === 'CRASHED' ? '#f44'
                          : '#fb0';
    flagEl.textContent = 'FLAGGED (F): ' + d.flagged + ' / ' + d.flag_target;
    flagEl.style.color = (d.flagged >= d.flag_target) ? '#ff0' : '#4f4';
  }).catch(()=>{});
}, 200);
</script>
</body>
</html>
"""


def _make_env(env_id, render_mode, gravity=None, enable_wind=False,
              wind_power=15.0, turbulence_power=1.5):
    """Build the play env, applying TASK/DYNAMICS options (gravity, wind).

    These differ in kind from the perception/control difficulties (blank, region,
    sticky, delay): those corrupt the frame or the keypress AROUND a normal env,
    whereas gravity and wind change the PHYSICS the env is built with, so they are
    applied here at construction (and they also change the eval task).

      - Wind (gymnasium >= 0.24 only): constructor kwargs enable_wind / wind_power /
        turbulence_power. Off by default -> any env still constructs unchanged.
      - Gravity: written to world.gravity AFTER construction, NOT via the kwarg,
        because the gymnasium constructor asserts -12 < gravity < 0 (so gravity=0,
        inverted, or crushing gravity are rejected there). The write persists across
        reset() and bypasses the assert; safe here because human_demo runs a single
        live env (no pickling / vecenv that would rebuild it from the stored kwargs).
    """
    import gymnasium as gym
    kw = dict(render_mode=render_mode)
    if enable_wind:
        kw.update(enable_wind=True, wind_power=float(wind_power),
                  turbulence_power=float(turbulence_power))
    env = gym.make(env_id, **kw)
    if gravity is not None:
        env.unwrapped.world.gravity = (0.0, float(gravity))
    return env


def _outcome_label(ep_rew, terminated, truncated):
    """How the episode ENDED, in words, with no number attached.

    LunarLander pays exactly +100 on a successful landing and -100 on a crash (or on flying
    off the side of the screen), so the final step's reward is a clean verdict. A run that
    hits the time limit ends on neither.
    """
    if not ep_rew:
        return "NO DATA"
    last = float(ep_rew[-1])
    if terminated and last >= 99:
        return "LANDED"
    if terminated and last <= -99:
        return "CRASHED"
    if truncated:
        return "OUT OF TIME"
    return "ENDED"


def _run_web(env_id, max_episodes, fps, discrete, seed, port,
             blank_mode="none", blank_prob=0.5, blank_k=2, blank_seed=0, block_len=10,
             blank_style="black", feedback="reward",
             region_frac=0.0, region_seed=0, region_outline=None, outline_thickness=3,
             sticky_p=0.0, sticky_seed=0, delay_k=0,
             gravity=None, enable_wind=False, wind_power=15.0, turbulence_power=1.5,
             flag_target=50, auto_quit=False):
    """Run demo collection with a browser-based UI.

    blank_mode != 'none' blanks displayed frames (frame-blanking difficulty): the
    human plays partly blind, producing degraded demonstrations. Recorded obs stay
    the true 8-D state; only the human's view (and hence actions) are affected.

    blank_style picks WHAT a blanked frame shows, independently of blank_mode's schedule:
      'black'  — the whole frame goes dark (original behaviour, the default)
      'vanish' — only the lander is removed and replaced by the terrain behind it, so the
                 pilot keeps the ground, pad and flags but loses the craft. This is the
                 spatially selective variant; see lander_vanish.py.
    """
    import gymnasium as gym
    import threading
    import json
    import io
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from socketserver import ThreadingMixIn
    from PIL import Image

    env = _make_env(env_id, 'rgb_array', gravity=gravity, enable_wind=enable_wind,
                    wind_power=wind_power, turbulence_power=turbulence_power)

    # --- frame-blanking (difficulty): blank displayed frames so the human plays blind ---
    _blank_rng = np.random.default_rng(blank_seed)
    _disp = {"n": 0, "block_on": False}

    # Imported only when needed: it pulls in scipy, which the keyboard/none paths don't want
    # to pay for. Costs ~13 ms/frame, comfortably inside the 50 ms budget at fps 20.
    _plate = None
    if blank_style == "vanish":
        from lander_vanish import BackgroundPlate
        _plate = BackgroundPlate()

    def _hide(fr):
        """Render a blanked frame in the configured style."""
        return _plate.apply(fr) if _plate is not None else np.zeros_like(fr)

    def _reset_plate():
        """Terrain is regenerated on reset, so a stale plate would paste the PREVIOUS
        episode's ground into this one."""
        if _plate is not None:
            _plate.reset()

    def _maybe_blank(fr):
        if blank_mode == "none":
            return fr
        if blank_mode == "deterministic":
            b = (_disp["n"] % int(blank_k)) == 0
        elif blank_mode == "block":
            # sustained blackouts: decide once per block of block_len displayed frames
            # (contiguous, so it doesn't blend into flicker like per-frame stochastic).
            if _disp["n"] % int(block_len) == 0:
                _disp["block_on"] = _blank_rng.random() < float(blank_prob)
            b = _disp["block_on"]
        else:  # stochastic (independent per displayed frame)
            b = _blank_rng.random() < float(blank_prob)
        _disp["n"] += 1
        if b:
            return _hide(fr)
        # not blanked: still let the plate learn the background from this frame
        if _plate is not None:
            _plate.observe(fr)
        return fr

    # --- region masking (difficulty, Sweep 1): one fixed black rectangle over the WHOLE
    # episode. Area = region_frac*frame (sides scaled by sqrt), location uniform-random per
    # episode (seeded from region_seed -> reproducible, same box every run). The human plays
    # with a persistent spatial blind spot; recorded obs stay the true 8-D state. ---
    _region_rng = np.random.default_rng(region_seed)
    _region = {"box": None}

    def _new_region_box(fr):
        if region_frac <= 0.0:
            _region["box"] = None
            return
        H, W = fr.shape[:2]
        s = float(region_frac) ** 0.5
        w = max(1, min(W, int(round(s * W))))
        h = max(1, min(H, int(round(s * H))))
        x0 = int(_region_rng.integers(0, W - w + 1))
        y0 = int(_region_rng.integers(0, H - h + 1))
        _region["box"] = (x0, y0, w, h)

    def _maybe_mask(fr):
        box = _region["box"]
        if box is None:
            return fr
        x0, y0, w, h = box
        fr = fr.copy()
        fr[y0:y0 + h, x0:x0 + w] = 0
        if region_outline:
            # Border drawn INSIDE the box, on top of the black fill, so the mask is
            # distinguishable from the black sky. Occluded pixels are unchanged (the
            # ring is opaque too, just not black). Cosmetic only.
            t = max(1, int(outline_thickness))
            fr[y0:y0 + t,         x0:x0 + w] = region_outline      # top
            fr[y0 + h - t:y0 + h, x0:x0 + w] = region_outline      # bottom
            fr[y0:y0 + h, x0:x0 + t]         = region_outline      # left
            fr[y0:y0 + h, x0 + w - t:x0 + w] = region_outline      # right
        return fr

    # Control-interface difficulty (sticky / delay). Reset at every episode boundary so state
    # never leaks across episodes. Recorded actions are the EFFECTIVE ones the env ran.
    def _noop_action():
        return 0 if discrete else np.zeros(env.action_space.shape[0], dtype=np.float32)

    _apply_control, _reset_control = make_control_corruptor(
        sticky_p, sticky_seed, delay_k, _noop_action)

    # Shared state between HTTP server and game loop
    lock = threading.Lock()
    shared = {
        'frame_bytes': b'',
        'pressed_keys': set(),
        'special': None,
        'ep': 0, 'max_ep': max_episodes,
        'step': 0, 'reward': 0.0,
        # What the pilot is told. 'reward' is the default = previous behaviour, unchanged.
        # 'outcome' hides the number and reports only LANDED / CRASHED at the end, so the
        # human judges the flight rather than chasing a score they can read off the screen.
        'show_reward': feedback in ("reward", "both"),
        'outcome': '',      # LANDED / CRASHED / OUT OF TIME, set only at episode end
        'saved': 0, 'msg': 'Starting...',
        'flagged': 0, 'flag_target': flag_target,
    }

    def encode_frame(frame):
        # full resolution + higher quality (600x400 JPEG encodes in a few ms)
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=88)
        return buf.getvalue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == '/':
                self._respond(200, 'text/html', _WEB_HTML.encode())
            elif self.path.startswith('/frame'):
                with lock:
                    data = shared['frame_bytes']
                if data:
                    self._respond(200, 'image/jpeg', data,
                                  extra={'Cache-Control': 'no-store'})
                else:
                    self.send_response(204)
                    self.send_header('Content-Length', '0')
                    self.end_headers()
            elif self.path == '/status':
                with lock:
                    info = {k: shared[k] for k in
                            ('ep','max_ep','step','reward','saved','flagged','flag_target',
                             'msg','show_reward','outcome')}
                self._respond(200, 'application/json',
                              json.dumps(info).encode())
            else:
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()

        def do_POST(self):
            cl = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(cl) if cl else b''
            if self.path == '/keys':
                data = json.loads(body) if body else []
                with lock:
                    shared['pressed_keys'] = set(data)
            elif self.path == '/special':
                with lock:
                    shared['special'] = body.decode().strip() if body else None
            self.send_response(200)
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _respond(self, code, ctype, data, extra=None):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            if extra:
                for k, v in extra.items():
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass  # suppress request logs

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedServer(('0.0.0.0', port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    print(f"\n{'='*60}")
    print(f"  Web UI running on port {port}")
    print(f"  If on a remote cluster, open a tunnel first:")
    print(f"    ssh -L {port}:localhost:{port} <cluster>")
    print(f"  Then open: http://localhost:{port}")
    print(f"{'='*60}\n")

    # ---- game loop (runs in main thread) ----
    episodes = []
    ep_times = []
    all_attempts = []  # log ALL episodes (saved + discarded) with timing
    flags = []         # aligned with `episodes`: True = FLAGGED (to-use this session)
    n_flagged = 0      # count of flagged episodes; target = flag_target

    ep = 0
    ep_label = str(max_episodes) if max_episodes > 0 else '∞'
    while max_episodes <= 0 or ep < max_episodes:   # max_episodes<=0 => unlimited, quit with Q
        obs, _ = env.reset(seed=seed + ep)
        frame = env.render()
        _new_region_box(frame)          # one fixed mask box for this whole episode
        _reset_control()                # sticky/delay state does not leak across episodes
        _reset_plate()                  # terrain is regenerated: never reuse the old plate
        frame = _maybe_mask(frame)
        with lock:
            shared['frame_bytes'] = encode_frame(frame)
            shared['ep'] = ep + 1
            shared['step'] = 0
            shared['reward'] = 0.0
            shared['outcome'] = ''      # clear last episode's verdict before the next flight
            shared['msg'] = 'Press SPACE to start episode'
            shared['special'] = None

        # --- Wait for SPACE before starting the episode ---
        print(f"\n  Episode {ep+1}/{ep_label}  |  Waiting for SPACE to start...")
        waiting = True
        while waiting:
            with lock:
                special = shared['special']
                shared['special'] = None
            if special == 'Space':
                waiting = False
            elif special in ('KeyQ', 'Escape'):
                env.close()
                server.shutdown()
                return episodes, ep_times, all_attempts, flags
            time.sleep(0.05)

        ep_start = time.time()
        with lock:
            shared['msg'] = 'Playing — use W/A/D/S'

        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        while not done:
            t_step = time.perf_counter()   # frame deadline starts HERE, before any work
            with lock:
                shared['special'] = None    # ignore special keys mid-episode
                keys = set(shared['pressed_keys'])
            # No mid-episode controls: the episode runs until it terminates on its own
            # (land / crash / truncate); the flag-or-keep decision happens afterwards.

            # Map held keys to action (no key = action 0 = coast/do nothing)
            if discrete:
                if 'KeyW' in keys or 'ArrowUp' in keys:
                    action = 2
                elif 'KeyA' in keys or 'ArrowLeft' in keys:
                    action = 1
                elif 'KeyD' in keys or 'ArrowRight' in keys:
                    action = 3
                else:
                    action = 0
            else:
                act_dim = env.action_space.shape[0]
                action = np.zeros(act_dim, dtype=np.float32)
                if 'KeyW' in keys or 'ArrowUp' in keys:
                    action[0] = 1.0
                if 'KeyA' in keys or 'ArrowLeft' in keys:
                    if act_dim > 1:
                        action[1] = -1.0
                if 'KeyD' in keys or 'ArrowRight' in keys:
                    if act_dim > 1:
                        action[1] = 1.0

            # Corrupt the control loop (sticky / delay). No-op when both are off.
            # The human's key becomes `action`; what the lander actually runs is `eff_action`.
            eff_action = _apply_control(action)

            next_obs, reward, terminated, truncated, info = env.step(eff_action)
            # Record the EFFECTIVE action: a demo must be a consistent (s, a, s') sequence,
            # so it stores what the env ran, not the key the human pressed.
            ep_act.append(eff_action if discrete else np.asarray(eff_action).copy())
            ep_rew.append(reward)
            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            # Render + encode every step (smoother playback)
            frame = _maybe_mask(_maybe_blank(env.render()))
            with lock:
                shared['frame_bytes'] = encode_frame(frame)
                shared['step'] = step
                shared['reward'] = ep_reward

            if terminated or truncated:
                terminal = terminated
                done = True

            # Sleep to a DEADLINE, not a fixed pad: the step+render+encode cost is absorbed
            # into the frame budget instead of added on top. Without this, achieved rate is
            # `1/(1/fps + overhead)` and the overhead depends on what is on screen — a blanked
            # frame is ~4 KB of JPEG vs ~196 KB, so blanking silently sped the game up (the
            # b=10 human blanking confound). Now requested fps == achieved steps/s.
            time.sleep(max(0.0, (1.0 / fps) - (time.perf_counter() - t_step)))

        ep_end = time.time()

        # NOTE: dead branch in web mode — `discard` is initialised False above and never set
        # True here (mid-episode keys are ignored; X is handled in the deciding block below).
        # Kept because _run_visual/_run_curses share the same shape. Do not rely on it.
        if discard:
            all_attempts.append({
                'status': 'discarded_during',
                'duration_sec': ep_end - ep_start,
                'reward': ep_reward,
                'steps': step,
            })
            with lock:
                shared['msg'] = f'Episode {ep+1} discarded'
            print(f"  Episode {ep+1} discarded.")
            time.sleep(1.5)
            continue

        if len(ep_act) > 0:
            # Save-all with flagging: F = save & FLAG (use this session), SPACE = save
            # (kept but not flagged), X = discard. Every non-discarded episode is saved.
            # What the pilot is told about how it went. The reward number and the
            # crash/land verdict are independently switchable (--feedback), because a
            # visible score invites optimising the number instead of flying well, while
            # a bare LANDED/CRASHED is the judgement we actually want them making.
            bits = [f'Episode done: {step} steps']
            if feedback in ("reward", "both"):
                bits.append(f'reward={ep_reward:+.2f}')
            head = ', '.join(bits)
            verdict = _outcome_label(ep_rew, terminal, not terminal)
            with lock:
                # The verdict goes to its own colour-coded element, not into `msg`.
                shared['outcome'] = verdict if feedback in ("outcome", "both") else ''
                shared['msg'] = (f'{head}  |  '
                                 f'F=INCLUDE  SPACE=keep(not incl.)  X=discard  Q=finish  |  '
                                 f'flagged {n_flagged}/{flag_target}')
            # The console always keeps the full record regardless of what the pilot sees —
            # hiding a number from the UI must not hide it from the run log.
            print(f"  Episode {ep+1}: {step} steps, reward={ep_reward:+.2f}, {verdict}  "
                  f"— F=include / SPACE=keep / X=discard / Q=finish?")

            deciding = True
            save_it = False
            flag_it = False
            while deciding:
                with lock:
                    special = shared['special']
                    shared['special'] = None
                if special == 'KeyF':
                    save_it = True; flag_it = True; deciding = False
                elif special == 'Space':
                    save_it = True; deciding = False
                elif special == 'KeyX':
                    deciding = False
                elif special in ('KeyQ', 'Escape'):
                    # Quit — everything already collected is saved by main()
                    env.close()
                    server.shutdown()
                    return episodes, ep_times, all_attempts, flags
                time.sleep(0.05)

            if save_it:
                episodes.append((ep_obs, ep_act, ep_rew, terminal))
                flags.append(flag_it)
                ep_times.append(ep_end - ep_start)
                if flag_it:
                    n_flagged += 1
                all_attempts.append({
                    'status': 'flagged' if flag_it else 'saved',
                    'duration_sec': ep_end - ep_start,
                    'reward': ep_reward,
                    'steps': step,
                })
                hit_target = n_flagged >= flag_target
                reached = ('  — TARGET REACHED, finishing' if hit_target and auto_quit else
                           '  — TARGET REACHED, press Q to finish' if hit_target else '')
                with lock:
                    shared['saved'] = len(episodes)
                    shared['flagged'] = n_flagged
                    shared['msg'] = (f'{"FLAGGED" if flag_it else "Saved"}!  total {len(episodes)}, '
                                     f'flagged {n_flagged}/{flag_target}{reached}')
                print(f"  -> {'FLAGGED' if flag_it else 'Saved'} (total {len(episodes)}, "
                      f"flagged {n_flagged}/{flag_target})")
                if hit_target and auto_quit:
                    print(f"  -> auto-quit: {n_flagged}/{flag_target} flagged.")
                    break
            else:
                all_attempts.append({
                    'status': 'discarded_after',
                    'duration_sec': ep_end - ep_start,
                    'reward': ep_reward,
                    'steps': step,
                })
                with lock:
                    shared['msg'] = 'Discarded — try again'
                print(f"  -> Discarded")
                time.sleep(1.0)
                continue  # don't increment ep, retry

        time.sleep(1.5)
        ep += 1

    with lock:
        shared['msg'] = 'Done — saving...'
    env.close()
    server.shutdown()
    return episodes, ep_times, all_attempts, flags


# ---------------------------------------------------------------------------
# Visual mode — pygame rendering + keyboard input (see the actual game)
# ---------------------------------------------------------------------------

def _blit_frame(screen, frame):
    """Blit an rgb_array frame (H,W,3 uint8) onto a pygame surface."""
    import pygame
    # frame is (H, W, 3) — pygame wants (W, H, 3) for surfarray
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))


def _draw_overlay(screen, text_lines):
    """Draw text overlay on the pygame surface."""
    import pygame
    font = pygame.font.SysFont('monospace', 22, bold=True)
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    screen.blit(overlay, (0, 0))
    y = screen.get_height() // 2 - len(text_lines) * 15
    for line in text_lines:
        text_surf = font.render(line, True, (255, 255, 100))
        x = (screen.get_width() - text_surf.get_width()) // 2
        screen.blit(text_surf, (x, y))
        y += 32
    pygame.display.flip()


def _wait_for_start(screen, frame, ep_num, max_ep, saved):
    """Show 'Press SPACE to start' overlay and block until SPACE is pressed."""
    import pygame
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return 'quit'
                elif event.key == pygame.K_SPACE:
                    return 'start'
        _blit_frame(screen, frame)
        _draw_overlay(screen, [
            f"Episode {ep_num}/{max_ep}  |  Saved: {saved}",
            "",
            "Press SPACE to start",
            "",
            "W=up  A=left  D=right  S=coast",
            "SPACE=end episode  X=discard  Q=quit",
        ])
        clock.tick(15)


def _run_visual(env_id, max_episodes, fps, discrete, seed):
    """Render the game visually and capture keyboard input via pygame.

    Uses render_mode='rgb_array' and blits frames to a pygame window manually.
    This avoids OpenGL/GLX issues on headless displays (VNC, Xvfb).

    The environment only steps when you are actively pressing a control key
    (W/A/D/S or arrows). If no key is held, the game pauses — the lander
    stays frozen until you act. Each episode starts only after you press SPACE.
    """
    import pygame
    import gymnasium as gym

    # Use rgb_array to avoid GLX/OpenGL issues, then blit to our own pygame window
    env = gym.make(env_id, render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    frame = env.render()  # (H, W, 3) uint8

    # Create pygame window matching frame size
    pygame.init()
    h, w = frame.shape[:2]
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f'Human Demo - {env_id}')

    clock = pygame.time.Clock()
    episodes = []
    ep_times = []

    ep = 0
    while ep < max_episodes:
        obs, _ = env.reset()
        frame = env.render()

        # --- Wait for user to press SPACE before starting ---
        print(f"\n  Episode {ep+1}/{max_episodes}  |  Saved: {len(episodes)}")
        print("  Press SPACE in the game window to start...")
        result = _wait_for_start(screen, frame, ep + 1, max_episodes, len(episodes))
        if result == 'quit':
            env.close()
            pygame.quit()
            return episodes, ep_times

        ep_start = time.time()
        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        print("  GO! Controls: W=up  A=left  D=right  S=coast  SPACE=end  X=discard  Q=quit")

        while not done:
            quit_all = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_all = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        quit_all = True
                    elif event.key == pygame.K_x:
                        discard = True
                    elif event.key == pygame.K_SPACE:
                        done = True

            if quit_all:
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(time.time() - ep_start)
                env.close()
                pygame.quit()
                return episodes, ep_times

            if discard or done:
                break

            # Read held keys — only step the env if a control key is pressed.
            # If no key is held, the game pauses (lander freezes in place).
            keys = pygame.key.get_pressed()
            has_input = (
                keys[pygame.K_w] or keys[pygame.K_UP] or
                keys[pygame.K_a] or keys[pygame.K_LEFT] or
                keys[pygame.K_d] or keys[pygame.K_RIGHT] or
                keys[pygame.K_s] or keys[pygame.K_DOWN]
            )

            if not has_input:
                # No key pressed — just show the last frame, don't step
                _blit_frame(screen, frame)
                pygame.display.flip()
                clock.tick(fps)
                continue

            # Map held keys to action
            if discrete:
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action = 2  # main engine
                elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    action = 1  # left
                elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    action = 3  # right
                else:
                    action = 0  # noop (S key)
            else:
                act_dim = env.action_space.shape[0]
                action = np.zeros(act_dim, dtype=np.float32)
                if keys[pygame.K_w] or keys[pygame.K_UP]:
                    action[0] = 1.0
                if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                    if act_dim > 1:
                        action[1] = -1.0
                if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                    if act_dim > 1:
                        action[1] = 1.0

            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            _blit_frame(screen, frame)
            pygame.display.flip()

            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)

            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            if terminated or truncated:
                terminal = terminated
                done = True

            clock.tick(fps)

        ep_end = time.time()

        if discard:
            print(f"  Episode {ep+1} discarded.")
            continue

        if len(ep_act) > 0:
            episodes.append((ep_obs, ep_act, ep_rew, terminal))
            ep_times.append(ep_end - ep_start)
            print(f"  Episode {ep+1} done: {step} steps, reward={ep_reward:+.2f}")

        ep += 1

    env.close()
    pygame.quit()
    return episodes, ep_times


# ---------------------------------------------------------------------------
# Curses-based real-time demo collection
# ---------------------------------------------------------------------------

def _run_curses(env, max_episodes, fps, discrete):
    import curses

    episodes = []   # list of (obs_list, act_list, rew_list, done_flag)
    ep_times = []

    def _inner(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(1000 / fps))

        ep = 0
        while ep < max_episodes:
            ep_start = time.time()
            obs, _ = env.reset()
            ep_obs, ep_act, ep_rew = [obs.copy()], [], []
            step = 0
            ep_reward = 0.0
            done = False
            last_act = 0 if discrete else np.zeros(env.action_space.shape[0], dtype=np.float32)
            discard = False
            terminal = False

            while not done:
                stdscr.clear()
                stdscr.addstr(0, 0, f"{'=' * 56}")
                mode_str = "DISCRETE" if discrete else "CONTINUOUS"
                stdscr.addstr(1, 0, f"  DEMO [{mode_str}]  |  Ep {ep+1}/{max_episodes}  |  Saved: {len(episodes)}")
                stdscr.addstr(2, 0, f"{'=' * 56}")
                for i, line in enumerate(_state_string(obs, step, ep_reward, last_act).split('\n')):
                    stdscr.addstr(4 + i, 0, line)
                row = 4 + i + 2
                stdscr.addstr(row, 0, "  W=up  A=left  D=right  S=coast")
                stdscr.addstr(row+1, 0, "  SPACE=end episode  X=discard  Q=quit")
                stdscr.refresh()

                key = stdscr.getch()

                quit_all = False

                if key == ord('q') or key == ord('Q'):
                    quit_all = True
                    break
                elif key == ord('x') or key == ord('X'):
                    discard = True
                    break
                elif key == ord(' '):
                    done = True
                    break

                # Map key to action
                if discrete:
                    if key == ord('w') or key == curses.KEY_UP:
                        action = 2
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        action = 1
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        action = 3
                    else:
                        action = 0
                else:
                    act_dim = env.action_space.shape[0]
                    action = np.zeros(act_dim, dtype=np.float32)
                    if key == ord('w') or key == curses.KEY_UP:
                        action[0] = 1.0
                    elif key == ord('a') or key == curses.KEY_LEFT:
                        if act_dim > 1:
                            action[1] = -1.0
                    elif key == ord('d') or key == curses.KEY_RIGHT:
                        if act_dim > 1:
                            action[1] = 1.0

                next_obs, reward, terminated, truncated, info = env.step(action)
                last_act = action if discrete else action.copy()
                ep_act.append(action if discrete else action.copy())
                ep_rew.append(reward)

                obs = next_obs
                ep_obs.append(obs.copy())
                step += 1
                ep_reward += reward

                if terminated or truncated:
                    terminal = terminated
                    done = True

            ep_end = time.time()

            if quit_all:
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(ep_end - ep_start)
                break

            if discard:
                stdscr.clear()
                stdscr.addstr(0, 0, f"  Episode {ep+1} discarded.")
                stdscr.refresh()
                curses.napms(600)
                continue

            if len(ep_act) > 0:
                episodes.append((ep_obs, ep_act, ep_rew, terminal))
                ep_times.append(ep_end - ep_start)

            stdscr.clear()
            stdscr.addstr(0, 0, f"  Episode {ep+1} done: {step} steps, reward={ep_reward:+.2f}")
            stdscr.addstr(1, 0, f"  Total saved: {len(episodes)} episodes")
            stdscr.refresh()
            curses.napms(1200)
            ep += 1

    curses.wrapper(_inner)
    return episodes, ep_times


# ---------------------------------------------------------------------------
# Text-based fallback
# ---------------------------------------------------------------------------

def _run_text(env, max_episodes, discrete):
    episodes = []
    ep_times = []

    for ep in range(max_episodes):
        ep_start = time.time()
        obs, _ = env.reset()
        ep_obs, ep_act, ep_rew = [obs.copy()], [], []
        step = 0
        ep_reward = 0.0
        done = False
        discard = False
        terminal = False

        print(f"\n{'=' * 56}")
        print(f"  Episode {ep+1}/{max_episodes}  |  Saved so far: {len(episodes)}")
        act_type = "DISCRETE (0=noop, 1=left, 2=main, 3=right)" if discrete else "CONTINUOUS"
        print(f"  Action space: {act_type}")
        print(f"{'=' * 56}")

        while not done:
            print(_state_string(obs, step, ep_reward))
            if discrete:
                print("  Enter: w/a/d/s or action index (0-3), space=end, x=discard, q=quit")
            else:
                print("  Enter: w/a/d/s, or 'main,lat' (e.g. 0.5,-1), space=end, x=discard, q=quit")

            try:
                raw = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = 'q'

            if raw == 'q':
                if len(ep_act) > 0 and not discard:
                    episodes.append((ep_obs, ep_act, ep_rew, True))
                    ep_times.append(time.time() - ep_start)
                return episodes, ep_times
            elif raw == 'x':
                discard = True
                print("  Discarded.")
                break
            elif raw in ('', 'space', ' '):
                break

            if discrete:
                if raw in PRESETS_DISCRETE:
                    action = PRESETS_DISCRETE[raw]
                else:
                    try:
                        action = int(raw)
                        if action < 0 or action >= env.action_space.n:
                            print(f"  (out of range, using noop)")
                            action = 0
                    except ValueError:
                        print("  (unrecognized, using noop)")
                        action = 0
            else:
                act_dim = env.action_space.shape[0]
                if raw in PRESETS_CONTINUOUS:
                    action = PRESETS_CONTINUOUS[raw].copy()
                    if act_dim != 2:
                        action = np.zeros(act_dim, dtype=np.float32)
                        action[0] = PRESETS_CONTINUOUS[raw][0]
                else:
                    try:
                        vals = [float(x) for x in raw.replace(' ', ',').split(',')]
                        action = np.zeros(act_dim, dtype=np.float32)
                        for j, v in enumerate(vals[:act_dim]):
                            action[j] = np.clip(v, -1.0, 1.0)
                    except ValueError:
                        print("  (unrecognized, using coast)")
                        action = np.zeros(act_dim, dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_act.append(action if discrete else action.copy())
            ep_rew.append(reward)
            obs = next_obs
            ep_obs.append(obs.copy())
            step += 1
            ep_reward += reward

            if terminated or truncated:
                terminal = terminated
                done = True

        ep_end = time.time()

        if discard:
            continue

        if len(ep_act) > 0:
            episodes.append((ep_obs, ep_act, ep_rew, terminal))
            ep_times.append(ep_end - ep_start)
            print(f"  Saved episode {ep+1}: {step} steps, reward={ep_reward:+.2f}")

    return episodes, ep_times


# ---------------------------------------------------------------------------
# Save as imitation TrajectoryWithRew via serialize.save()
# ---------------------------------------------------------------------------

def save_demos(episodes, ep_times, output_dir, all_attempts=None):
    """Convert collected episodes to TrajectoryWithRew and save via imitation.data.serialize."""
    from imitation.data.types import TrajectoryWithRew
    from imitation.data import serialize

    trajectories = []
    for ep_obs, ep_act, ep_rew, terminal in episodes:
        obs = np.array(ep_obs, dtype=np.float32)        # (T+1, obs_dim)
        if isinstance(ep_act[0], (int, np.integer)):
            acts = np.array(ep_act, dtype=np.int64)      # (T,)
        else:
            acts = np.array(ep_act, dtype=np.float32)     # (T, act_dim)
        rews = np.array(ep_rew, dtype=np.float64)        # (T,)

        traj = TrajectoryWithRew(
            obs=obs,
            acts=acts,
            rews=rews,
            infos=None,
            terminal=terminal,
        )
        trajectories.append(traj)

    os.makedirs(output_dir, exist_ok=True)
    serialize.save(output_dir, trajectories)

    total_steps = sum(len(t) for t in trajectories)
    total_reward = sum(float(t.rews.sum()) for t in trajectories)
    print(f"\nSaved {len(trajectories)} episodes ({total_steps} steps, "
          f"total reward={total_reward:+.1f}) -> {output_dir}")

    # Timing CSV for saved episodes only
    csv_path = os.path.join(output_dir, 'timing.csv')
    cumulative = 0.0
    with open(csv_path, 'w') as f:
        f.write("episode_idx,duration_sec,cumulative_sec,ep_reward,ep_length\n")
        for i, (dt, (_, _, ep_rew, _)) in enumerate(zip(ep_times, episodes)):
            cumulative += dt
            reward = sum(ep_rew)
            length = len(ep_rew)
            f.write(f"{i},{dt:.3f},{cumulative:.3f},{reward:.2f},{length}\n")
    print(f"Saved timing CSV -> {csv_path}")
    print(f"Total demo collection time (saved only): {cumulative:.1f} sec")

    # Full session log: ALL episodes (saved + discarded) with timing
    if all_attempts:
        log_path = os.path.join(output_dir, 'all_episodes_log.csv')
        total_cumulative = 0.0
        with open(log_path, 'w') as f:
            f.write("attempt,status,duration_sec,cumulative_sec,reward,steps\n")
            for i, att in enumerate(all_attempts):
                total_cumulative += att['duration_sec']
                f.write(f"{i},{att['status']},{att['duration_sec']:.3f},"
                        f"{total_cumulative:.3f},{att['reward']:.2f},{att['steps']}\n")
        n_saved = sum(1 for a in all_attempts if a['status'] == 'saved')
        n_discarded = sum(1 for a in all_attempts if a['status'] != 'saved')
        print(f"Saved full session log -> {log_path}")
        print(f"  Total attempts: {len(all_attempts)} ({n_saved} saved, {n_discarded} discarded)")
        print(f"  Total wall-clock time (all attempts): {total_cumulative:.1f} sec")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_control_corruptor(sticky_p, sticky_seed, delay_k, noop_fn):
    """Control-interface difficulties: corrupt what the human's keypress DOES.

    Unlike blanking/region (which hide the view), these leave the view perfect and break
    the control loop. Returns `(apply, reset)`.

      sticky_p : repeat the PREVIOUS action with probability p (Machado et al. 2018).
                 Feels like an unresponsive controller.
      delay_k  : apply each action k STEPS late, via a FIFO primed with `noop_fn()`.
                 The human feels k/fps seconds of lag.

    Order per step is sticky THEN delay: an unresponsive controller repeats the last
    command, and the wire then delays whatever it sent. Both are no-ops at 0.

    Identical semantics are used on the PT side (see the PT repo's example-clip script),
    so a given (p, k) means the same thing in both arms.
    """
    rng = np.random.default_rng(sticky_seed)
    state = {"prev": None, "queue": None}

    def reset():
        state["prev"] = None
        state["queue"] = collections.deque(
            [noop_fn() for _ in range(delay_k)], maxlen=delay_k) if delay_k > 0 else None

    def apply(action):
        if sticky_p > 0.0 and state["prev"] is not None and rng.random() < float(sticky_p):
            action = state["prev"]
        state["prev"] = action
        if delay_k > 0:
            q = state["queue"]
            out = q.popleft()      # the action issued k steps ago
            q.append(action)
            return out
        return action

    reset()
    return apply, reset


_OUTLINE_COLORS = {
    'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
    'yellow': (255, 255, 0), 'cyan': (0, 255, 255), 'magenta': (255, 0, 255),
    'white': (255, 255, 255),
}


def _outline_rgb(name, fail):
    """'red' -> (255,0,0); None -> None. Keeps the CLI colour-name-based like ffmpeg's drawbox."""
    if not name:
        return None
    rgb = _OUTLINE_COLORS.get(str(name).lower())
    if rgb is None:
        fail(f"--region-outline must be one of {sorted(_OUTLINE_COLORS)}, got {name!r}.")
    return rgb


def _session_meta(args, episodes, ep_times, dcfg, physics_fps):
    """Everything needed to know how a demo set was collected — above all, its speed.

    `episodes` are (obs, act, rew, terminal) tuples; one action == one env step, so
    achieved steps/s = total actions / total wall-clock seconds.
    """
    steps = sum(len(a) for _, a, _, _ in episodes)
    secs = sum(ep_times)
    achieved = (steps / secs) if secs > 0 else None
    return {
        "requested_fps": args.fps,
        "achieved_steps_per_sec": round(achieved, 3) if achieved else None,
        "physics_fps": physics_fps,
        "realtime_factor_requested": round(args.fps / physics_fps, 4),
        "realtime_factor_achieved": round(achieved / physics_fps, 4) if achieved else None,
        "difficulty": args.difficulty,
        "difficulty_pct": args.difficulty_pct,
        "blank_mode": dcfg["blank_mode"],
        "blank_prob": dcfg["blank_prob"],
        "blank_style": dcfg["blank_style"],
        "feedback": args.feedback,      # what the pilot could see; affects how they flew
        "block_len": args.block_len,
        "blank_seed": args.blank_seed,
        "region_frac": dcfg["region_frac"],
        "region_seed": args.region_seed,
        "region_outline": args.region_outline,
        "sticky_p": dcfg["sticky_p"],
        "sticky_seed": args.sticky_seed,
        "delay_k": dcfg["delay_k"],
        "gravity": args.gravity,
        "enable_wind": args.enable_wind,
        "wind_power": args.wind_power if args.enable_wind else None,
        "turbulence_power": args.turbulence_power if args.enable_wind else None,
        "env": args.env,
        "env_seed": args.seed,
        "max_episodes": args.max_episodes,
        "flag_target": args.flag_target,
        "auto_quit": args.auto_quit,
        "n_episodes": len(episodes),
        "total_steps": steps,
        "total_wallclock_sec": round(secs, 3),
        "argv": sys.argv,
    }


def _write_session_meta(output_dir, meta):
    import json
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "session_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    rf = meta["realtime_factor_achieved"]
    print(f"  session_meta.json -> {output_dir}  "
          f"(fps {meta['requested_fps']} requested, "
          f"{meta['achieved_steps_per_sec']} achieved"
          + (f" = {rf:.2f}x real time)" if rf else ")"))


def _resolve_difficulty(args, fail):
    """Map --difficulty/--difficulty-pct onto the per-technique _run_web kwargs.

    Returns a dict: blank_mode, blank_prob, blank_style, region_frac, sticky_p, delay_k.

    --difficulty wins when given; otherwise the legacy per-technique flags are used
    verbatim, so every command that worked before still works. `fail` is a callable
    that reports a usage error (argparse's parser.error).

    Two families, deliberately different in kind:
      PERCEPTION  (blank, vanish, region) — hide the view; severity is a percentage.
      CONTROL     (sticky, delay) — corrupt the keypress; sticky is a probability
                  (percentage), delay is a whole number of steps (--delay-k), because
                  "50% of a step" is meaningless.
    """
    cfg = dict(blank_mode=args.blank_mode, blank_prob=args.blank_prob,
               blank_style='black',
               region_frac=args.region_frac, sticky_p=args.sticky_p, delay_k=args.delay_k)

    if args.difficulty == 'none':
        if args.difficulty_pct:
            fail('--difficulty-pct given without --difficulty; nothing to apply it to.')
        return cfg

    # `--delay-k` is the severity knob FOR `--difficulty delay`, so it is not a conflict there.
    others = {'blank_mode': args.blank_mode != 'none', 'region_frac': args.region_frac > 0.0,
              'sticky_p': args.sticky_p > 0.0, 'delay_k': args.delay_k > 0}
    if args.difficulty == 'delay':
        others.pop('delay_k')
    if any(others.values()):
        clash = ', '.join('--' + k.replace('_', '-') for k, v in others.items() if v)
        fail(f'--difficulty {args.difficulty} conflicts with {clash}; pass one or the other.')

    if args.difficulty == 'delay':
        if args.difficulty_pct:
            fail('--difficulty delay takes --delay-k (whole steps), not --difficulty-pct.')
        if args.delay_k <= 0:
            fail('--difficulty delay requires --delay-k > 0 (e.g. --delay-k 3).')
        return cfg   # delay_k already carried through

    pct = args.difficulty_pct
    if not 0.0 < pct <= 100.0:
        fail(f'--difficulty {args.difficulty} needs --difficulty-pct in (0, 100], got {pct}.')
    frac = pct / 100.0

    if args.difficulty == 'blank':
        cfg.update(blank_mode='block', blank_prob=frac)
    elif args.difficulty == 'vanish':
        # Same block schedule as 'blank' — only the fill differs, so the two are directly
        # comparable at matched (block_len, pct): identical frames are hidden either way.
        cfg.update(blank_mode='block', blank_prob=frac, blank_style='vanish')
    elif args.difficulty == 'region':
        cfg.update(region_frac=frac)
    elif args.difficulty == 'sticky':
        cfg.update(sticky_p=frac)
    else:
        fail(f'unhandled --difficulty {args.difficulty!r}')  # unreachable; argparse restricts
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Collect human demonstrations for GAIL (imitation library format)")
    parser.add_argument('--env', type=str, default='LunarLander-v2',
                        help='Gymnasium environment ID (default: LunarLander-v2)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: demos/<env>/human_demos)')
    parser.add_argument('--max_episodes', type=int, default=0,
                        help='Max episodes to collect; 0 = unlimited, play until you press Q '
                             '(default: 0). Discarded (X) attempts do not count.')
    parser.add_argument('--fps', type=int, default=20,
                        help='Target env steps per wall-clock second, in web/visual/curses modes '
                             '(default: 20). LunarLander physics is 50 steps/s, so --fps 20 = 0.40x '
                             'real time — matching the 20fps PT preference clips. The web loop '
                             'sleeps to a deadline, so achieved == requested.')
    parser.add_argument('--mode', type=str, choices=['web', 'visual', 'curses', 'text'], default='web',
                        help='Input mode: web (browser UI, default), visual (pygame window), curses (text), text (step-by-step)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for web mode (default: 8080)')
    parser.add_argument('--seed', type=int, default=42)
    # --- Unified difficulty interface (preferred). Resolves onto the per-technique flags
    # below, which keep working unchanged. Adding a new technique = one `choices` entry
    # here + one branch in `_resolve_difficulty`. ---
    parser.add_argument('--difficulty',
                        choices=['none', 'blank', 'vanish', 'region', 'sticky', 'delay'],
                        default='none',
                        help="Supervision-difficulty technique (web mode). PERCEPTION: 'blank' = "
                             "temporal blackouts (uses --block-len); 'vanish' = same blackout "
                             "schedule but only the LANDER disappears, terrain stays visible; "
                             "'region' = one fixed black "
                             "rectangle per episode. CONTROL: 'sticky' = repeat the previous action "
                             "w.p. p; 'delay' = apply each action k steps late (uses --delay-k). "
                             "Severity is --difficulty-pct, except delay which uses --delay-k. "
                             "Default: none.")
    parser.add_argument('--feedback', choices=['reward', 'outcome', 'both', 'none'],
                        default='reward',
                        help="What the pilot is told about how the flight went (web mode). "
                             "'reward' = the running score plus the episode total (default, "
                             "unchanged); 'outcome' = no numbers, just LANDED / CRASHED / OUT "
                             "OF TIME at the end; 'both'; 'none' = no feedback at all. The "
                             "console log and the saved data always keep the full reward "
                             "regardless — this only changes what is on screen.")
    parser.add_argument('--difficulty-pct', dest='difficulty_pct', type=float, default=0.0,
                        help='Severity of --difficulty, in PERCENT (e.g. 25 / 50 / 75). '
                             'blank/vanish -> %% of blocks hidden; region -> %% of the frame AREA '
                             'masked; sticky -> %% chance of repeating the previous action. '
                             'Not used by delay. Ignored when --difficulty=none.')
    # --- Control-interface difficulties: the view is perfect, the CONTROL is corrupted. ---
    parser.add_argument('--sticky-p', dest='sticky_p', type=float, default=0.0,
                        help='Sticky actions: repeat the PREVIOUS action with probability p '
                             '(0 = off). Feels like an unresponsive controller.')
    parser.add_argument('--sticky-seed', dest='sticky_seed', type=int, default=0)
    parser.add_argument('--delay-k', dest='delay_k', type=int, default=0,
                        help='Action delay: apply each action k STEPS late via a FIFO primed with '
                             'no-ops (0 = off). The human feels k/fps seconds of lag: at --fps 20, '
                             'k=3 is 150 ms (clearly laggy, still flyable), k=6 is 300 ms (beyond '
                             'the ~250 ms human reaction budget).')
    # --- Frame-blanking difficulty (Exp 1): blank the human's view (web mode) ---
    # Opt-in; default 'none' = unchanged. Human plays blind on blanked frames -> degraded demos.
    parser.add_argument('--blank-mode', dest='blank_mode',
                        choices=['none', 'stochastic', 'deterministic', 'block'], default='none',
                        help='Blank displayed frames so the human plays partly blind (web mode). '
                             'block = sustained blackouts (recommended, matches the PT video blanker); '
                             'stochastic per-frame flickers/blends. Default off.')
    parser.add_argument('--blank-prob', dest='blank_prob', type=float, default=0.5)
    parser.add_argument('--blank-k', dest='blank_k', type=int, default=2)
    parser.add_argument('--block-len', dest='block_len', type=int, default=10,
                        help='block mode: length of each blackout in displayed frames '
                             '(live play, so ms depends on frame rate — unlike the 20fps PT clips).')
    parser.add_argument('--blank-seed', dest='blank_seed', type=int, default=0)
    # --- Region masking difficulty (Sweep 1): fixed black rectangle over the whole episode ---
    # Opt-in; default 0.0 = unchanged. Spatial twin of blanking: area=frac*frame, location
    # uniform-random per episode (seeded), constant within the episode.
    parser.add_argument('--region-frac', dest='region_frac', type=float, default=0.0,
                        help='Fraction of the frame AREA masked by a fixed black box (web mode): '
                             '0.25/0.5/0.75. Location random-per-episode (seeded). Default 0 = off.')
    parser.add_argument('--region-seed', dest='region_seed', type=int, default=0)
    parser.add_argument('--region-outline', dest='region_outline', type=str, default=None,
                        metavar='COLOR',
                        help="Draw a coloured border inside the masked box (e.g. 'red') so it is "
                             "distinguishable from the black sky. Cosmetic: the occluded pixels "
                             "are unchanged. Default None = no outline.")
    parser.add_argument('--outline-thickness', dest='outline_thickness', type=int, default=3)
    # --- Task/dynamics difficulties (LunarLander): change the PHYSICS, not the view/keys. ---
    # Unlike the perception/control flags above, these alter the underlying task (and the eval
    # task too). Opt-in; defaults leave the env unchanged. Applied in _make_env at construction.
    parser.add_argument('--gravity', type=float, default=None, metavar='G',
                        help='LunarLander gravity (world units/s^2; env default -10). Applied by '
                             'writing world.gravity, so unlike the constructor kwarg it accepts '
                             'the FULL range incl. 0 (e.g. -3 easy, 0 = zero-g float, -12 hard). '
                             'Default None = unchanged.')
    parser.add_argument('--enable-wind', dest='enable_wind', action='store_true',
                        help='Turn on LunarLander wind + turbulence (gymnasium only). Off by '
                             'default = no wind. A steady, slowly-oscillating sideways push '
                             '(--wind-power) plus a tipping torque (--turbulence-power).')
    parser.add_argument('--wind-power', dest='wind_power', type=float, default=15.0,
                        help='Max linear wind force when --enable-wind (recommended 0-20, default '
                             '15 = up to ~0.3x gravity of sideways accel). Ignored without '
                             '--enable-wind.')
    parser.add_argument('--turbulence-power', dest='turbulence_power', type=float, default=1.5,
                        help='Max rotational wind (turbulence) when --enable-wind (recommended '
                             '0-2, default 1.5). This is what mostly makes the lander tip. '
                             'Ignored without --enable-wind.')
    parser.add_argument('--flag-target', dest='flag_target', type=int, default=50,
                        help='web mode: target number of FLAGGED (F-key) episodes; a '
                             '"target reached" prompt appears at this count. ALL episodes '
                             '(flagged or not) are saved; flags.json records which are flagged.')
    parser.add_argument('--auto-quit', dest='auto_quit', action='store_true',
                        help='web mode: stop as soon as --flag-target episodes are flagged. '
                             'Default off = keep playing until you press Q (so you can collect a '
                             'surplus and flag only the best).')
    args = parser.parse_args()

    dcfg = _resolve_difficulty(args, parser.error)
    outline_rgb = _outline_rgb(args.region_outline, parser.error)

    import gymnasium as gym

    # Probe env for info display
    probe_env = gym.make(args.env)
    discrete = _is_discrete(probe_env)
    # The env's own real-time rate (LunarLander: FPS = 50, i.e. one step = 20 ms of game time).
    physics_fps = int(probe_env.metadata.get('render_fps', 50))
    if discrete:
        print(f"Environment: {args.env}  (obs={probe_env.observation_space.shape[0]}, "
              f"discrete actions: {probe_env.action_space.n})")
    else:
        print(f"Environment: {args.env}  (obs={probe_env.observation_space.shape[0]}, "
              f"act={probe_env.action_space.shape[0]})")
    print(f"Mode: {args.mode}   Max episodes: {args.max_episodes or 'unlimited'}")
    print(f"Speed: --fps {args.fps} vs physics {physics_fps} steps/s "
          f"=> {args.fps / physics_fps:.2f}x real time")
    probe_env.close()

    all_attempts = []
    flags = None    # web mode fills this in (per-episode flagged bool); other modes: None

    if args.mode == 'web':
        episodes, ep_times, all_attempts, flags = _run_web(
            args.env, args.max_episodes, args.fps, discrete, args.seed, args.port,
            blank_mode=dcfg["blank_mode"], blank_prob=dcfg["blank_prob"],
            blank_k=args.blank_k, blank_seed=args.blank_seed, block_len=args.block_len,
            blank_style=dcfg["blank_style"], feedback=args.feedback,
            region_frac=dcfg["region_frac"], region_seed=args.region_seed,
            region_outline=outline_rgb, outline_thickness=args.outline_thickness,
            sticky_p=dcfg["sticky_p"], sticky_seed=args.sticky_seed, delay_k=dcfg["delay_k"],
            gravity=args.gravity, enable_wind=args.enable_wind,
            wind_power=args.wind_power, turbulence_power=args.turbulence_power,
            flag_target=args.flag_target, auto_quit=args.auto_quit)
    elif args.mode == 'visual':
        episodes, ep_times = _run_visual(
            args.env, args.max_episodes, args.fps, discrete, args.seed)
    elif args.mode == 'curses':
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        episodes, ep_times = _run_curses(env, args.max_episodes, args.fps, discrete)
        env.close()
    else:
        env = gym.make(args.env)
        env.reset(seed=args.seed)
        episodes, ep_times = _run_text(env, args.max_episodes, discrete)
        env.close()

    if len(episodes) == 0:
        print("No demos collected.")
        return

    # Auto-number: find next session_N folder that doesn't exist
    base_dir = args.output or f"demos/{args.env}/human_demos"
    session = 1
    while os.path.exists(os.path.join(base_dir, f"session_{session}")):
        session += 1
    output_dir = os.path.join(base_dir, f"session_{session}")

    save_demos(episodes, ep_times, output_dir, all_attempts=all_attempts)

    # Speed provenance: the fps a session was played at used to live nowhere, so it had to be
    # reconstructed from timing.csv + shell history. Record requested AND achieved, since the
    # two differ if anything ever regresses the deadline pacing.
    meta = _session_meta(args, episodes, ep_times, dcfg, physics_fps)
    _write_session_meta(output_dir, meta)

    # Flagging (web mode): ALL episodes are saved above; record which are FLAGGED
    # (to-use) in flags.json, and also write a flagged-only subset for training.
    if flags is not None:
        import json
        flagged_idx = [i for i, f in enumerate(flags) if f]
        with open(os.path.join(output_dir, "flags.json"), "w") as fh:
            json.dump({"flagged_indices": flagged_idx, "n_total": len(episodes),
                       "n_flagged": len(flagged_idx), "flag_target": args.flag_target}, fh, indent=2)
        print(f"  flags: {len(flagged_idx)}/{len(episodes)} flagged  (flags.json written)")
        if flagged_idx:
            save_demos([episodes[i] for i in flagged_idx],
                       [ep_times[i] for i in flagged_idx], output_dir + "_flagged")
            # Same meta in the _flagged sibling: that dir is what GAIL trains on, so it must
            # never be orphaned from the speed it was collected at.
            _write_session_meta(output_dir + "_flagged",
                                _session_meta(args, [episodes[i] for i in flagged_idx],
                                              [ep_times[i] for i in flagged_idx],
                                              dcfg, physics_fps))
            print(f"  flagged-only demos -> {output_dir}_flagged  ({len(flagged_idx)} demos)")

    print(f"\n  Next run will save to: {os.path.join(base_dir, f'session_{session+1}')}")


if __name__ == '__main__':
    main()
