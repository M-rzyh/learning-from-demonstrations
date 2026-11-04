#!/usr/bin/env python3
"""Check the vanish mask on LIVE rendered frames, not recorded clips.

The mask thresholds were tuned on H.264-compressed preference videos. The GAIL arm masks the
raw rgb_array render instead: no compression, so colours are exact and detection should be
strictly easier — but "should be" is how the earlier particle leak survived, so verify.

Steps a real LunarLander with a thrust-happy policy (to guarantee engine plume), masks every
frame, and asserts on each:
  - no hull-coloured pixel survives          (the lander is actually gone)
  - no tinted pixel survives in the sky      (no engine particles pointing at it)
  - the ground is still there                (vanishing, not blanking)
and reports per-frame cost against the 50 ms budget at fps 20.

    /scratch/marzii/envs/imitation-gail/bin/python imitation/test_lander_vanish_live.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lander_vanish import HULL, lander_mask, inpaint_columns, _l1  # noqa: E402

STEPS = 300
FPS = 20


def main():
    import gymnasium as gym
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)

    n = hull_left = sky_dots = no_ground = lander_seen = 0
    t_tot = 0.0
    shapes = set()

    for i in range(STEPS):
        fr = env.render()
        shapes.add(fr.shape)

        t = time.perf_counter()
        mk = lander_mask(fr)
        out = inpaint_columns(fr, mk) if mk.any() else fr
        t_tot += time.perf_counter() - t

        a = out.astype(np.int16)
        r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        lander_seen += int(mk.any())
        hull_left += int((_l1(r, g, b, HULL) < 90).sum())
        s = r + g + b
        tint = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        # upper half only: helipad flags live near the ground and are legitimately tinted
        sky_dots += int(((s > 40) & (s < 720) & (tint > 3))[:fr.shape[0] // 2].sum())
        no_ground += int((s > 720).sum() < 500)
        n += 1

        # bias toward the main engine so there is always a plume to remove
        act = 2 if rng.random() < 0.6 else int(rng.integers(0, 4))
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    ms = 1000 * t_tot / n
    print(f"frames                 : {n}   render shapes {shapes}")
    print(f"lander detected on     : {lander_seen}/{n}")
    print(f"hull px surviving      : {hull_left}   {'OK' if hull_left == 0 else 'LEAK'}")
    print(f"sky dots surviving     : {sky_dots}   {'OK' if sky_dots == 0 else 'LEAK'}")
    print(f"frames w/o ground      : {no_ground}   {'OK' if no_ground == 0 else 'TERRAIN GONE'}")
    print(f"cost                   : {ms:.1f} ms/frame  "
          f"({100 * ms / (1000 / FPS):.0f}% of the {1000 // FPS} ms budget at {FPS} fps)  "
          f"{'FITS' if ms < 1000 / FPS else 'TOO SLOW'}")
    ok = hull_left == 0 and sky_dots == 0 and no_ground == 0 and ms < 1000 / FPS
    print("\n" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
