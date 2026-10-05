#!/usr/bin/env python3
"""Prove IRL3's live lander-vanish is identical to PT's offline reference, and fast enough.

The two arms of the vanishing study only compare if the human sees the SAME thing in both:
PT labels pre-masked clips, GAIL pilots a live-masked window. Those are separate
implementations in separate repos (different gym pins, no shared import), so "the same" has
to be checked rather than assumed.

`inpaint_columns` is a line-for-line port. `lander_mask` computes the same arithmetic on 2-D
channel views instead of reducing over the channel axis (38 ms -> 11 ms), so the real question
this answers is whether that rearrangement is bit-identical on real frames.

Also times both, since the whole reason for the rewrite was latency: at 20 fps the mask has
50 ms of headroom, and a frame that misses it silently slows the human's clock — which would
confound perception difficulty with speed, the exact confound that made the earlier b=10
blanking result unusable.

    /scratch/marzii/envs/imitation-gail/bin/python imitation/test_lander_vanish_parity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/marzii/PT/PreferenceTransformer/scripts/preference")

import lander_vanish as live                                    # noqa: E402  (IRL3, live)
import blank_lander_videos as ref                               # noqa: E402  (PT, reference)

CLIP = ("/scratch/marzii/PT/lunarlander/frame_blanking/pt_human/videos_human_n350_clean"
        "/batch_000/pair_000/seg_A.mp4")


def frames(path):
    rd = imageio_ffmpeg.read_frames(str(path))
    meta = rd.__next__()
    w, h = meta["size"]
    return [np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in rd]


def main():
    fr = frames(CLIP)
    print(f"{len(fr)} frames from {Path(CLIP).parent.name}/seg_A.mp4  {fr[0].shape}")

    mask_diff = fill_diff = 0
    lander_seen = 0
    t_ref = t_live = 0.0

    for f in fr:
        t = time.perf_counter(); m_r = ref.lander_mask(f, 90, 0, 3)
        o_r = ref.inpaint_columns(f, m_r) if m_r.any() else f
        t_ref += time.perf_counter() - t

        t = time.perf_counter(); m_l = live.lander_mask(f, 90, 0, 3)
        o_l = live.inpaint_columns(f, m_l) if m_l.any() else f
        t_live += time.perf_counter() - t

        lander_seen += int(m_r.any())
        mask_diff += int((m_r != m_l).sum())
        fill_diff += int((o_r != o_l).sum())

    n = len(fr)
    print(f"\nlander visible on {lander_seen}/{n} frames")
    print(f"mask  pixels differing : {mask_diff}  {'OK' if mask_diff == 0 else 'MISMATCH'}")
    print(f"filled pixels differing: {fill_diff}  {'OK' if fill_diff == 0 else 'MISMATCH'}")
    print(f"\nper-frame cost   PT loop: {1000*t_ref/n:6.1f} ms"
          f"   vectorised: {1000*t_live/n:6.1f} ms   "
          f"({t_ref/max(t_live, 1e-9):.1f}x faster)")
    budget = 1000 / 20
    print(f"20 fps budget is {budget:.0f} ms/frame -> "
          f"{'FITS' if 1000*t_live/n < budget else 'TOO SLOW'} "
          f"({100*(1000*t_live/n)/budget:.0f}% of budget)")

    sys.exit(0 if mask_diff == 0 and fill_diff == 0 else 1)


if __name__ == "__main__":
    main()
