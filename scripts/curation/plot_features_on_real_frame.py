"""Overlay state features on a REAL PT video frame, all in WHITE (black-outlined so they stay
visible over the white terrain), no label boxes, no title. Shows:
  v_x, v_y (in-flight, on the lander), touchdown_vx, touchdown_vy (at the landing point),
  theta (angle), omega (angular velocity), land_x_offset.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_features_on_real_frame.py <frame.png>
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc

FRAME = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-3151758/-home-marzii-PT-PreferenceTransformer/37bf69d7-eaa2-4358-82fe-d197cfb72920/scratchpad/frames/frame_0.35.png"
OUT = "/home/marzii/IRL3/figures/features_on_real_frame.png"
W = "white"
STROKE = [pe.withStroke(linewidth=3, foreground="black")]


def main():
    img = plt.imread(FRAME)
    rgb = (img[..., :3] * 255).astype(float) if img.max() <= 1 else img[..., :3].astype(float)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    allm = (B > 95) & (R > 45) & (R < 185) & (G < 155) & (B > R + 10)   # body + legs
    ys, xs = np.where(allm); cx, cy = xs.mean(), ys.mean()
    body_m = (B > 195) & (R > 90) & (R < 175) & (G < 135)               # bright body only
    byy, bxx = np.where(body_m); bcx, bcy = bxx.mean(), byy.mean()
    # magnitude of tilt = rotation of the body's min-area bounding box
    bp = np.c_[bxx - bcx, byy - bcy].astype(float)
    best = (1e18, 0.0)
    for a in np.arange(-44, 45, 0.5):
        rr = np.radians(a); c, s = np.cos(rr), np.sin(rr)
        X = bp[:, 0] * c - bp[:, 1] * s; Y = bp[:, 0] * s + bp[:, 1] * c
        if (X.ptp() * Y.ptp()) < best[0]: best = (X.ptp() * Y.ptp(), a)
    mag = abs(best[1])
    # lean direction (head leans opposite the legs): legs = dim purple below the body
    lgy, lgx = np.where(allm & ~body_m)
    lean = (1.0 if (bcx - lgx.mean()) >= 0 else -1.0) if len(lgx) > 15 else np.sign(best[1] or 1)
    rad = np.radians(mag) * lean
    tilt = np.degrees(rad)

    ym = (R > 170) & (G > 150) & (B < 130)
    fxs, fys = np.where(ym)[1], np.where(ym)[0]
    split = np.median(fxs)
    padx = float((np.median(fxs[fxs <= split]) + np.median(fxs[fxs > split])) / 2)
    groundy = float(np.median(fys)) + 6

    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    ax.imshow(rgb.astype(np.uint8)); ax.axis("off")

    def arw(p0, p1, lw=2.4, rad=None):
        cs = f"arc3,rad={rad}" if rad is not None else "arc3"
        a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, color=W, lw=lw,
                            connectionstyle=cs, zorder=6); a.set_path_effects(STROKE); ax.add_patch(a)

    def lab(xy, t, ha="left", fs=11):
        ax.annotate(t, xy, color=W, fontsize=fs, ha=ha, va="center", zorder=9,
                    fontweight="bold", path_effects=STROKE)

    # ---- in-flight velocities on the lander ----
    L = 60
    arw((cx, cy), (cx + L, cy)); lab((cx + L + 6, cy), "v_x")
    arw((cx, cy), (cx, cy + L)); lab((cx + 6, cy + L + 6), "v_y")

    # ---- theta: angle between the lander HEADING (through its head) and world-vertical ----
    Lref = 88
    heading = np.array([np.sin(rad), -np.cos(rad)])
    l1, = ax.plot([cx, cx], [cy, cy - Lref], ls=(0, (4, 3)), color=W, lw=1.7, zorder=5)   # vertical
    l2, = ax.plot([cx - 36 * heading[0], cx + Lref * heading[0]],
                  [cy - 36 * heading[1], cy + Lref * heading[1]], ls="-",
                  color=W, lw=1.9, zorder=5)                                              # heading through body
    l1.set_path_effects(STROKE); l2.set_path_effects(STROKE)
    r = 48; angs = np.linspace(0, rad, 24)
    arc = np.array([[cx + r * np.sin(a), cy - r * np.cos(a)] for a in angs])
    la, = ax.plot(arc[:, 0], arc[:, 1], color=W, lw=2.0, zorder=6); la.set_path_effects(STROKE)
    mid = rad / 2
    lab((cx + (r + 15) * np.sin(mid), cy - (r + 15) * np.cos(mid)), "θ", ha="center")

    # ---- omega (angular velocity): curved arrow above ----
    arw((cx - 32, cy - 30), (cx + 30, cy - 34), lw=2.1, rad=-0.5)
    lab((cx - 36, cy - 44), "ω", ha="right")

    # ---- descent projection to the landing point ----
    ln2, = ax.plot([cx, cx], [cy + L, groundy - 8], ls=(0, (2, 3)), color=W, lw=1.1, zorder=4)
    ln2.set_path_effects(STROKE)

    # ---- land_x_offset: pad-center -> landing x ----
    lo, = ax.plot([padx, padx], [groundy - 34, groundy], ls=":", color=W, lw=1.2, zorder=4)
    lo.set_path_effects(STROKE)
    a = FancyArrowPatch((padx, groundy - 30), (cx, groundy - 30), arrowstyle="<|-|>",
                        mutation_scale=11, color=W, lw=2.0, zorder=6); a.set_path_effects(STROKE)
    ax.add_patch(a)
    lab(((padx + cx) / 2, groundy - 44), "land_x_offset", ha="center", fs=10)

    fig.tight_layout()
    fig.savefig(OUT, dpi=170, bbox_inches="tight", pad_inches=0.02)
    print("saved", OUT, f"| lander=({cx:.0f},{cy:.0f}) tilt={tilt:.0f} pad_x={padx:.0f} groundy={groundy:.0f}")


if __name__ == "__main__":
    main()
