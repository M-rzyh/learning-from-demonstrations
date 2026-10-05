"""A LunarLander schematic (same style as figures/image.png) annotated with the 9 ranking
features, each drawn where it physically applies: in-flight, at touchdown, global, and the
cross-demo one as an inset.

    /scratch/marzii/envs/imitation-gail/bin/python scripts/curation/plot_nine_features_diagram.py
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Arc, RegularPolygon, FancyBboxPatch

FLY_C, TD_C, GOOD_C, BAD_C = "#4fc3f7", "#ffd54f", "#69f0ae", "#ff6e6e"
OUT = "/home/marzii/IRL3/figures/nine_features_lander_diagram.png"


def lander(ax, x, y, s=0.42, color="#7c6cf0", alpha=1.0, tilt=0.0):
    d = RegularPolygon((x, y), numVertices=4, radius=s, orientation=np.radians(45 + tilt),
                       facecolor=color, edgecolor="white", lw=1.2, alpha=alpha, zorder=6)
    ax.add_patch(d)


def arrow(ax, p0, p1, c, lw=2.2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 color=c, lw=lw, zorder=7))


def label(ax, xy, text, c, ha="left"):
    ax.annotate(text, xy, color=c, fontsize=9.5, ha=ha, va="center", zorder=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc="black", ec=c, lw=1.1, alpha=0.9))


def main():
    fig, ax = plt.subplots(figsize=(13, 8.6))
    fig.patch.set_facecolor("black"); ax.set_facecolor("black")
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    # ---- terrain + landing pad + flags ----
    terrain = [(0, 1.6), (2, 1.2), (3.2, 1.9), (4.3, 2.0), (5.7, 2.0), (6.8, 1.7),
               (8.2, 2.3), (9.6, 1.4), (11, 2.0), (13, 1.5), (13, 0), (0, 0)]
    ax.add_patch(Polygon(terrain, closed=True, facecolor="white", edgecolor="none", zorder=1))
    padc = 5.0
    for fx in (4.3, 5.7):
        ax.plot([fx, fx], [2.0, 2.7], color="#c9b100", lw=2, zorder=2)
        ax.add_patch(Polygon([(fx, 2.7), (fx + 0.28, 2.58), (fx, 2.46)], closed=True,
                             facecolor="#e8d100", zorder=2))
    ax.plot([4.3, 5.7], [2.0, 2.0], color="#e8d100", lw=2.5, zorder=2)

    # ---- dashed descent trajectory ----
    tx = np.array([3.6, 3.9, 4.4, 4.9, 5.25, 5.4]); ty = np.array([6.6, 5.9, 4.9, 3.6, 2.7, 2.2])
    ax.plot(tx, ty, ls=(0, (4, 3)), color="#888", lw=1.4, zorder=3)

    # ================= IN-FLIGHT (cyan) =================
    fx, fy = 3.9, 5.9
    lander(ax, fx, fy, tilt=-14)
    arrow(ax, (fx, fy - .3), (fx, fy - 1.35), FLY_C)                       # descent_rate (down)
    arrow(ax, (fx + .3, fy), (fx + 1.25, fy + .18), FLY_C)                 # vx_mean_abs (lateral)
    ax.add_patch(Arc((fx, fy), 1.5, 1.5, angle=90, theta1=-14, theta2=14, color=FLY_C, lw=2))  # theta
    arrow(ax, (fx - .95, fy + .55), (fx - .55, fy + .95), FLY_C, lw=1.8)   # angvel (rotation)
    label(ax, (fx - .1, fy - 1.7), "descent_rate  ⟨max(−v_y,0)⟩   ↓", FLY_C, ha="center")
    label(ax, (fx + 1.35, fy + .18), "vx_mean_abs  ⟨|v_x|⟩   ↓", FLY_C)
    label(ax, (fx + .15, fy + 1.25), "angle_mean_abs  ⟨|θ|⟩   ↓", FLY_C)
    label(ax, (fx - 1.35, fy + 1.15), "angvel_mean_abs  ⟨|ω|⟩   ↓", FLY_C, ha="right")

    # ================= AT TOUCHDOWN (yellow) =================
    txd, tyd = 5.4, 2.25
    lander(ax, txd, tyd, s=0.34, color="#7c6cf0", alpha=0.85, tilt=-6)
    arrow(ax, (txd, tyd + .1), (txd, tyd - .55), TD_C, lw=1.8)             # touchdown_vy
    arrow(ax, (txd + .15, tyd), (txd + .8, tyd + .05), TD_C, lw=1.8)       # touchdown_vx
    ax.annotate("", (padc, 1.75), (txd, 1.75),
                arrowprops=dict(arrowstyle="<->", color=TD_C, lw=1.8))     # land_x_offset
    ax.plot([padc, padc], [1.55, 2.0], color=TD_C, ls=":", lw=1.2)
    label(ax, (txd + .9, tyd - .35), "touchdown_vy  |v_{y,T}|   ↓\ntouchdown_vx  |v_{x,T}|   ↓", TD_C)
    label(ax, ((padc + txd) / 2, 1.35), "land_x_offset  |x_T|   ↓", TD_C, ha="center")

    # ================= GLOBAL =================
    label(ax, (0.35, 8.15), "return   R = Σ r_t     ↑ HIGHER better", GOOD_C)
    ax.annotate("(everything else: ↓ LOWER better)", (0.4, 7.65), color="#bbb", fontsize=8.5, ha="left")

    # ================= CROSS-DEMO inset (red) =================
    ix, iy, iw, ih = 8.9, 4.5, 3.7, 3.4
    ax.add_patch(FancyBboxPatch((ix, iy), iw, ih, boxstyle="round,pad=0.05",
                                fc="#140000", ec=BAD_C, lw=1.5, zorder=5))
    cx, cy = ix + iw / 2, iy + ih - 0.75
    ax.annotate("same state, 3 demos →", (cx, cy + 0.6), color="#ddd", fontsize=8.5,
                ha="center", va="center", zorder=9)
    lander(ax, cx, cy, s=0.32, color="#7c6cf0")
    for dx, dy, cc in [(-1.05, -0.55, "#ff6e6e"), (0.0, -1.15, "#ffb0b0"), (1.05, -0.45, "#ff9a9a")]:
        arrow(ax, (cx, cy - .2), (cx + dx, cy + dy - .2), cc, lw=1.9)
    ax.annotate("3 different actions", (cx, cy - 1.55), color="#ddd", fontsize=8.5,
                ha="center", va="center", zorder=9)
    label(ax, (cx, iy + 0.45), "action_divergence  ⟨1/k Σ 1[a_j ≠ a_i]⟩   ↓", BAD_C, ha="center")

    ax.set_title("The 9 ranking features on a LunarLander descent\n"
                 "cyan = in-flight stability   ·   yellow = touchdown quality   ·   "
                 "green = outcome   ·   red = cross-demo consistency",
                 color="white", fontsize=13, pad=14)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor="black"); print("saved", OUT)


if __name__ == "__main__":
    main()
