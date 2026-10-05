"""Make the LunarLander VANISH from a rendered frame, keeping the terrain visible.

Live counterpart of PT/PreferenceTransformer/scripts/preference/blank_lander_videos.py.
That script rewrites recorded preference clips offline; this one masks frames as they are
rendered for the human pilot, so the same perception-difficulty axis applies to the GAIL
demonstration arm.

KEEP IN SYNC with the PT copy — the two repos pin different gym versions and cannot import
from each other, exactly as `make_control_corruptor` (in human_demo.py) is mirrored by PT's
`scripts/preference/control_wrappers.py`.

The masks here are BIT-IDENTICAL to PT's, verified frame-by-frame on real clips by
imitation/test_lander_vanish_parity.py. The code differs only in arithmetic layout: PT
reduces over the channel axis (`a.sum(2)`, `a.max(2) - a.min(2)`), which numpy does slowly,
while this file does the same arithmetic on the three 2-D channel views. Same result, 38 ms
-> 11 ms per frame. That matters because this runs inside a live 50 ms frame budget: a mask
that overran it would silently slow the human's clock during blackouts only, confounding
perception difficulty with speed — the exact confound that made the earlier b=10 blanking
result unusable. Do not "simplify" back to axis reductions.

WHY INPAINT AND NOT A BLACK BOX
  Sky renders pure black and ground pure white, so a flat black fill is invisible against sky
  but leaves a lander-shaped silhouette against the ground — leaking exactly what we remove.
  Each removed pixel is instead filled from the nearest surviving pixel in its own column.

WHAT COUNTS AS "THE LANDER"
  hull/legs   : unique colour (128,102,230); sky (0,0,0), ground (255,255,255), flags
                (204,204,0) — a colour match plus largest-connected-component is reliable.
  engine plume: particles fade (255,127,127) -> (51,51,51) and trail behind the craft, so
                hull-only masking would leave a flame pointing straight at it.
  fringe      : anti-aliasing smears the sprite edge, so the mask is dilated a few px.
"""
import numpy as np
from scipy import ndimage

HULL = (128, 102, 230)      # lander hull + legs (unique in the scene)
FLAG = (204, 204, 0)        # helipad flags — scenery, must NOT be removed
SKY_MAX = 40                # sum(rgb) below this == black sky
GROUND_MIN = 720            # sum(rgb) above this == white ground
GROUND_MID = 382            # midpoint of 0..765, splits sky from ground


def _l1(r, g, b, c):
    """L1 colour distance to `c`, computed per channel (see module docstring)."""
    return np.abs(r - c[0]) + np.abs(g - c[1]) + np.abs(b - c[2])


def lander_mask(fr, hull_tol=90, plume_radius=0, dilate=3, min_px=60):
    """Boolean mask of everything that is NOT the static scene.

    KEEP-LIST, not a find-list. LunarLander's background is exactly three things: black sky,
    white ground, yellow flags -- so anything else on screen IS the lander or its plume, and
    the mask is just the complement of the background.

    This replaced a "find the lander" rule (match the hull colour, then decide which other
    pixels look like flame). That rule has to JUDGE whether a pixel belongs to the lander, and
    it was wrong twice: orange particles near the pad were mistaken for flags and protected,
    and fully-faded particles are pure grey (51,51,51), which a colour-tint test walks past.
    A keep-list cannot make either mistake -- it never asks what a pixel is, only whether it
    is one of the three things known to belong here.

    It also removes the terrain's grey anti-aliased outline, which BackgroundPlate then paints
    straight back. That is only safe because the fill copies a real observed background rather
    than guessing a colour; measured offline against the old rule the repainted outline differs
    by mean 2.2/255. Deleting generously is safe once the fill is truthful.

    hull_tol / plume_radius / min_px are accepted and ignored, keeping the signature identical
    to the PT mirror.
    """
    a = fr.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    s = r + g + b
    black = s <= SKY_MAX
    white = s >= GROUND_MIN
    # Flags are YELLOW (r == g, b low). The L1 tolerance stays generous to catch their
    # anti-aliased edges, but 200 alone also matches ORANGE engine particles -- (204,97,90)
    # is only L1 197 from (204,204,0) -- so require yellowness too. Measured: true flag
    # pixels have |r-g| <= 11, those particles |r-g| = 107.
    yellow = (_l1(r, g, b, FLAG) < 200) & (np.abs(r - g) < 40)
    yellow = ndimage.binary_dilation(yellow, iterations=3)   # keep their soft edges too
    # dilate: the sprite edge is smeared by anti-aliasing, so grow the mask a little
    return ndimage.binary_dilation(~(black | white | yellow), iterations=dilate)


def moving_mask(fr, dilate=3):
    """Narrow mask: pixels belonging to the LANDER specifically, not merely non-background.

    Two different questions need two different masks, and conflating them is a trap:

      ERASE  -- "might this be the lander?"  Answer generously (lander_mask, the keep-list):
                missing a lander pixel leaks its position.
      LEARN  -- "is this safe to record as background?"  Answer conservatively HERE: the
                keep-list also rejects the terrain's grey anti-aliased outline, so learning
                through it would mean the outline is never recorded, and every hidden frame
                would repaint the horizon by guesswork -- making the horizon change shape
                exactly when the lander is hidden, a cue perfectly correlated with what we
                are hiding.

    Terrain anti-aliasing is grey AND borders the ground; the lander and its plume are either
    colour-tinted or floating free in the sky. A mistake here is cheap: at worst a faint
    particle enters the plate for one frame and the next frame overwrites it. Erasure is
    unaffected either way.
    """
    a = fr.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    s = r + g + b
    tinted = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 3
    yellow = (_l1(r, g, b, FLAG) < 200) & (np.abs(r - g) < 40)
    near_flag = ndimage.binary_dilation(yellow, iterations=3)
    far_from_ground = ~ndimage.binary_dilation(s >= GROUND_MIN, iterations=3)
    lander = (s > SKY_MAX) & (s < GROUND_MIN) & ~near_flag & (tinted | far_from_ground)
    return ndimage.binary_dilation(lander, iterations=dilate)


def inpaint_columns(fr, mask):
    """Fill each masked pixel from the nearest unmasked pixel in its column, SNAPPED to pure
    sky or pure ground.

    The scene is two flat regions, so a vertical nearest-neighbour fill lands on the right
    side of the horizon. Copying the neighbour's exact value would smear any contamination (a
    faint particle just outside the mask) down the column as a visible vertical ghost streak,
    so each filled pixel is snapped to pure black or pure white — whichever its source is
    closer to. That guarantees a flat, artefact-free fill.

    The loop is over the ~60 columns the lander actually occupies, not the frame, so it costs
    ~2 ms. A fully vectorised whole-frame version was tried and measured 3x SLOWER.
    """
    out = fr.copy()
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return out
    for x in np.unique(xs):
        bad = np.where(mask[:, x])[0]
        good = np.where(~mask[:, x])[0]
        if len(good) == 0:                      # fully masked column: nothing to copy from
            continue
        idx = np.searchsorted(good, bad)
        lo = np.clip(idx - 1, 0, len(good) - 1)
        hi = np.clip(idx, 0, len(good) - 1)
        pick = np.where(np.abs(good[lo] - bad) <= np.abs(good[hi] - bad), good[lo], good[hi])
        src_is_ground = fr[pick, x].astype(int).sum(1) > GROUND_MID
        out[bad, x] = np.where(src_is_ground[:, None], 255, 0)
    return out


class BackgroundPlate:
    """Running reconstruction of the static background, built from frames as they arrive.

    The offline PT masker takes the per-pixel median over a whole recorded episode, which it
    can do because the episode already exists. Live, there is no future to look at — so
    instead every pixel NOT currently covered by the lander is remembered, and covered pixels
    are filled from that memory. Within an episode the terrain, pad and flags never move, so
    a pixel only has to be seen once.

    This replaces filling with a guessed colour, which is where the two worst leaks came from:
    a black/white fill turns any covered flag into a bright white block (measured offline: up
    to 38% of the flag area destroyed, across 662 frames), and that block advertises the
    lander's position more loudly than the lander would.

    Seeding: at episode start the lander is already on screen, so its own pixels have never
    been observed. Those are filled by the column method until the lander moves off them,
    which happens within a few frames since it starts falling immediately. `reset()` must be
    called at every episode boundary — the terrain is regenerated, so a stale plate would
    paste the PREVIOUS episode's ground into this one.
    """

    def __init__(self):
        self.plate = None
        self.seen = None

    def reset(self):
        self.plate = None
        self.seen = None

    def observe(self, fr, dilate=3):
        """Learn the background from a frame the human is seeing normally.

        Uses `moving_mask` (narrow), NOT the keep-list: the keep-list rejects the terrain's
        grey outline, so learning through it would never record the outline and every hidden
        frame would repaint the horizon by guesswork. Must be called on UNBLANKED frames too,
        otherwise at 25% the plate only ever sees the frames it is meant to repair.
        """
        if self.plate is None:
            self.plate = fr.copy()
            self.seen = np.zeros(fr.shape[:2], bool)
        free = ~moving_mask(fr, dilate)
        self.plate[free] = fr[free]          # remember everything not hidden by the lander
        self.seen |= free

    def apply(self, fr, hull_tol=90, plume_radius=0, dilate=3):
        """Erase everything non-background, filling from the remembered background."""
        self.observe(fr, dilate)
        mk = lander_mask(fr, hull_tol, plume_radius, dilate)   # generous: what to blank out
        if not mk.any():
            return fr
        out = fr.copy()
        take = mk & self.seen
        out[take] = self.plate[take]
        leftover = mk & ~self.seen           # never yet observed (start-of-episode only)
        return inpaint_columns(out, leftover) if leftover.any() else out


def vanish(fr, hull_tol=90, plume_radius=0, dilate=3):
    """Stateless single-frame erase. Prefer BackgroundPlate for live play — without a plate
    the fill has to guess a colour and cannot represent the flags."""
    mk = lander_mask(fr, hull_tol, plume_radius, dilate)
    return inpaint_columns(fr, mk) if mk.any() else fr
