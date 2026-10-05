"""Canonical output location for IRL3 plots.

All PNGs live in /home/marzii/IRL3/figures (NOT alongside the plotting code in
scripts/). Plot scripts should route their output through `fig_path()`:

    from figures_dir import fig_path
    fig.savefig(fig_path(args.output), dpi=150)

`fig_path` resolves a bare filename ("gail_curves.png") into the figures dir and
leaves an explicit path ("/tmp/x.png", "./x.png") untouched, so callers that pass
a full path keep working exactly as before.
"""
from pathlib import Path

FIGURES = Path("/home/marzii/IRL3/figures")


def fig_path(name) -> Path:
    """Bare filename -> figures/<name>; anything with a directory part is left as-is."""
    p = Path(name)
    if p.parent != Path("."):
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / p.name
