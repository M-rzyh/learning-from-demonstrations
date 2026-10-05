"""Compatibility shim for dm_control_gymnasium_wrapper.

This module patches dm_control's mjbindings size table to ignore fields that
are not present in the installed MuJoCo build. This avoids crashes such as:
AttributeError: 'MjModel' object has no attribute 'B_colind'.

It then loads the original wrapper from /scratch/marzii/imitation_runs and
re-exports its public symbols.
"""
from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import mujoco
from dm_control.mujoco.wrapper.mjbindings import sizes


def _patch_sizes() -> None:
    mjmodel_sizes = sizes.array_sizes.get("mjmodel", {})
    if not mjmodel_sizes:
        return
    missing = [name for name in mjmodel_sizes.keys() if not hasattr(mujoco.MjModel, name)]
    for name in missing:
        mjmodel_sizes.pop(name, None)


def _load_original() -> ModuleType:
    path = "/scratch/marzii/imitation_runs/dm_control_gymnasium_wrapper.py"
    spec = importlib.util.spec_from_file_location("_dm_control_gymnasium_wrapper_orig", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load original wrapper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_patch_sizes()
_orig = _load_original()

# Re-export public attributes from the original module.
for _name in dir(_orig):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_orig, _name)

__all__ = [name for name in globals().keys() if not name.startswith("_")]
