"""Launcher for the imitation library's train_imitation experiment (BC / DAgger / SQIL).

Mirrors ../train_adversarial_launcher.py: importing imitation.envs.action_repeat registers the
custom FS10 action-repeat env before sacred parses the command, then hands off to the sacred
CLI. Run e.g.:  python bc/train_imitation_launcher.py bc with lunar_lander ...
"""
import imitation.envs.action_repeat  # noqa: F401  (triggers gym.register for FS envs)
from imitation.scripts.train_imitation import train_imitation_ex

if __name__ == "__main__":
    train_imitation_ex.run_commandline()
