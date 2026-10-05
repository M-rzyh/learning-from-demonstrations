"""Launcher that registers custom envs (ActionRepeat FS10) before running train_rl."""
import imitation.envs.action_repeat  # noqa: F401  (triggers gym.register)
from imitation.scripts.train_rl import train_rl_ex

if __name__ == "__main__":
    train_rl_ex.run_commandline()
