"""Launcher that registers custom envs (ActionRepeat FS10) before running train_adversarial."""
import imitation.envs.action_repeat  # noqa: F401  (triggers gym.register)
from imitation.scripts.train_adversarial import train_adversarial_ex

if __name__ == "__main__":
    train_adversarial_ex.run_commandline()
