"""ActionRepeat (frame-skip) wrapper + registration for LunarLander-v2-FS10."""

import gymnasium as gym


class ActionRepeatWrapper(gym.Wrapper):
    """Repeat each action for `k` underlying env steps.

    Reward is summed across the repeated steps; the loop ends early if the
    underlying env terminates or truncates. Exposes one (obs, r, term, trunc, info)
    per policy decision.
    """

    def __init__(self, env, k=10):
        super().__init__(env)
        self.k = int(k)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def _make_lunar_lander_fs10(**kwargs):
    from gymnasium.envs.box2d import LunarLander
    base = LunarLander(**kwargs)
    return ActionRepeatWrapper(base, k=10)


gym.register(
    id="LunarLander-v2-FS10",
    entry_point=_make_lunar_lander_fs10,
    max_episode_steps=40,
)
