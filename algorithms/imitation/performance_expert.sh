python - <<'PY'
from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np

model_path = "/scratch/marzii/imitation_runs/expert_ppo_walker_3000000/4258333/ppo_walker.zip"
env = gym.make("Walker2d-v4")
model = PPO.load(model_path, device="cuda")

returns = []
for ep in range(10):
    obs, _ = env.reset()
    done = False
    trunc = False
    ep_ret = 0.0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, _ = env.step(action)
        ep_ret += reward
    returns.append(ep_ret)
    print(f"episode {ep+1}: {ep_ret:.2f}")

print("mean:", np.mean(returns))
print("std:", np.std(returns))
print("min:", np.min(returns))
print("max:", np.max(returns))
env.close()
PY
