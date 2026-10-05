"""Render a trained GAIL/PPO generator policy flying LunarLander to an MP4.

    python render_policy_video.py --run_dir <gail_run_dir> --out out.mp4 [--seed 0] [--title "..."]

Loads <run_dir>/checkpoints/final/gen_policy/model.zip (falls back to checkpoints/final/model.zip
or an explicit --model), rolls ONE full episode (deterministic) in LunarLander-v2 at cap 1000,
and writes an MP4 with a small title bar (condition + episode return). Reusable across conditions.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, gymnasium as gym, imageio_ffmpeg
from PIL import Image, ImageDraw
import imitation.envs.action_repeat  # noqa: F401
from stable_baselines3 import PPO


def find_model(run_dir, explicit):
    if explicit:
        return explicit
    for c in ["checkpoints/final/gen_policy/model.zip", "checkpoints/final/model.zip"]:
        p = Path(run_dir) / c
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"no policy .zip under {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir"); ap.add_argument("--model"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=50); ap.add_argument("--title", default="")
    args = ap.parse_args()

    policy = PPO.load(find_model(args.run_dir, args.model))
    env = gym.make("LunarLander-v2", render_mode="rgb_array", max_episode_steps=args.max_steps)
    obs, _ = env.reset(seed=args.seed)
    frames, R, done = [], 0.0, False
    while not done:
        frames.append(env.render())
        a, _ = policy.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(int(a)); R += r; done = term or trunc
    frames.append(env.render()); env.close()

    h, w, _ = frames[0].shape
    HDR = 26 if args.title else 0
    label = f"{args.title}   return {R:+.0f}" if args.title else ""
    hdr = None
    if HDR:
        im = Image.new("RGB", (w, HDR), (0, 0, 0)); d = ImageDraw.Draw(im)
        d.text((6, 7), label, fill=(255, 255, 255)); hdr = np.array(im)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wr = imageio_ffmpeg.write_frames(args.out, (w, h + HDR), fps=args.fps, codec="libx264",
                                     pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                     macro_block_size=1, quality=7)
    wr.send(None)
    for f in frames:
        out = np.concatenate([hdr, f], axis=0) if HDR else f
        wr.send(np.ascontiguousarray(out, dtype=np.uint8))
    wr.close()
    print(f"wrote {args.out}  ({len(frames)} frames, return {R:+.1f})")


if __name__ == "__main__":
    main()
