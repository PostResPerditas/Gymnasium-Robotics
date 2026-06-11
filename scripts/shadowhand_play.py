"""Play, evaluate, and record a ShadowHand manipulation policy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained ShadowHand policy or a random rollout."
    )
    parser.add_argument("--env-id", default="HandManipulateBlockRotateZ-v1")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to an SB3 SAC zip. Omit this to visualize a random policy.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--human", action="store_true", help="Use MuJoCo human viewer.")
    parser.add_argument("--video-dir", default=None, help="Write one gif/mp4 per episode.")
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="gif")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def load_sac(model_path: str, env: gym.Env, device: str) -> Any:
    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing stable_baselines3. Install `stable-baselines3[extra]` before "
            "loading a trained policy."
        ) from exc

    return SAC.load(model_path, env=env, device=device)


def write_video(video_dir: Path, env_id: str, episode: int, frames: list[Any], args: argparse.Namespace) -> None:
    if not frames:
        return

    video_dir.mkdir(parents=True, exist_ok=True)
    safe_env_id = env_id.replace("/", "_").replace(":", "_")
    path = video_dir / f"{safe_env_id}_episode_{episode:03d}.{args.video_format}"
    imageio.mimsave(path, frames, fps=args.fps)
    print("video:", path)


def main() -> None:
    args = parse_args()
    gym.register_envs(gymnasium_robotics)

    render_mode = "human" if args.human else ("rgb_array" if args.video_dir else None)
    env = gym.make(args.env_id, render_mode=render_mode)
    model = load_sac(args.model_path, env, args.device) if args.model_path else None

    successes = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            frames = []
            total_reward = 0.0
            last_info = {}
            max_steps = args.max_steps or getattr(env.spec, "max_episode_steps", 0) or 1_000

            if args.video_dir:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            for step in range(max_steps):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(
                        obs, deterministic=not args.stochastic
                    )

                obs, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)

                if args.video_dir:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                elif args.human:
                    env.render()

                if terminated or truncated:
                    break

            success = float(last_info.get("is_success", 0.0))
            successes.append(success)
            print(
                f"episode={episode} steps={step + 1} reward={total_reward:.3f} "
                f"success={success:.0f}"
            )

            if args.video_dir:
                write_video(Path(args.video_dir), args.env_id, episode, frames, args)
    finally:
        env.close()

    if successes:
        print(f"success_rate={sum(successes) / len(successes):.3f}")


if __name__ == "__main__":
    main()
