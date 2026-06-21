"""Play and record ShadowHand shaped PPO policies with VecNormalize support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np


SUPPORTED_ALGOS = {"ddpg", "ppo", "sac", "td3"}
DEFAULT_ENV_ID = "HandManipulateBlockRotateZ-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a trained ShadowHand shaped PPO policy or a random rollout, "
            "including reward wrapper and VecNormalize restoration."
        )
    )
    parser.add_argument(
        "--env-id",
        default=None,
        help="Env id. Defaults to model config env_id, then HandManipulateBlockRotateZ-v1.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to an SB3 zip. Omit this to visualize a random policy.",
    )
    parser.add_argument(
        "--algo",
        choices=("auto", *sorted(SUPPORTED_ALGOS)),
        default="auto",
        help="SB3 algorithm used by the model. auto reads config.json next to the model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Training config.json path. Defaults to <model-path parent>/config.json "
            "when a model path is provided."
        ),
    )
    parser.add_argument(
        "--env-kwargs-json",
        default=None,
        help="JSON object passed to gym.make, overriding env_kwargs from config.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("env", "isaac"),
        default=None,
        help="Reward wrapper override. Defaults to reward_mode from config.",
    )
    parser.add_argument(
        "--reward-kwargs-json",
        default=None,
        help="JSON object passed to the reward wrapper.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Path to SB3 VecNormalize stats. Defaults to config vecnormalize_path or model directory.",
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


def load_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be a valid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object.")
    return value


def load_training_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path: Path | None = Path(args.config).expanduser() if args.config else None
    if config_path is None and args.model_path:
        config_path = Path(args.model_path).expanduser().parent / "config.json"

    if config_path is None or not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise SystemExit(f"Training config must contain a JSON object: {config_path}")

    print("config:", config_path.resolve())
    return config


def resolve_env_config(
    args: argparse.Namespace,
    training_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    env_id = args.env_id or str(training_config.get("env_id") or DEFAULT_ENV_ID)
    env_kwargs = dict(training_config.get("env_kwargs") or {})
    if args.env_kwargs_json:
        env_kwargs = load_json_object(args.env_kwargs_json, "--env-kwargs-json")
    return env_id, env_kwargs


def resolve_reward_config(
    args: argparse.Namespace,
    training_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    reward_mode = args.reward_mode or str(training_config.get("reward_mode") or "env")
    reward_kwargs = dict(training_config.get("reward_kwargs") or {})
    if args.reward_kwargs_json:
        reward_kwargs = load_json_object(args.reward_kwargs_json, "--reward-kwargs-json")
    return reward_mode, reward_kwargs


def resolve_algo(args: argparse.Namespace, training_config: Mapping[str, Any]) -> str:
    if args.algo != "auto":
        return args.algo

    algo = training_config.get("algo")
    if algo is None:
        raise SystemExit(
            "--algo auto could not infer the algorithm. Pass --algo ppo/sac/td3/ddpg "
            "or provide --config pointing to the training config.json."
        )

    algo = str(algo).lower()
    if algo not in SUPPORTED_ALGOS:
        raise SystemExit(f"Unsupported algo in config: {algo!r}.")
    return algo


def resolve_vecnormalize_path(
    args: argparse.Namespace,
    training_config: Mapping[str, Any],
) -> Path | None:
    if args.vecnormalize_path:
        path = Path(args.vecnormalize_path).expanduser()
        return path if path.exists() else None

    config_path = training_config.get("vecnormalize_path")
    if config_path:
        path = Path(str(config_path)).expanduser()
        if path.exists():
            return path

    if args.model_path:
        path = Path(args.model_path).expanduser().parent / "vecnormalize.pkl"
        if path.exists():
            return path
    return None


def make_raw_env(
    env_id: str,
    env_kwargs: Mapping[str, Any],
    reward_mode: str,
    reward_kwargs: Mapping[str, Any],
    render_mode: str | None,
) -> gym.Env:
    from shadowhand_wrappers import apply_shadowhand_reward_wrapper

    env = gym.make(env_id, render_mode=render_mode, **dict(env_kwargs))
    return apply_shadowhand_reward_wrapper(env, reward_mode, dict(reward_kwargs))


def load_model(model_path: str, algo: str, env: gym.Env, device: str) -> Any:
    try:
        from stable_baselines3 import DDPG, PPO, SAC, TD3
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing stable_baselines3. Install `stable-baselines3[extra]` before "
            "loading a trained policy."
        ) from exc

    algo_classes = {
        "ddpg": DDPG,
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }
    resolved_model_path = str(Path(model_path).expanduser())
    return algo_classes[algo].load(resolved_model_path, env=env, device=device)


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
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    training_config = load_training_config(args)
    env_id, env_kwargs = resolve_env_config(args, training_config)
    reward_mode, reward_kwargs = resolve_reward_config(args, training_config)
    vecnormalize_path = resolve_vecnormalize_path(args, training_config)

    render_mode = "human" if args.human else ("rgb_array" if args.video_dir else None)
    spec = gym.spec(env_id)
    max_steps = args.max_steps or getattr(spec, "max_episode_steps", 0) or 1_000

    if vecnormalize_path:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        env = DummyVecEnv(
            [
                lambda: make_raw_env(
                    env_id,
                    env_kwargs,
                    reward_mode,
                    reward_kwargs,
                    render_mode,
                )
            ]
        )
        env = VecNormalize.load(str(vecnormalize_path), env)
        env.training = False
        env.norm_reward = False
        vectorized_env = True
        print("vecnormalize:", vecnormalize_path.resolve())
    else:
        env = make_raw_env(env_id, env_kwargs, reward_mode, reward_kwargs, render_mode)
        vectorized_env = False

    if args.model_path:
        algo = resolve_algo(args, training_config)
        print("algo:", algo.upper())
        print("model:", Path(args.model_path).expanduser().resolve())
        print("env_id:", env_id)
        print("reward_mode:", reward_mode)
        model = load_model(args.model_path, algo, env, args.device)
    else:
        model = None

    successes = []
    try:
        for episode in range(args.episodes):
            if vectorized_env:
                env.seed(args.seed + episode)
                obs = env.reset()
            else:
                obs, _ = env.reset(seed=args.seed + episode)
            frames = []
            total_reward = 0.0
            last_info = {}

            if args.video_dir:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            for step in range(max_steps):
                if model is None:
                    if vectorized_env:
                        action = np.asarray([env.action_space.sample()])
                    else:
                        action = env.action_space.sample()
                else:
                    action, _ = model.predict(
                        obs, deterministic=not args.stochastic
                    )

                if vectorized_env:
                    obs, rewards, dones, infos = env.step(action)
                    total_reward += float(rewards[0])
                    last_info = infos[0]
                    done = bool(dones[0])
                else:
                    obs, reward, terminated, truncated, last_info = env.step(action)
                    total_reward += float(reward)
                    done = terminated or truncated

                if args.video_dir:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                elif args.human:
                    env.render()

                if done:
                    break

            success = float(last_info.get("is_success", 0.0))
            successes.append(success)
            print(
                f"episode={episode} steps={step + 1} reward={total_reward:.3f} "
                f"success={success:.0f}"
            )

            if args.video_dir:
                write_video(Path(args.video_dir), env_id, episode, frames, args)
    finally:
        env.close()

    if successes:
        print(f"success_rate={sum(successes) / len(successes):.3f}")


if __name__ == "__main__":
    main()
