"""Train a ShadowHand in-hand manipulation baseline with SAC + HER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import gymnasium_robotics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Gymnasium-Robotics ShadowHand manipulation with SAC + HER."
    )
    parser.add_argument(
        "--env-id",
        default="HandManipulateBlockRotateZ-v1",
        help="Gymnasium-Robotics env id. Start with RotateZ, then move to XYZ/Full.",
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device passed to SB3. Use cuda for GPU training, auto for fallback.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="runs/shadowhand")
    parser.add_argument("--model-dir", default="runs/models")
    parser.add_argument("--tensorboard-log", default="runs/tensorboard")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument(
        "--goal-selection-strategy",
        choices=("future", "final", "episode"),
        default="future",
    )
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        help="Reset/step/render one episode sample and exit before importing SB3.",
    )
    return parser.parse_args()


def register_envs() -> None:
    gym.register_envs(gymnasium_robotics)


def make_env(env_id: str, seed: int, monitor_file: Path | None = None) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)

    if monitor_file is not None:
        from stable_baselines3.common.monitor import Monitor

        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_file), info_keywords=("is_success",))
    return env


def import_training_stack() -> Tuple[Any, ...]:
    try:
        import torch
        from stable_baselines3 import HerReplayBuffer, SAC
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install PyTorch with a CUDA wheel first, then "
            "`pip install 'stable-baselines3[extra]'` in a Python 3.10+ environment."
        ) from exc

    return torch, SAC, HerReplayBuffer, CallbackList, CheckpointCallback, EvalCallback


def check_env(env_id: str, seed: int) -> None:
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frame = env.render()
    print("env_id:", env_id)
    print("action_space:", env.action_space)
    print("obs_shapes:", {key: value.shape for key, value in obs.items()})
    print("step:", reward, terminated, truncated, info)
    print("frame:", frame.shape, frame.dtype, int(frame.min()), int(frame.max()))
    env.close()


def main() -> None:
    args = parse_args()
    register_envs()

    if args.check_env_only:
        check_env(args.env_id, args.seed)
        return

    torch, SAC, HerReplayBuffer, CallbackList, CheckpointCallback, EvalCallback = (
        import_training_stack()
    )

    run_name = args.run_name or f"{args.env_id}_sac_her_seed{args.seed}"
    log_dir = Path(args.log_dir) / run_name
    model_dir = Path(args.model_dir) / run_name
    tb_dir = Path(args.tensorboard_log)
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())

    env = make_env(args.env_id, args.seed, log_dir / "train_monitor.csv")
    eval_env = make_env(args.env_id, args.seed + 10_000, log_dir / "eval_monitor.csv")

    callbacks = []
    if args.eval_freq > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(log_dir),
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )
    if args.save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(model_dir),
                name_prefix="checkpoint",
            )
        )
    callback = CallbackList(callbacks) if callbacks else None

    with (model_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": args.n_sampled_goal,
            "goal_selection_strategy": args.goal_selection_strategy,
        },
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tensorboard_log=str(tb_dir),
        device=args.device,
        verbose=args.verbose,
        seed=args.seed,
    )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
            tb_log_name=run_name,
        )
        final_path = model_dir / "final_model"
        model.save(final_path)
        print("saved:", final_path.with_suffix(".zip"))
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
