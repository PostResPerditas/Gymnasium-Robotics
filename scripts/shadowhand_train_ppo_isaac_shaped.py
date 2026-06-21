"""Train ShadowHand PPO with Isaac-style shaped reward and VecNormalize."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym


CONFIG_DIR = Path(__file__).resolve().parent / "config"
OFF_POLICY_ALGOS = {"sac", "td3", "ddpg"}
ON_POLICY_ALGOS = {"ppo"}
SUPPORTED_ALGOS = OFF_POLICY_ALGOS | ON_POLICY_ALGOS
HER_ALGOS = OFF_POLICY_ALGOS

DEFAULTS: dict[str, Any] = {
    "env_id": "HandManipulateBlockRotateZ-v1",
    "env_kwargs": {},
    "algo": "ppo",
    "policy": "MultiInputPolicy",
    "policy_kwargs": {},
    "algo_kwargs": {},
    "reward_mode": "isaac",
    "reward_kwargs": {},
    "use_her": False,
    "timesteps": 100_000,
    "seed": 0,
    "device": "cuda",
    "run_name": None,
    "run_dir": None,
    "output_root": None,
    "log_dir": "runs/shadowhand",
    "model_dir": "runs/models",
    "tensorboard_log": "runs/tensorboard",
    "num_envs": 1,
    "eval_num_envs": 1,
    "vec_env": "subproc",
    "vec_normalize": True,
    "norm_obs": True,
    "norm_reward": False,
    "norm_clip_obs": 10.0,
    "norm_clip_reward": 10.0,
    "norm_epsilon": 1e-8,
    "buffer_size": 200_000,
    "learning_starts": 5_000,
    "batch_size": 8192,
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "tau": 0.05,
    "train_freq": 1,
    "gradient_steps": 1,
    "n_sampled_goal": 4,
    "goal_selection_strategy": "future",
    "n_steps": 512,
    "n_epochs": 5,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 4.0,
    "max_grad_norm": 1.0,
    "eval_freq": 10_000,
    "n_eval_episodes": 10,
    "save_freq": 50_000,
    "log_interval": 10,
    "verbose": 1,
    "progress_bar": False,
    "check_env_only": False,
}

CONFIG_SECTIONS = {
    "algorithm",
    "ddpg",
    "env",
    "environment",
    "her",
    "normalization",
    "output",
    "parallel",
    "paths",
    "ppo",
    "reward",
    "sac",
    "td3",
    "train",
    "training",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Gymnasium-Robotics ShadowHand PPO with an Isaac-style shaped "
            "reward wrapper and optional VecNormalize."
        ),
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        help=(
            "JSON/TOML config path. Bare names are resolved under scripts/config, "
            "for example --config block_rotatez_sac_her_4m_parallel.json."
        ),
    )
    parser.add_argument(
        "--env-id",
        help="Gymnasium-Robotics env id. Start with RotateZ, then move to XYZ/Full.",
    )
    parser.add_argument(
        "--env-kwargs-json",
        help="JSON object passed to gym.make, overriding env_kwargs from config.",
    )
    parser.add_argument(
        "--algo",
        choices=sorted(SUPPORTED_ALGOS),
        help="SB3 algorithm. SAC/TD3/DDPG can use HER; PPO is on-policy and cannot.",
    )
    parser.add_argument("--policy", help="SB3 policy class name.")
    parser.add_argument(
        "--policy-kwargs-json",
        help="JSON object passed as SB3 policy_kwargs.",
    )
    parser.add_argument(
        "--algo-kwargs-json",
        help="JSON object merged into the SB3 algorithm constructor kwargs.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("env", "isaac"),
        help="Reward source. env keeps environment reward; isaac applies a shaped ShadowHand reward wrapper.",
    )
    parser.add_argument(
        "--reward-kwargs-json",
        help="JSON object passed to the shaped reward wrapper.",
    )
    her_group = parser.add_mutually_exclusive_group()
    her_group.add_argument(
        "--use-her",
        dest="use_her",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable HerReplayBuffer for SAC/TD3/DDPG.",
    )
    her_group.add_argument(
        "--no-her",
        dest="use_her",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable HerReplayBuffer.",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device",
        help="PyTorch device passed to SB3. Use cuda for GPU training, auto for fallback.",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--run-dir",
        help=(
            "Exact per-run export directory. Defaults inside it are logs/, models/, "
            "and tensorboard/."
        ),
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Portable export root for another host. Defaults become "
            "<root>/logs/<run>, <root>/models/<run>, and <root>/tensorboard."
        ),
    )
    parser.add_argument("--log-dir", help="Base log directory, or override under output-root.")
    parser.add_argument(
        "--model-dir",
        help="Base model directory, or override under output-root.",
    )
    parser.add_argument(
        "--tensorboard-log",
        help="TensorBoard directory, or override under output-root.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Number of parallel training environments. 1 keeps the old single-env path.",
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        help="Number of parallel evaluation environments.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("dummy", "subproc"),
        help="Vectorized-env backend when num-envs or eval-num-envs is greater than 1.",
    )
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--vec-normalize",
        dest="vec_normalize",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Wrap vector envs with SB3 VecNormalize.",
    )
    normalize_group.add_argument(
        "--no-vec-normalize",
        dest="vec_normalize",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable SB3 VecNormalize.",
    )
    obs_norm_group = parser.add_mutually_exclusive_group()
    obs_norm_group.add_argument(
        "--norm-obs",
        dest="norm_obs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Normalize observations when VecNormalize is enabled.",
    )
    obs_norm_group.add_argument(
        "--no-norm-obs",
        dest="norm_obs",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Do not normalize observations.",
    )
    reward_norm_group = parser.add_mutually_exclusive_group()
    reward_norm_group.add_argument(
        "--norm-reward",
        dest="norm_reward",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Normalize rewards/returns when VecNormalize is enabled.",
    )
    reward_norm_group.add_argument(
        "--no-norm-reward",
        dest="norm_reward",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Do not normalize rewards/returns.",
    )
    parser.add_argument("--norm-clip-obs", type=float)
    parser.add_argument("--norm-clip-reward", type=float)
    parser.add_argument("--norm-epsilon", type=float)
    parser.add_argument("--buffer-size", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--train-freq", type=int)
    parser.add_argument("--gradient-steps", type=int)
    parser.add_argument("--n-sampled-goal", type=int)
    parser.add_argument(
        "--goal-selection-strategy",
        choices=("future", "final", "episode"),
    )
    parser.add_argument("--n-steps", type=int, help="PPO rollout steps per env.")
    parser.add_argument("--n-epochs", type=int, help="PPO optimization epochs.")
    parser.add_argument("--gae-lambda", type=float, help="PPO GAE lambda.")
    parser.add_argument("--clip-range", type=float, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, help="Entropy coefficient.")
    parser.add_argument("--vf-coef", type=float, help="PPO value-function coefficient.")
    parser.add_argument("--max-grad-norm", type=float, help="PPO gradient clipping norm.")
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--n-eval-episodes", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--verbose", type=int)
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show SB3 progress bar.",
    )
    parser.add_argument(
        "--check-env-only",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Reset/step/render one episode sample and exit before importing SB3.",
    )
    for action in parser._actions:
        if action.help is None:
            action.help = ""
    return parser


def load_json_object(text: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} must be a valid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object.")
    return value


def resolve_config_path(config: str) -> Path:
    candidate = Path(config).expanduser()
    if candidate.exists():
        return candidate

    config_dir_candidate = CONFIG_DIR / config
    if config_dir_candidate.exists():
        return config_dir_candidate

    if candidate.suffix == "":
        for suffix in (".json", ".toml"):
            suffixed = CONFIG_DIR / f"{config}{suffix}"
            if suffixed.exists():
                return suffixed

    raise SystemExit(
        f"Config file not found: {config}. Checked the given path and {CONFIG_DIR}."
    )


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as config_file:
            value = json.load(config_file)
    elif suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "TOML configs require Python 3.11+ tomllib. Use JSON on older Python."
            ) from exc
        with path.open("rb") as config_file:
            value = tomllib.load(config_file)
    else:
        raise SystemExit(f"Unsupported config suffix {suffix!r}; use .json or .toml.")

    if not isinstance(value, dict):
        raise SystemExit(f"Config must contain a JSON/TOML object: {path}")
    return value


def flatten_config(raw_config: Mapping[str, Any], source: Path) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for raw_key, raw_value in raw_config.items():
        key = str(raw_key).replace("-", "_")
        if key in CONFIG_SECTIONS:
            if not isinstance(raw_value, Mapping):
                raise SystemExit(f"Config section {raw_key!r} must be an object: {source}")
            for raw_subkey, sub_value in raw_value.items():
                subkey = str(raw_subkey).replace("-", "_")
                if key in {"env", "environment"} and subkey == "kwargs":
                    flat["env_kwargs"] = sub_value
                elif key == "algorithm" and subkey == "kwargs":
                    flat["algo_kwargs"] = sub_value
                elif key == "algorithm" and subkey == "policy_kwargs":
                    flat["policy_kwargs"] = sub_value
                elif key == "reward" and subkey == "mode":
                    flat["reward_mode"] = sub_value
                elif key == "reward" and subkey == "kwargs":
                    flat["reward_kwargs"] = sub_value
                else:
                    flat[subkey] = sub_value
        else:
            flat[key] = raw_value

    unknown = sorted(set(flat) - set(DEFAULTS))
    if unknown:
        raise SystemExit(
            f"Unknown config key(s) in {source}: {', '.join(unknown)}. "
            "Use CLI-style names such as env_id, num_envs, or learning_rate."
        )
    return flat


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    cli_values = vars(parser.parse_args())

    config_source: Path | None = None
    config_values: dict[str, Any] = {}
    if "config" in cli_values:
        config_source = resolve_config_path(cli_values.pop("config"))
        config_values = flatten_config(load_config(config_source), config_source)

    for cli_key, target_key, label in (
        ("env_kwargs_json", "env_kwargs", "--env-kwargs-json"),
        ("policy_kwargs_json", "policy_kwargs", "--policy-kwargs-json"),
        ("algo_kwargs_json", "algo_kwargs", "--algo-kwargs-json"),
        ("reward_kwargs_json", "reward_kwargs", "--reward-kwargs-json"),
    ):
        if cli_key in cli_values:
            cli_values[target_key] = load_json_object(cli_values.pop(cli_key), label)

    args_dict = copy.deepcopy(DEFAULTS)
    args_dict.update(config_values)
    args_dict.update(cli_values)
    explicit_keys = set(config_values) | set(cli_values)
    args_dict["config"] = str(config_source) if config_source else None
    args_dict["_explicit_keys"] = sorted(explicit_keys)
    normalize_args(args_dict)
    return argparse.Namespace(**args_dict)


def require_mapping(value: Any, key: str) -> None:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{key} must be an object/dict, got {type(value).__name__}.")


def normalize_args(args: dict[str, Any]) -> None:
    args["algo"] = str(args["algo"]).lower()
    args["vec_env"] = str(args["vec_env"]).lower()
    args["reward_mode"] = str(args["reward_mode"]).lower()
    args["goal_selection_strategy"] = str(args["goal_selection_strategy"]).lower()

    if args["algo"] not in SUPPORTED_ALGOS:
        raise SystemExit(
            f"Unsupported --algo {args['algo']!r}. Choose one of {sorted(SUPPORTED_ALGOS)}."
        )
    if args["vec_env"] not in {"dummy", "subproc"}:
        raise SystemExit("--vec-env must be either dummy or subproc.")
    if args["reward_mode"] not in {"env", "isaac"}:
        raise SystemExit("--reward-mode must be either env or isaac.")
    if args["num_envs"] < 1:
        raise SystemExit("--num-envs must be >= 1.")
    if args["eval_num_envs"] < 1:
        raise SystemExit("--eval-num-envs must be >= 1.")
    if args["timesteps"] < 1:
        raise SystemExit("--timesteps must be >= 1.")
    if args["batch_size"] < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args["eval_freq"] < 0 or args["save_freq"] < 0:
        raise SystemExit("--eval-freq and --save-freq must be >= 0.")
    if args["norm_clip_obs"] <= 0 or args["norm_clip_reward"] <= 0:
        raise SystemExit("--norm-clip-obs and --norm-clip-reward must be > 0.")
    if args["norm_epsilon"] <= 0:
        raise SystemExit("--norm-epsilon must be > 0.")

    require_mapping(args["env_kwargs"], "env_kwargs")
    require_mapping(args["policy_kwargs"], "policy_kwargs")
    require_mapping(args["algo_kwargs"], "algo_kwargs")
    require_mapping(args["reward_kwargs"], "reward_kwargs")

    if args["use_her"] and args["algo"] not in HER_ALGOS:
        print(
            f"warning: {args['algo'].upper()} is on-policy and cannot use HER; "
            "disabling use_her.",
            flush=True,
        )
        args["use_her"] = False


def register_envs() -> None:
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)


def apply_reward_wrapper(env: gym.Env, args: argparse.Namespace) -> gym.Env:
    from shadowhand_wrappers import apply_shadowhand_reward_wrapper

    return apply_shadowhand_reward_wrapper(
        env,
        args.reward_mode,
        dict(args.reward_kwargs),
    )


def make_single_env(
    args: argparse.Namespace,
    seed: int,
    monitor_file: Path | None = None,
) -> gym.Env:
    env = gym.make(args.env_id, **dict(args.env_kwargs))
    env = apply_reward_wrapper(env, args)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    if monitor_file is not None:
        from stable_baselines3.common.monitor import Monitor

        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_file), info_keywords=("is_success",))
    return env


def make_env_factory(
    args: argparse.Namespace,
    seed: int,
    rank: int,
):
    def _init() -> gym.Env:
        register_envs()
        env_seed = seed + rank
        env = gym.make(args.env_id, **dict(args.env_kwargs))
        env = apply_reward_wrapper(env, args)
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        return env

    return _init


def make_training_env(
    args: argparse.Namespace,
    num_envs: int,
    seed: int,
    monitor_file: Path,
    stack: Mapping[str, Any] | None = None,
    is_eval: bool = False,
):
    if num_envs == 1 and not args.vec_normalize:
        return make_single_env(args, seed, monitor_file)

    if stack is None:
        raise RuntimeError("Vectorized environments require the SB3 training stack.")

    env_fns = [
        make_env_factory(args, seed, rank)
        for rank in range(num_envs)
    ]
    if args.vec_env == "dummy" or num_envs == 1:
        vec_env = stack["DummyVecEnv"](env_fns)
    else:
        vec_env = stack["SubprocVecEnv"](env_fns)

    monitor_file.parent.mkdir(parents=True, exist_ok=True)
    vec_env = stack["VecMonitor"](
        vec_env,
        filename=str(monitor_file),
        info_keywords=("is_success",),
    )
    if args.vec_normalize:
        vec_env = stack["VecNormalize"](
            vec_env,
            training=not is_eval,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_reward if not is_eval else False,
            clip_obs=args.norm_clip_obs,
            clip_reward=args.norm_clip_reward,
            gamma=args.gamma,
            epsilon=args.norm_epsilon,
        )
    return vec_env


def import_training_stack() -> dict[str, Any]:
    try:
        import torch
        from stable_baselines3 import DDPG, PPO, SAC, TD3, HerReplayBuffer
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecMonitor,
            VecNormalize,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install PyTorch with a CUDA wheel first, then "
            "`pip install 'stable-baselines3[extra]'` in a Python 3.10+ environment."
        ) from exc

    return {
        "torch": torch,
        "algorithms": {
            "ddpg": DDPG,
            "ppo": PPO,
            "sac": SAC,
            "td3": TD3,
        },
        "HerReplayBuffer": HerReplayBuffer,
        "CallbackList": CallbackList,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
        "VecMonitor": VecMonitor,
        "VecNormalize": VecNormalize,
    }


def check_env(args: argparse.Namespace) -> None:
    env = gym.make(args.env_id, render_mode="rgb_array", **dict(args.env_kwargs))
    env = apply_reward_wrapper(env, args)
    obs, _ = env.reset(seed=args.seed)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frame = env.render()
    print("env_id:", args.env_id)
    print("env_kwargs:", dict(args.env_kwargs))
    print("reward_mode:", args.reward_mode)
    print("action_space:", env.action_space)
    print("obs_shapes:", {key: value.shape for key, value in obs.items()})
    print("step:", reward, terminated, truncated, info)
    # print("frame:", frame.shape, frame.dtype, int(frame.min()), int(frame.max()))
    env.close()


def resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    suffix = args.algo
    if args.use_her and args.algo in HER_ALGOS:
        suffix = f"{suffix}_her"
    return f"{args.env_id}_{suffix}_seed{args.seed}"


def resolve_output_paths(args: argparse.Namespace) -> dict[str, Any]:
    run_name = resolve_run_name(args)
    explicit = set(args._explicit_keys)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser()
        log_dir = Path(args.log_dir).expanduser() if "log_dir" in explicit else run_dir / "logs"
        model_dir = (
            Path(args.model_dir).expanduser()
            if "model_dir" in explicit
            else run_dir / "models"
        )
        tb_dir = (
            Path(args.tensorboard_log).expanduser()
            if "tensorboard_log" in explicit
            else run_dir / "tensorboard"
        )
    elif args.output_root:
        output_root = Path(args.output_root).expanduser()
        log_base = (
            Path(args.log_dir).expanduser()
            if "log_dir" in explicit
            else output_root / "logs"
        )
        model_base = (
            Path(args.model_dir).expanduser()
            if "model_dir" in explicit
            else output_root / "models"
        )
        tb_dir = (
            Path(args.tensorboard_log).expanduser()
            if "tensorboard_log" in explicit
            else output_root / "tensorboard"
        )
        log_dir = log_base / run_name
        model_dir = model_base / run_name
    else:
        log_dir = Path(args.log_dir).expanduser() / run_name
        model_dir = Path(args.model_dir).expanduser() / run_name
        tb_dir = Path(args.tensorboard_log).expanduser()

    return {
        "run_name": run_name,
        "log_dir": log_dir,
        "model_dir": model_dir,
        "tensorboard_log": tb_dir,
        "resolved": {
            "log_dir": str(log_dir.resolve()),
            "model_dir": str(model_dir.resolve()),
            "tensorboard_log": str(tb_dir.resolve()),
        },
    }


def callback_freq(total_step_freq: int, num_envs: int) -> int:
    if total_step_freq <= 0:
        return 0
    return max(total_step_freq // max(num_envs, 1), 1)


def resolve_policy_kwargs(policy_kwargs: Mapping[str, Any], torch_module: Any) -> dict[str, Any]:
    kwargs = dict(policy_kwargs)
    activation_fn = kwargs.get("activation_fn")
    if isinstance(activation_fn, str):
        activation_map = {
            "elu": torch_module.nn.ELU,
            "leaky_relu": torch_module.nn.LeakyReLU,
            "relu": torch_module.nn.ReLU,
            "tanh": torch_module.nn.Tanh,
        }
        key = activation_fn.lower()
        if key not in activation_map:
            raise SystemExit(
                f"Unsupported policy activation_fn {activation_fn!r}. "
                f"Choose one of {sorted(activation_map)}."
            )
        kwargs["activation_fn"] = activation_map[key]
    return kwargs


def build_model_kwargs(
    args: argparse.Namespace,
    stack: Mapping[str, Any],
    tensorboard_log: Path,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "tensorboard_log": str(tensorboard_log),
        "device": args.device,
        "verbose": args.verbose,
        "seed": args.seed,
    }
    if args.policy_kwargs:
        kwargs["policy_kwargs"] = resolve_policy_kwargs(
            args.policy_kwargs,
            stack["torch"],
        )

    if args.algo in OFF_POLICY_ALGOS:
        kwargs.update(
            {
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "tau": args.tau,
                "train_freq": args.train_freq,
                "gradient_steps": args.gradient_steps,
            }
        )
        if args.use_her:
            kwargs.update(
                {
                    "replay_buffer_class": stack["HerReplayBuffer"],
                    "replay_buffer_kwargs": {
                        "n_sampled_goal": args.n_sampled_goal,
                        "goal_selection_strategy": args.goal_selection_strategy,
                    },
                }
            )
    elif args.algo == "ppo":
        kwargs.update(
            {
                "batch_size": args.batch_size,
                "n_steps": args.n_steps,
                "n_epochs": args.n_epochs,
                "gae_lambda": args.gae_lambda,
                "clip_range": args.clip_range,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
                "max_grad_norm": args.max_grad_norm,
            }
        )

    kwargs.update(dict(args.algo_kwargs))
    return kwargs


def serializable_config(
    args: argparse.Namespace,
    paths: Mapping[str, Any],
    effective_eval_freq: int,
    effective_save_freq: int,
) -> dict[str, Any]:
    data = {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "config"
    }
    data.update(
        {
            "config_source": args.config,
            "command": shlex.join(sys.argv),
            "resolved_paths": paths["resolved"],
            "vecnormalize_path": (
                str((paths["model_dir"] / "vecnormalize.pkl").resolve())
                if args.vec_normalize
                else None
            ),
            "effective_eval_freq": effective_eval_freq,
            "effective_save_freq": effective_save_freq,
        }
    )
    return data


def main() -> None:
    args = parse_args()
    register_envs()

    if args.check_env_only:
        check_env(args)
        return

    stack = import_training_stack()
    torch = stack["torch"]
    paths = resolve_output_paths(args)
    run_name = paths["run_name"]
    log_dir = paths["log_dir"]
    model_dir = paths["model_dir"]
    tb_dir = paths["tensorboard_log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    print("algo:", args.algo.upper())
    print("use_her:", args.use_her)
    print("reward_mode:", args.reward_mode)
    print("num_envs:", args.num_envs, f"({args.vec_env})")
    print("eval_num_envs:", args.eval_num_envs)
    print("vec_normalize:", args.vec_normalize)
    print("log_dir:", paths["resolved"]["log_dir"])
    print("model_dir:", paths["resolved"]["model_dir"])
    print("tensorboard_log:", paths["resolved"]["tensorboard_log"])

    env = make_training_env(
        args,
        args.num_envs,
        args.seed,
        log_dir / "train_monitor.csv",
        stack=stack,
        is_eval=False,
    )
    eval_env = make_training_env(
        args,
        args.eval_num_envs,
        args.seed + 10_000,
        log_dir / "eval_monitor.csv",
        stack=stack,
        is_eval=True,
    )

    effective_eval_freq = callback_freq(args.eval_freq, args.num_envs)
    effective_save_freq = callback_freq(args.save_freq, args.num_envs)
    callbacks = []
    if effective_eval_freq > 0:
        callbacks.append(
            stack["EvalCallback"](
                eval_env,
                best_model_save_path=str(model_dir),
                log_path=str(log_dir),
                eval_freq=effective_eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )
    if effective_save_freq > 0:
        callbacks.append(
            stack["CheckpointCallback"](
                save_freq=effective_save_freq,
                save_path=str(model_dir),
                name_prefix="checkpoint",
            )
        )
    callback = stack["CallbackList"](callbacks) if callbacks else None

    with (model_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            serializable_config(args, paths, effective_eval_freq, effective_save_freq),
            config_file,
            indent=2,
            sort_keys=True,
        )

    algo_cls = stack["algorithms"][args.algo]
    model = algo_cls(
        args.policy,
        env,
        **build_model_kwargs(args, stack, tb_dir),
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
        print("saved:", final_path.with_suffix(".zip").resolve())
        if args.vec_normalize and isinstance(env, stack["VecNormalize"]):
            vecnormalize_path = model_dir / "vecnormalize.pkl"
            env.save(str(vecnormalize_path))
            print("saved:", vecnormalize_path.resolve())
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
