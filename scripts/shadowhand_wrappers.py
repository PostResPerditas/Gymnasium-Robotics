"""Local wrappers for ShadowHand training and playback."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def _quat_conjugate_wxyz(quat: np.ndarray) -> np.ndarray:
    result = np.array(quat, dtype=np.float64, copy=True)
    result[..., 1:] *= -1.0
    return result


def _quat_mul_wxyz(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = np.moveaxis(quat_a, -1, 0)
    w1, x1, y1, z1 = np.moveaxis(quat_b, -1, 0)
    return np.stack(
        (
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ),
        axis=-1,
    )


def _goal_distance(achieved_goal: np.ndarray, desired_goal: np.ndarray) -> tuple[float, float]:
    achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
    desired_goal = np.asarray(desired_goal, dtype=np.float64)
    d_pos = float(np.linalg.norm(achieved_goal[..., :3] - desired_goal[..., :3]))

    quat_diff = _quat_mul_wxyz(
        achieved_goal[..., 3:7],
        _quat_conjugate_wxyz(desired_goal[..., 3:7]),
    )
    # Quaternions q and -q describe the same pose; abs() avoids a spurious 2*pi path.
    d_rot = float(2.0 * np.arccos(np.clip(np.abs(quat_diff[..., 0]), -1.0, 1.0)))
    return d_pos, d_rot


class IsaacStyleShadowHandReward(gym.Wrapper):
    """Replace sparse GoalEnv reward with IsaacGym-style shaped rewards.

    Gymnasium-Robotics ShadowHand goals are encoded as position + wxyz quaternion.
    This wrapper keeps observations and dynamics unchanged; it only rewrites reward
    and adds a few scalar diagnostics to ``info``.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        dist_reward_scale: float = -10.0,
        rot_reward_scale: float = 1.0,
        rot_eps: float = 0.1,
        action_penalty_scale: float = -0.0002,
        reach_goal_bonus: float = 250.0,
        fall_distance: float = 0.24,
        fall_penalty: float = 0.0,
        success_tolerance: float = 0.1,
        reward_scale: float = 0.01,
        require_position_success: bool = False,
        position_success_tolerance: float = 0.01,
    ):
        super().__init__(env)
        self.dist_reward_scale = float(dist_reward_scale)
        self.rot_reward_scale = float(rot_reward_scale)
        self.rot_eps = float(rot_eps)
        self.action_penalty_scale = float(action_penalty_scale)
        self.reach_goal_bonus = float(reach_goal_bonus)
        self.fall_distance = float(fall_distance)
        self.fall_penalty = float(fall_penalty)
        self.success_tolerance = float(success_tolerance)
        self.reward_scale = float(reward_scale)
        self.require_position_success = bool(require_position_success)
        self.position_success_tolerance = float(position_success_tolerance)

    def step(self, action: Any):
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        reward, reward_info = self._compute_reward(obs, action)
        shaped_info = dict(info)
        shaped_info.update(reward_info)
        return obs, reward, terminated, truncated, shaped_info

    def _compute_reward(self, obs: Any, action: Any) -> tuple[float, dict[str, float]]:
        if not isinstance(obs, dict) or "achieved_goal" not in obs or "desired_goal" not in obs:
            raise RuntimeError("IsaacStyleShadowHandReward requires a GoalEnv dict observation.")

        goal_dist, rot_dist = _goal_distance(obs["achieved_goal"], obs["desired_goal"])
        action_penalty = float(np.sum(np.square(action)))
        dist_reward = goal_dist * self.dist_reward_scale
        rot_reward = self.rot_reward_scale / (abs(rot_dist) + self.rot_eps)
        raw_reward = dist_reward + rot_reward + action_penalty * self.action_penalty_scale

        rot_success = abs(rot_dist) <= self.success_tolerance
        pos_success = goal_dist <= self.position_success_tolerance
        success = rot_success and (pos_success if self.require_position_success else True)
        if success:
            raw_reward += self.reach_goal_bonus
        if goal_dist >= self.fall_distance:
            raw_reward += self.fall_penalty

        return raw_reward * self.reward_scale, {
            "is_success": float(success),
            "goal_dist": goal_dist,
            "rot_dist": rot_dist,
            "raw_shaped_reward": raw_reward,
        }


def apply_shadowhand_reward_wrapper(
    env: gym.Env,
    reward_mode: str,
    reward_kwargs: dict[str, Any] | None = None,
) -> gym.Env:
    if reward_mode == "env":
        return env
    if reward_mode == "isaac":
        return IsaacStyleShadowHandReward(env, **(reward_kwargs or {}))
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")
