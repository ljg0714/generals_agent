"""Potential-based reward shaping for Generals.io.

Ported from the generals-bots paper's `composite_reward_fn` (JAX) to numpy.
All functions operate on the 15-channel padded observation tensor
(15, H, W) that generals-bots produces, or on a batch (B, 15, H, W).

Channel indices:
    2  cities mask
    5  owned_cells mask
    9  owned_land_count   (broadcast scalar)
    10 owned_army_count   (broadcast scalar)
    11 opponent_land_count (broadcast scalar)
    12 opponent_army_count (broadcast scalar)

The terminal (+1/-1) reward is handled by the training loop from the env
winner info; these functions only shape the non-terminal steps.
"""
from __future__ import annotations

import numpy as np

# Weights (match the generals-bots composite_reward_fn defaults)
MAX_ARMY_RATIO = 1.6
MAX_LAND_RATIO = 1.3
RATIO_WEIGHT = 0.3
CASTLE_WEIGHT = 0.4


def _ratio_potential(mine: np.ndarray, opponent: np.ndarray, max_ratio: float) -> np.ndarray:
    """Clamped log ratio of my/opponent resource, mapped to [-1, 1]."""
    ratio = mine / np.maximum(opponent, 1.0)
    r = np.log(np.maximum(ratio, 1e-8)) / np.log(max_ratio)
    return np.clip(r, -1.0, 1.0)


def army_ratio_potential(obs_batch: np.ndarray) -> np.ndarray:
    return _ratio_potential(obs_batch[:, 10, 0, 0], obs_batch[:, 12, 0, 0], MAX_ARMY_RATIO)


def land_ratio_potential(obs_batch: np.ndarray) -> np.ndarray:
    return _ratio_potential(obs_batch[:, 9, 0, 0], obs_batch[:, 11, 0, 0], MAX_LAND_RATIO)


def castles_owned(obs_batch: np.ndarray) -> np.ndarray:
    """Number of cities (castles) the agent owns."""
    return np.sum(obs_batch[:, 2] * obs_batch[:, 5], axis=(1, 2))


def potentials(obs_batch: np.ndarray) -> np.ndarray:
    """Shaping potential for a batch of observations (B, 15, H, W)."""
    return (
        RATIO_WEIGHT * (army_ratio_potential(obs_batch) + land_ratio_potential(obs_batch))
        + CASTLE_WEIGHT * castles_owned(obs_batch)
    )


def composite_reward(prior_obs: np.ndarray, obs: np.ndarray, terminated: np.ndarray) -> np.ndarray:
    """Shaped reward (difference of potentials) for non-terminal steps.

    Args:
        prior_obs: (B, 15, H, W) observation before the step.
        obs: (B, 15, H, W) observation after the step.
        terminated: (B,) bool — True steps get 0 shaping (terminal reward
            is handled separately by the training loop).

    Returns:
        (B,) float32 shaped reward.
    """
    prior_pot = potentials(prior_obs)
    pot = potentials(obs)
    shaped = pot - prior_pot
    return np.where(np.asarray(terminated, dtype=bool), 0.0, shaped).astype(np.float32)
