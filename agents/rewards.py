"""Potential-based reward shaping for Generals.io.

Ported from the generals-bots paper's `composite_reward_fn` (JAX) to numpy.
All functions operate on the 15-channel padded observation tensor
(15, H, W) that generals-bots produces, or on a batch (B, 15, H, W).

Channel indices:
    2  cities mask
    5  owned_cells mask
    6  opponent_cells mask
    9  owned_land_count   (broadcast scalar)
    10 owned_army_count   (broadcast scalar)
    11 opponent_land_count (broadcast scalar)
    12 opponent_army_count (broadcast scalar)

The potential is a RELATIVE-advantage proxy. The original port rewarded only
the agent's own absolute growth (`log1p(my_land) + log1p(my_army)`); that
saturates in the late game (slope 1/x) and even biases AGAINST attacking,
because army spent in combat lowers the potential while the terminal +-1 is
diluted to ~0.02 by gamma over a long game. Subtracting the opponent's counts
makes every successful strike — my army/land up OR the opponent's down —
raise the potential at any game stage, keeping a dense per-step gradient
through the endgame.

The terminal (+1/-1) reward is handled by the training loop from the env
winner info; these functions only shape the non-terminal steps.
"""
from __future__ import annotations

import numpy as np

# Weights (ratio + absolute potentials combined; see `potentials`).
MAX_ARMY_RATIO = 1.6
MAX_LAND_RATIO = 1.3
RATIO_WEIGHT = 0.2
ABS_WEIGHT = 0.1
# Opponent-decline terms are weighted 2x the own-growth terms, so reducing the
# opponent (attacks, taking their land) pays more than endlessly expanding into
# neutral ground — on big maps the own-growth terms reward expansion forever,
# which stalls the game. Keep in sync with config.ABS_OPP_WEIGHT.
ABS_OPP_WEIGHT = 0.2
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
    """Number of cities (castles) the agent owns (visible)."""
    return np.sum(obs_batch[:, 2] * obs_batch[:, 5], axis=(1, 2))


def castles_owned_opp(obs_batch: np.ndarray) -> np.ndarray:
    """Number of cities the opponent owns (visible)."""
    return np.sum(obs_batch[:, 2] * obs_batch[:, 6], axis=(1, 2))


def potentials(obs_batch: np.ndarray) -> np.ndarray:
    """Shaping potential for a batch of observations (B, 15, H, W).

    Combines RELATIVE (ratio) and RELATIVE-LOG (advantage) terms so the signal
    stays dense in the mid/late game:
      ratio    = RATIO_WEIGHT * (army_ratio_pot + land_ratio_pot)
      absolute = ABS_WEIGHT * (log1p(my_land)  - log1p(opp_land)
                             + log1p(my_army)  - log1p(opp_army))
      castles  = CASTLE_WEIGHT * (castles_mine - castles_opp)
    The absolute terms are a log-ratio advantage proxy (no clamp, so they
    never pin): capturing land or a city raises it, and reducing the
    opponent's army/land raises it too — a successful attack is rewarded at
    any game stage. Net cities make denying the opponent's cities valuable
    in the endgame.
    """
    ratio = RATIO_WEIGHT * (army_ratio_potential(obs_batch) + land_ratio_potential(obs_batch))
    # Own growth and opponent decline use separate weights; ABS_OPP_WEIGHT is
    # higher so hurting the opponent pays more than expanding.
    absolute = (
        ABS_WEIGHT * (np.log1p(obs_batch[:, 9, 0, 0]) + np.log1p(obs_batch[:, 10, 0, 0]))
        - ABS_OPP_WEIGHT * (np.log1p(obs_batch[:, 11, 0, 0]) + np.log1p(obs_batch[:, 12, 0, 0]))
    )
    castles = CASTLE_WEIGHT * (castles_owned(obs_batch) - castles_owned_opp(obs_batch))
    return ratio + absolute + castles


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
