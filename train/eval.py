"""Evaluate a trained Generals.io policy against fixed opponents."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generals_env import GeneralsEnv
from agents.ppo_agent import PPOAgent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.grid import GridFactory


def make_grid_factory(cfg) -> GridFactory:
    return GridFactory(
        mode=cfg.GRID_MODE,
        min_grid_dims=cfg.MIN_GRID_DIMS,
        max_grid_dims=cfg.MAX_GRID_DIMS,
        mountain_density=cfg.MOUNTAIN_DENSITY,
        city_density=cfg.CITY_DENSITY,
    )


def evaluate(network, cfg, num_games: int = 20, opponents=None, device="cpu",
             greedy: bool = False, seed: int = 0) -> dict:
    """Play `num_games` against each opponent; return per-opponent stats.

    The policy is player 0, the opponent is player 1.
    """
    policy = PPOAgent(network, device=device, pad_to=cfg.PAD_TO, greedy=greedy)
    if opponents is None:
        opponents = [RandomAgent(), ExpanderAgent()]

    rng = np.random.default_rng(seed)
    results = {}
    for opp in opponents:
        wins = losses = draws = 0
        lengths = []
        for _ in range(num_games):
            env = GeneralsEnv(opponent=opp, grid_factory=make_grid_factory(cfg),
                              pad_observations_to=cfg.PAD_TO, truncation=cfg.TRUNCATION)
            env.reset(seed=int(rng.integers(0, 2**31)))
            done = False
            length = 0
            while not done:
                a0 = policy.act_batched(env.last_obs[None], env.last_mask[None])[0]
                a1 = opp.act(env.opponent_observation())
                _, _, term, trunc, info0 = env.step(a0, a1)
                done = term or trunc
                length += 1
                if length > cfg.TRUNCATION + 5:
                    break
            if term:
                if bool(info0["winner"]):
                    wins += 1
                else:
                    losses += 1
            else:
                draws += 1
            lengths.append(length)
            env.close()

        total = wins + losses + draws
        results[str(opp)] = {
            "win_rate": wins / max(total, 1),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        }
    return results
