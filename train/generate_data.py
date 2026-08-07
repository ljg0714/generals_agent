"""Generate a behavior-cloning dataset from rule-based bot self-play.

Player 0 is the demonstrator (ExpanderAgent), player 1 is a bot opponent.
For each step we record the player-0 (observation, valid-move-mask, action)
tuple in the exact format the RL policy consumes, so the data can warm-start
PPO via supervised pretraining (train/pretrain.py).

Data is generated with the patched generals.io army growth (FastGame) so it
matches the RL training dynamics.

Usage:
    python train/generate_data.py --out checkpoints/dataset.npz --games 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.game_patch import FastGame
from agents.network import encode_action
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.action import compute_valid_move_mask
from generals.core.grid import GridFactory


def make_grid_factory(cfg) -> GridFactory:
    return GridFactory(
        mode=cfg.GRID_MODE,
        min_grid_dims=cfg.MIN_GRID_DIMS,
        max_grid_dims=cfg.MAX_GRID_DIMS,
        mountain_density=cfg.MOUNTAIN_DENSITY,
        city_density=cfg.CITY_DENSITY,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="checkpoints/dataset.npz")
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--grid-min", type=int, default=None)
    p.add_argument("--grid-max", type=int, default=None)
    p.add_argument("--pad-to", type=int, default=None)
    p.add_argument("--truncation", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.grid_min or args.grid_max:
        gmin = args.grid_min if args.grid_min is not None else args.grid_max
        gmax = args.grid_max if args.grid_max is not None else args.grid_min
        config.MIN_GRID_DIMS = (gmin, gmin)
        config.MAX_GRID_DIMS = (gmax, gmax)
    P = args.pad_to or config.PAD_TO
    T = args.truncation or config.TRUNCATION

    rng = np.random.default_rng(args.seed)
    grid_factory = make_grid_factory(config)
    demonstrators = [ExpanderAgent("A")]
    opponents = [ExpanderAgent("B"), RandomAgent("B")]

    # Preallocate (memory-safe up to max_samples; ~35 KB per sample at P=24).
    max_samples = args.max_samples
    obs_arr = np.zeros((max_samples, 15, P, P), dtype=np.float32)
    mask_arr = np.zeros((max_samples, P, P, 4), dtype=bool)
    act_arr = np.zeros((max_samples,), dtype=np.int64)

    n = 0
    t0 = time.time()
    for g in range(args.games):
        demo = demonstrators[g % len(demonstrators)]
        opp = opponents[g % len(opponents)]
        grid = grid_factory.generate()
        game = FastGame(grid, ["A", "B"])
        demo.reset()
        opp.reset()

        for _ in range(T):
            oa = game.agent_observation("A")
            ob = game.agent_observation("B")
            act_a = demo.act(oa)
            act_b = opp.act(ob)

            obs_t = np.asarray(oa.as_tensor(pad_to=P), dtype=np.float32)
            mask_t = np.asarray(compute_valid_move_mask(oa), dtype=bool)
            act_idx = int(encode_action(np.asarray(act_a, dtype=np.int32), P))

            obs_arr[n] = obs_t
            mask_arr[n] = mask_t
            act_arr[n] = act_idx
            n += 1

            game.step({"A": act_a, "B": act_b})
            if game.is_done() or n >= max_samples:
                break

        if n >= max_samples:
            print("Reached max_samples; stopping early.")
            break
        if (g + 1) % 20 == 0:
            print(f"game {g + 1}: {n} samples ({time.time() - t0:.0f}s)")

    obs_arr, mask_arr, act_arr = obs_arr[:n], mask_arr[:n], act_arr[:n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=obs_arr, masks=mask_arr, actions=act_arr, pad_to=P)
    print(f"Saved {n} samples to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
