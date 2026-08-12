"""Generate a behavior-cloning dataset from rule-based bot self-play.

Player 0 is the demonstrator (ExpanderAgent or HumanExeAgent), player 1 is a bot
opponent. For each step we record the player-0 (observation, valid-move-mask,
action) tuple in the exact format the RL policy consumes, so the data can warm-start
PPO via supervised pretraining (train/pretrain.py). Observations are the
31-channel memory-augmented tensor (paper arXiv:2507.06825) that the policy is
trained on.

Data is generated with the patched generals.io army growth (FastGame) so it
matches the RL training dynamics. Only steps whose demonstrator action is valid
(per the move mask) are kept, matching the paper's "behavioral cloning on valid
moves only".

Usage:
    python train/generate_data.py --out checkpoints/dataset.npz --games 120
    python train/generate_data.py --demonstrator human_exe --games 50 \
        --grid-min 18 --grid-max 20 --pad-to 24 --out checkpoints/dataset_hx.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.game_patch import FastGame
from agents.memory_aug import MemoryAugmenter
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


def _is_valid_action(action, mask) -> bool:
    """True if `action` ([pass,row,col,dir,split]) is legal under the (P,P,4) mask."""
    action = np.asarray(action, dtype=np.int32)
    if action[0] == 1:  # pass is always legal
        return True
    r, c, d = int(action[1]), int(action[2]), int(action[3])
    if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]:
        return False
    return bool(mask[r, c, d])


def _save(out: str, obs, masks, actions, pad_to: int, n: int) -> None:
    """Write a (possibly partial) dataset checkpoint to `out`."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=obs[:n], masks=masks[:n], actions=actions[:n], pad_to=pad_to)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="checkpoints/dataset.npz")
    p.add_argument("--games", type=int, default=120)
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--grid-min", type=int, default=None)
    p.add_argument("--grid-max", type=int, default=None)
    p.add_argument("--pad-to", type=int, default=None)
    p.add_argument("--truncation", type=int, default=None)
    p.add_argument("--demonstrator", type=str, default="expander",
                   choices=["expander", "human_exe"])
    p.add_argument("--save-every", type=int, default=10,
                   help="save a checkpoint every N games (crash-safe; default 10)")
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
    opponents = [ExpanderAgent("B"), RandomAgent("B")]

    # Preallocate (memory-safe up to max_samples; ~28 KB per sample at P=24, 31 ch).
    max_samples = args.max_samples
    obs_arr = np.zeros((max_samples, config.INPUT_CHANNELS, P, P), dtype=np.float32)
    mask_arr = np.zeros((max_samples, P, P, 4), dtype=bool)
    act_arr = np.zeros((max_samples,), dtype=np.int64)

    augmenter = MemoryAugmenter(history_size=config.MEMORY_HISTORY_SIZE)

    n = 0
    t0 = time.time()
    for g in range(args.games):
        # Fresh stateful demonstrator per game.
        if args.demonstrator == "human_exe":
            from agents.human_exe_agent import HumanExeAgent
            demo = HumanExeAgent(player_id="A")
        else:
            demo = ExpanderAgent("A")
        opp = opponents[g % len(opponents)]
        grid = grid_factory.generate()
        game = FastGame(grid, ["A", "B"])
        opp.reset()
        augmenter._mem = {}  # fresh memory per game

        for _ in range(T):
            oa = game.agent_observation("A")
            ob = game.agent_observation("B")
            if args.demonstrator == "human_exe":
                act_a = demo.act_on_game(game, "A")
            else:
                act_a = demo.act(oa)
            act_b = opp.act(ob)

            obs_t = np.asarray(oa.as_tensor(pad_to=P), dtype=np.float32)
            mask_t = np.asarray(compute_valid_move_mask(oa), dtype=bool)
            if not _is_valid_action(act_a, mask_t):
                continue  # keep valid demonstrator moves only
            with torch.no_grad():
                obs_aug = augmenter.augment_observation(torch.from_numpy(obs_t[None]))
            act_idx = int(encode_action(np.asarray(act_a, dtype=np.int32), P))

            obs_arr[n] = obs_aug[0].numpy()
            mask_arr[n] = mask_t
            act_arr[n] = act_idx
            n += 1

            game.step({"A": act_a, "B": act_b})
            if game.is_done() or n >= max_samples:
                break

        if n >= max_samples:
            print("Reached max_samples; stopping early.")
            break
        # Show per-game progress (human_exe runs ~1-2 min/game; every 20 games for the
        # fast expander demonstrator).
        if (g + 1) % 20 == 0 or args.demonstrator == "human_exe":
            dt = time.time() - t0
            rate = n / max(dt, 1e-6)
            print(f"[{args.demonstrator}] game {g + 1}/{args.games}: {n} samples "
                  f"({dt:.0f}s, {rate:.1f} samples/s)")
        # Crash-safe checkpoint: overwrite --out every `--save-every` games so a crash
        # only loses the last few games instead of the whole run.
        if (g + 1) % args.save_every == 0:
            _save(args.out, obs_arr, mask_arr, act_arr, P, n)
            print(f"  checkpoint saved ({n} samples so far) -> {args.out}")

    _save(args.out, obs_arr, mask_arr, act_arr, P, n)
    print(f"Saved {n} samples to {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
