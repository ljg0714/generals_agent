"""Smoke test: HumanExeAgent plays full games in the gymnasium env.

Plays the EklipZ bot (as player "A") against RandomAgent / ExpanderAgent, checks
every returned action is valid (mask-legal, clamped to pass otherwise), and
reports per-game results. The bot is slow (~0.2-1s per move), so the default is a
short smoke; use --games / --grid-* for longer runs.

Usage:
    python verify_human_exe.py
    python verify_human_exe.py --games 5 --grid-min 18 --grid-max 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config
from agents.game_patch import FastGame
from agents.human_exe_agent import HumanExeAgent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.action import compute_valid_move_mask
from generals.core.grid import GridFactory


def _is_action_valid(action, mask) -> bool:
    if int(action[0]) == 1:
        return True
    r, c, d = int(action[1]), int(action[2]), int(action[3])
    if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]:
        return False
    return bool(mask[r, c, d])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=3, help="games per opponent")
    p.add_argument("--grid-min", type=int, default=12)
    p.add_argument("--grid-max", type=int, default=14)
    p.add_argument("--truncation", type=int, default=None)
    args = p.parse_args()

    T = args.truncation or config.TRUNCATION
    grid_factory = GridFactory(
        mode="uniform", min_grid_dims=(args.grid_min, args.grid_min),
        max_grid_dims=(args.grid_max, args.grid_max),
        mountain_density=config.MOUNTAIN_DENSITY, city_density=config.CITY_DENSITY,
    )

    for opp_cls, name in [(RandomAgent, "random"), (ExpanderAgent, "expander")]:
        wins = draws = losses = invalid = 0
        lengths = []
        t0 = time.time()
        for _ in range(args.games):
            grid = grid_factory.generate()
            game = FastGame(grid, ["A", "B"])
            hx = HumanExeAgent(player_id="A")
            opp = opp_cls("B")
            steps = 0
            while not game.is_done() and steps < T:
                oa = game.agent_observation("A")
                ob = game.agent_observation("B")
                act_a = hx.act_on_game(game, "A")
                act_b = opp.act(ob)
                oa.as_tensor(pad_to=config.PAD_TO)  # pads oa in place
                mask = compute_valid_move_mask(oa)
                if not _is_action_valid(act_a, mask):
                    invalid += 1
                game.step({"A": act_a, "B": act_b})
                steps += 1
            lengths.append(steps)
            info = game.get_infos()
            result = "WIN" if info["A"]["is_winner"] else ("LOSS" if info["B"]["is_winner"] else "DRAW")
            if info["A"]["is_winner"]:
                wins += 1
            elif info["B"]["is_winner"]:
                losses += 1
            else:
                draws += 1
            print(f"  human_exe vs {name}: {result} after {steps} steps "
                  f"({time.time() - t0:.0f}s elapsed)")
            hx.reset()
        total = wins + draws + losses
        print(f"human_exe vs {name}: {wins}/{total} wins, draws {draws}, "
              f"avg_len {np.mean(lengths):.0f}, invalid_moves {invalid}")
    print("Done.")


if __name__ == "__main__":
    main()
