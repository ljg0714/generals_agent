"""GUI: watch HumanExeAgent (player 0) vs a built-in bot (player 1).

Uses the generals-bots human renderer to show the game in a window. The EklipZ
bot plays as player 0 ("agent_0") against ExpanderAgent / RandomAgent.

Usage:
    python play_human_exe.py                        # human_exe vs expander, 18-20 maps
    python play_human_exe.py --opponent random --episodes 2 --grid-min 12 --grid-max 14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config
from agents.generals_env import GeneralsEnv
from agents.human_exe_agent import HumanExeAgent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.grid import GridFactory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--opponent", default="expander", choices=["expander", "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--grid-min", type=int, default=18)
    p.add_argument("--grid-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    grid_factory = GridFactory(
        mode="uniform",
        min_grid_dims=(args.grid_min, args.grid_min),
        max_grid_dims=(args.grid_max, args.grid_max),
        mountain_density=config.MOUNTAIN_DENSITY,
        city_density=config.CITY_DENSITY,
    )
    opponent = ExpanderAgent() if args.opponent == "expander" else RandomAgent()

    for ep in range(args.episodes):
        env = GeneralsEnv(
            opponent=opponent,
            grid_factory=grid_factory,
            pad_observations_to=config.PAD_TO,
            truncation=config.TRUNCATION,
            render_mode="human",
        )
        env.reset(seed=args.seed * 1000 + ep)
        hx = HumanExeAgent(player_id="agent_0")
        done = False
        steps = 0
        print(f"\n=== Episode {ep + 1}: human_exe vs {args.opponent} "
              f"({args.grid_min}-{args.grid_max} maps) ===")
        while not done:
            a0 = hx.act_on_game(env._env.game, "agent_0")
            a1 = opponent.act(env.opponent_observation())
            _, _, term, trunc, info0 = env.step(a0, a1)
            done = term or trunc
            env.render()
            steps += 1
        if term:
            result = "WIN" if bool(info0["winner"]) else "LOSS"
        else:
            result = "DRAW"
        print(f"Episode {ep + 1}: {result} after {steps} steps")
        env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
