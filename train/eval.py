"""Evaluate a trained Generals.io policy against fixed opponents.

CLI: python train/eval.py --model checkpoints/latest.pt --grid-min 18 --grid-max 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.generals_env import GeneralsEnv
from agents.ppo_agent import PPOAgent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.grid import GridFactory

try:
    from agents.human_exe_agent import HumanExeAgent
    _HAS_HUMAN_EXE = True
except Exception:
    HumanExeAgent = None
    _HAS_HUMAN_EXE = False


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
            # HumanExeAgent is stateful per game -> a fresh instance each game.
            if _HAS_HUMAN_EXE and isinstance(opp, HumanExeAgent):
                opp = HumanExeAgent(player_id="agent_1")
            env = GeneralsEnv(opponent=opp, grid_factory=make_grid_factory(cfg),
                              pad_observations_to=cfg.PAD_TO, truncation=cfg.TRUNCATION)
            env.reset(seed=int(rng.integers(0, 2**31)))
            done = False
            length = 0
            while not done:
                a0 = policy.act_batched(env.last_obs[None], env.last_mask[None])[0]
                if _HAS_HUMAN_EXE and isinstance(opp, HumanExeAgent):
                    a1 = opp.act_on_game(env._env.game, env.opponent_id)
                else:
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


def _parse_dims(s: str) -> tuple[int, int]:
    parts = [int(p) for p in s.split(",")]
    return (parts[0], parts[0]) if len(parts) == 1 else (parts[0], parts[1])


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint vs built-in bots")
    p.add_argument("--model", default="checkpoints/latest.pt")
    p.add_argument("--grid-min", type=_parse_dims, default=None, help="min grid, e.g. 18 or 18,19")
    p.add_argument("--grid-max", type=_parse_dims, default=None, help="max grid, e.g. 20 or 20,20")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--opponents", default="random,expander",
                   help="comma-separated: random,expander,human_exe")
    p.add_argument("--greedy", action="store_true", help="greedy (deterministic) actions")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch

    import config
    from agents.network import PolicyValueNetwork

    if args.grid_min or args.grid_max:
        config.MIN_GRID_DIMS = args.grid_min or args.grid_max
        config.MAX_GRID_DIMS = args.grid_max or args.grid_min

    net = PolicyValueNetwork(
        input_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        grid_size=config.PAD_TO,
        value_hidden=config.VALUE_HIDDEN,
    )
    state = torch.load(args.model, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    net.load_state_dict(state)

    from generals.agents.expander_agent import ExpanderAgent
    from generals.agents.random_agent import RandomAgent

    opponents = []
    for name in args.opponents.split(","):
        name = name.strip()
        if name == "random":
            opponents.append(RandomAgent())
        elif name == "expander":
            opponents.append(ExpanderAgent())
        elif name == "human_exe":
            if not _HAS_HUMAN_EXE:
                raise ValueError("human_exe opponent requires the vendored bot deps "
                                 "(logbook, ortools, disjoint-set, unionfind)")
            opponents.append(HumanExeAgent(player_id="agent_1"))
        else:
            raise ValueError(f"unknown opponent: {name}")

    print(f"grid {config.MIN_GRID_DIMS}-{config.MAX_GRID_DIMS} | games={args.games} | "
          f"{'greedy' if args.greedy else 'sampling'}")
    res = evaluate(
        net, config, num_games=args.games, opponents=opponents,
        device=args.device, greedy=args.greedy, seed=args.seed,
    )
    for k, v in res.items():
        total = v["wins"] + v["losses"] + v["draws"]
        print(f"{k}: win_rate={v['win_rate']:.3f} ({v['wins']}/{total}) "
              f"draws={v['draws']} avg_len={v['avg_length']:.0f}")


if __name__ == "__main__":
    main()
