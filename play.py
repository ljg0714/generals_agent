"""Watch a trained Generals.io agent play against a bot in a GUI window.

Usage:
    python play.py --model checkpoints/model_500.pt --opponent expander
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import config
from agents.generals_env import GeneralsEnv
from agents.network import PolicyValueNetwork
from agents.ppo_agent import PPOAgent
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.grid import GridFactory


def load_network(model_path: str, device) -> PolicyValueNetwork:
    network = PolicyValueNetwork(
        input_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        grid_size=config.PAD_TO,
        value_hidden=config.VALUE_HIDDEN,
    )
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    network.load_state_dict(state)
    network.to(device)
    network.eval()
    return network


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, required=True, help="path to checkpoint .pt")
    p.add_argument("--opponent", type=str, default="expander", choices=["random", "expander"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--greedy", action="store_true", help="act greedily (no sampling)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    device = torch.device("cuda" if use_cuda else "cpu")
    network = load_network(args.model, device)
    policy = PPOAgent(network, device=device, pad_to=config.PAD_TO, greedy=args.greedy)

    opponent = ExpanderAgent() if args.opponent == "expander" else RandomAgent()
    grid_factory = GridFactory(
        mode=config.GRID_MODE,
        min_grid_dims=config.MIN_GRID_DIMS,
        max_grid_dims=config.MAX_GRID_DIMS,
        mountain_density=config.MOUNTAIN_DENSITY,
        city_density=config.CITY_DENSITY,
    )

    for ep in range(args.episodes):
        env = GeneralsEnv(opponent=opponent, grid_factory=grid_factory,
                          pad_observations_to=config.PAD_TO, truncation=config.TRUNCATION,
                          render_mode="human")
        env.reset(seed=args.seed * 1000 + ep)
        done = False
        step = 0
        print(f"\n=== Episode {ep + 1} (model vs {opponent}) ===")
        while not done:
            a0 = policy.act_batched(env.last_obs[None], env.last_mask[None])[0]
            a1 = opponent.act(env.opponent_observation())
            _, _, term, trunc, info0 = env.step(a0, a1)
            done = term or trunc
            env.render()
            step += 1
        if term:
            result = "WIN" if bool(info0["winner"]) else "LOSS"
        else:
            result = "DRAW"
        print(f"Episode {ep + 1}: {result} after {step} steps")
        env.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
