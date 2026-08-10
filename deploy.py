"""Deploy the trained PPO policy to the real generals.io server.

Usage:
  # Private lobby (bot server, force-starts games) — best for testing
  python deploy.py --user-id <SECRET> --lobby <lobby_id> --username "[Bot] PPOBot"

  # Public 1v1 queue (plays ranked-ish games against humans/bots)
  python deploy.py --user-id <SECRET> --one-v1 --public --username "[Bot] PPOBot"

The policy must be trained with the fog-memory observation (FastGame patches);
`DeployPPOAgent` re-injects the general memory and structure-type memory on the
live observation so training and deployment see the same input. `user_id` is
the secret code from your generals.io account settings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import config
from agents.deploy_agent import DeployPPOAgent
from agents.network import PolicyValueNetwork
from generals.remote import GeneralsIOClient


def load_agent(model_path: str, device: str) -> DeployPPOAgent:
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
    return DeployPPOAgent(network, device=device, pad_to=config.PAD_TO)


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy the trained policy to generals.io")
    p.add_argument("--model", default="checkpoints/deploy.pt")
    p.add_argument("--user-id", required=True, help="generals.io user_id (secret, from account settings)")
    p.add_argument("--username", default=None, help="registered bot display name, must start with [Bot]; register if given")
    p.add_argument("--bot-key", default=None, help="override the library's bot key")
    p.add_argument("--lobby", default=None, help="private lobby id to join and force-start (bot server)")
    p.add_argument("--one-v1", action="store_true", help="join the public 1v1 queue")
    p.add_argument("--public", action="store_true", help="connect to the public server (ws.generals.io) instead of the bot server")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-games", type=int, default=0, help="stop after N games (0 = play forever)")
    args = p.parse_args()

    if not args.lobby and not args.one_v1:
        p.error("provide --lobby <id> or --one-v1")

    agent = load_agent(args.model, args.device)
    print(f"Loaded {args.model} -> DeployPPOAgent (pad_to={config.PAD_TO}, greedy, fog-memory injection)")

    with GeneralsIOClient(agent, args.user_id, public_server=args.public) as client:
        if args.bot_key:
            client.bot_key = args.bot_key
        if args.username:
            try:
                client.register_agent(args.username)
            except ValueError as e:
                print(f"register_agent failed (continuing anyway): {e}")

        games = 0
        while args.num_games == 0 or games < args.num_games:
            if args.one_v1:
                client.join_1v1_queue()
                games += 1
            else:
                if client.status == "off":
                    client.join_private_lobby(args.lobby)
                elif client.status == "lobby":
                    client.join_game()
                    games += 1


if __name__ == "__main__":
    main()
