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
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config
from agents.deploy_agent import DeployPPOAgent
from agents.network import PolicyValueNetwork
from generals.core.config import Direction
from generals.remote import GeneralsIOClient

DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]


class SafeGeneralsIOClient(GeneralsIOClient):
    """Client whose source/destination index math is done in int64.

    The library's `Action` is a np.int8 ndarray; `_generate_action` computes
    `row * map_width` in int8, which OVERFLOWS once row * width > 127 (e.g.
    row 7 on a 19-wide map). The wrong source index is silently rejected by the
    server, so the bot stalls on one cell. Recompute in int64.
    """

    def _generate_action(self, observation):
        action = self.agent.act(observation)
        pass_or_play = int(action[0])
        if pass_or_play:
            return None
        i, j = int(action[1]), int(action[2])
        direction = int(action[3])
        split = int(action[4])
        source = np.array([i, j], dtype=np.int64)
        destination = source + np.array(DIRECTIONS[direction].value, dtype=np.int64)
        width = int(self.game_state.map[0])
        source_index = int(source[0]) * width + int(source[1])
        destination_index = int(destination[0]) * width + int(destination[1])
        return (source_index, destination_index, split)

    def join_1v1_queue(self):
        """Join 1v1 queue, surfacing server errors instead of hanging."""
        self._status = "queue"
        print("Joining 1v1 queue...")
        self.emit("join_1v1", (self.user_id, self.bot_key))
        while True:
            event, *data = self.receive()
            if event == "game_start":
                self._status = "game"
                self._initialize_game(data)
                self._play_game()
                break
            if event in ("error", "gio_error"):
                print(f"[join_1v1] SERVER ERROR ({event}): {data}")
                self._status = "off"
                return
            if event in ("pre_game_start",):
                continue  # normal generals.io event preceding game_start
            print(f"[join_1v1] unexpected event: {event} data={data}")


def load_agent(model_path: str, device: str, verbose: bool = False) -> DeployPPOAgent:
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
    return DeployPPOAgent(network, device=device, pad_to=config.PAD_TO, verbose=verbose)


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
    p.add_argument("--verbose", action="store_true", help="log every turn: map size, owned cells, chosen action")
    args = p.parse_args()

    if not args.lobby and not args.one_v1:
        p.error("provide --lobby <id> or --one-v1")
    if args.one_v1 and not args.public:
        p.error("--one-v1 requires --public: the 1v1 queue is on the public server (ws.generals.io)")

    agent = load_agent(args.model, args.device, verbose=args.verbose)
    print(f"Loaded {args.model} -> DeployPPOAgent (pad_to={config.PAD_TO}, greedy, fog-memory injection)")

    with SafeGeneralsIOClient(agent, args.user_id, public_server=args.public) as client:
        if args.bot_key:
            client.bot_key = args.bot_key
        if args.username:
            try:
                client.register_agent(args.username)
                print("Registration OK.")
            except ValueError as e:
                print(f"register_agent FAILED: {e}")
                if args.one_v1:
                    print("The public 1v1 queue requires a registered username. Check --bot-key "
                          "(you may need your own bot key from generals.io account settings).")
                    return

        games = 0
        while args.num_games == 0 or games < args.num_games:
            try:
                if args.one_v1:
                    client.join_1v1_queue()
                    games += 1
                else:
                    if client.status == "off":
                        client.join_private_lobby(args.lobby)
                    elif client.status == "lobby":
                        client.join_game()
                        games += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"game loop error, reconnecting... (games played={games})")
                time.sleep(2)


if __name__ == "__main__":
    main()
