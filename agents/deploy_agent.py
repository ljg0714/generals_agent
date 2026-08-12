"""Deployment wrapper for the memory-augmented PPO policy.

The live generals.io client provides raw 15-channel observations. The in-network
`MemoryAugmenter` (agents/memory_aug.py) now provides the same fog memory the
policy was trained on (persistent seen generals/cities/mountains, explored cells,
enemy sight, move-history), so this wrapper only needs to:

  1. Pad the observation to PAD_TO *before* the valid-move mask is computed, so
     the mask stays aligned with the padded network input.
  2. Zero channel 14 (priority) — the training env always sent 0, while the real
     client sends the player index.

Memory resets at the start of each game are handled automatically by
`augment_observation` (timestep channel == 0).
"""
from __future__ import annotations

import numpy as np

from agents.ppo_agent import PPOAgent


class DeployPPOAgent(PPOAgent):
    def __init__(self, network, device="cpu", pad_to: int = 24, verbose: bool = False):
        super().__init__(network, device=device, pad_to=pad_to, greedy=True)
        self.verbose = verbose

    def reset(self):
        """Called between episodes by callers that manage lifecycle."""
        super().reset()

    def act(self, observation):
        """observation: a live generals.io Observation (raw map size)."""
        try:
            raw_shape = observation.armies.shape  # before in-place padding
            # Pad BEFORE the base act() so compute_valid_move_mask sees the
            # padded grid and stays aligned with the padded network input.
            observation.pad_observation(self.pad_to)
            # Training env always sent priority = 0.
            observation.priority = 0
            action = super().act(observation)
        except Exception:
            # Never let one bad observation kill the game loop; log it.
            import traceback
            traceback.print_exc()
            from generals.core.action import Action
            action = Action(1, 0, 0, 0, 0)  # pass
        if self.verbose:
            h, w = raw_shape
            owned = int(np.asarray(observation.owned_cells, bool).sum())
            pass_or_play = bool(action[0])
            print(f"  [turn {observation.timestep}] map {w}x{h} owned={owned} "
                  f"action={'PASS' if pass_or_play else list(map(int, action[1:]))}")
        return action
