"""Deployment wrapper that injects fog memory into the live generals.io observation.

The PPO policy was trained in the `generals-bots` env where
`FastGame.agent_observation` injected two things the real generals.io client
does NOT provide:

  1. General memory — a once-seen general (channel 1) stays marked even after
     its cells re-enter fog.
  2. Structure-TYPE memory — once a cell is explored, channels 3/2
     (mountains/cities) keep marking whether it is a mountain or a city after
     it re-fogs; unexplored structures stay type-unknown (channel 8 only).

This wrapper replicates that memory on the live observation (the deployment
harness the user described), so training and deployment see the same input
distribution. It also zeroes channel 14 (priority) — the training env always
sent 0, while the real client sends the player index.

The valid-move mask is computed AFTER padding the observation to PAD_TO, so it
stays aligned with the padded network input (the base PPOAgent.act computes the
mask on the raw map, which breaks for maps smaller than PAD_TO).
"""
from __future__ import annotations

import numpy as np

from agents.ppo_agent import PPOAgent


class DeployPPOAgent(PPOAgent):
    def __init__(self, network, device="cpu", pad_to: int = 24, verbose: bool = False):
        super().__init__(network, device=device, pad_to=pad_to, greedy=True)
        self._memory = None
        self.verbose = verbose

    def reset(self):
        """Called between episodes by callers that manage lifecycle."""
        self._memory = None

    def act(self, observation):
        """observation: a live generals.io Observation (raw map size)."""
        try:
            raw_shape = observation.armies.shape  # before in-place padding
            obs = self._with_memory(observation)
            # Pad BEFORE the base act() so compute_valid_move_mask sees the
            # padded grid and stays aligned with the padded network input.
            obs.pad_observation(self.pad_to)
            action = super().act(obs)
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

    def _with_memory(self, obs):
        h, w = obs.armies.shape
        # A new game starts when the turn counter resets to 0 (the client never
        # calls reset()); also re-init if the map size changed.
        if (
            self._memory is None
            or self._memory["shape"] != (h, w)
            or obs.timestep == 0
        ):
            self._memory = {
                "shape": (h, w),
                "seen_cells": np.zeros((h, w), dtype=bool),
                "seen_mountains": np.zeros((h, w), dtype=bool),
                "seen_generals": np.zeros((h, w), dtype=bool),
            }
        m = self._memory

        # A cell is visible iff it is not plain fog (terrain -3, fog_cells) and
        # not a fogged structure (terrain -4, structures_in_fog).
        fog = np.asarray(obs.fog_cells, dtype=bool) | np.asarray(obs.structures_in_fog, dtype=bool)
        visible = ~fog
        m["seen_cells"] |= visible
        # `obs.mountains` only carries currently-visible mountains (terrain -2);
        # accumulate so they are remembered once seen. Same for generals.
        m["seen_mountains"] |= np.asarray(obs.mountains, dtype=bool)
        m["seen_generals"] |= np.asarray(obs.generals, dtype=bool)

        # Inject the memories (mirrors FastGame.agent_observation):
        #   - generals: once seen, stay marked in channel 1.
        #   - mountains: once seen, stay marked in channel 3.
        #   - cities: only cities whose cell was ever visible are type-known
        #     (channel 2); unseen cities stay hidden (only channel 8).
        obs.generals = m["seen_generals"]
        obs.mountains = m["seen_mountains"]
        obs.cities = np.asarray(obs.cities, dtype=bool) & m["seen_cells"]
        # Training env always sent priority = 0.
        obs.priority = 0
        return obs
