"""Wrap a PyTorch policy so it can act as a generals-bots `Agent`.

Used both for self-play opponents and for evaluation / GUI play. Because the
policy consumes memory-augmented observations (31 channels), the agent owns the
augmentation memory: `self._mem` holds one `mem` dict per *env index* (or a
single `None` key for eval/play/deploy). This lets a shared pool snapshot act for
many environments without cross-env state pollution, and keeps eval/deploy from
touching the learner network's own augmenter state.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from generals.agents.agent import Agent
from generals.core.action import Action, compute_valid_move_mask

import config
from agents.memory_aug import BUFFER_KEYS, augment_observation, zero_mem
from agents.network import PolicyValueNetwork, decode_actions


class PPOAgent(Agent):
    def __init__(
        self,
        network: PolicyValueNetwork,
        device: str | torch.device = "cpu",
        pad_to: int = 24,
        greedy: bool = False,
        id: str = "PPO",
        history_size: int | None = None,
    ):
        super().__init__(id=id)
        self.device = torch.device(device)
        self.network = network.to(self.device)
        self.pad_to = pad_to
        self.greedy = greedy
        self.history_size = history_size if history_size is not None else config.MEMORY_HISTORY_SIZE
        self.network.eval()
        # Per-env augmentation memory: key -> {buffer_name: tensor(batch 1)}.
        self._mem: dict = {}

    def reset(self):
        """Clear all augmentation memory (start fresh games)."""
        self._mem = {}

    def _ensure_mem(self, key, H: int, W: int) -> dict:
        m = self._mem.get(key)
        if m is None or m["last_army"].shape[-1] != W:
            m = {}
            zero_mem(m, 1, H, W, self.device, self.history_size)
            self._mem[key] = m
        return m

    def act(self, observation) -> Action:
        """observation: a generals Observation (padded here if needed)."""
        obs = np.asarray(observation.as_tensor(pad_to=self.pad_to), dtype=np.float32)
        mask = np.asarray(compute_valid_move_mask(observation), dtype=bool)  # (P, P, 4)
        action = self._act_batched(obs[None], mask[None])[0]
        a = action.astype(int)
        return Action(a[0], a[1], a[2], a[3], a[4])

    def act_batched(
        self,
        obs: np.ndarray,
        masks: np.ndarray,
        greedy: bool | None = None,
        env_keys=None,
    ) -> np.ndarray:
        """Batch action selection.

        Args:
            obs: (G, 15, H, W) float32 padded raw observations.
            masks: (G, H, W, 4) bool valid-move masks.
            greedy: override the instance greedy flag if given.
            env_keys: per-sample environment keys (used to index `self._mem` so a
                shared opponent keeps each env's memory isolated). Default: one
                shared `None` key.

        Returns:
            (G, 5) int32 actions [pass, row, col, direction, split].
        """
        return self._act_batched(obs, masks, greedy=greedy, env_keys=env_keys)

    def _act_batched(self, obs: np.ndarray, masks: np.ndarray, greedy: bool | None = None,
                     env_keys=None) -> np.ndarray:
        G = obs.shape[0]
        H, W = obs.shape[-2], obs.shape[-1]
        if env_keys is None:
            env_keys = [None] * G

        with torch.no_grad():
            # Stack each env's memory into one batch, augment, then scatter back.
            mems = [self._ensure_mem(k, H, W) for k in env_keys]
            # Each per-env mem is already batch-1; concat along the batch dim.
            stacked = {k: torch.cat([m[k] for m in mems], dim=0) for k in BUFFER_KEYS}

            obs_t = torch.from_numpy(np.asarray(obs)).to(self.device)
            obs_aug = augment_observation(obs_t, stacked, self.history_size)

            for j, key in enumerate(env_keys):
                for k in BUFFER_KEYS:
                    # Keep the batch dim: each per-env mem is batch-1.
                    self._mem[key][k] = stacked[k][j:j + 1]

            mask_t = torch.from_numpy(np.asarray(masks)).to(self.device)
            logits, _ = self.network(obs_aug, mask_t)
            if (greedy if greedy is not None else self.greedy):
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = Categorical(logits=logits).sample()
            idx = idx.cpu().numpy()
        return decode_actions(idx, self.pad_to)
