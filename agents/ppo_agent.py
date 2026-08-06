"""Wrap a PyTorch policy so it can act as a generals-bots `Agent`.

Used both for self-play opponents and for evaluation / GUI play.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Categorical

from generals.agents.agent import Agent
from generals.core.action import Action, compute_valid_move_mask

from agents.network import PolicyValueNetwork, decode_actions


class PPOAgent(Agent):
    def __init__(
        self,
        network: PolicyValueNetwork,
        device: str | torch.device = "cpu",
        pad_to: int = 24,
        greedy: bool = False,
        id: str = "PPO",
    ):
        super().__init__(id=id)
        self.network = network
        self.device = torch.device(device)
        self.pad_to = pad_to
        self.greedy = greedy
        self.network.eval()

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
    ) -> np.ndarray:
        """Batch action selection.

        Args:
            obs: (G, 15, H, W) float32 padded observations.
            masks: (G, H, W, 4) bool valid-move masks.
            greedy: override the instance greedy flag if given.

        Returns:
            (G, 5) int32 actions [pass, row, col, direction, split].
        """
        return self._act_batched(obs, masks, greedy=greedy)

    def _act_batched(self, obs: np.ndarray, masks: np.ndarray, greedy: bool | None = None) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(self.device)
            mask_t = torch.from_numpy(masks).to(self.device)
            logits, _ = self.network(obs_t, mask_t)
            if (greedy if greedy is not None else self.greedy):
                idx = torch.argmax(logits, dim=-1)
            else:
                idx = Categorical(logits=logits).sample()
            idx = idx.cpu().numpy()
        return decode_actions(idx, self.pad_to)

    def reset(self):
        pass
