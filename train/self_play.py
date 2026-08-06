"""Self-play opponent pool for Generals.io PPO training.

The pool keeps the most recent `pool_size` snapshots of the policy. Each rollout
sampled from the pool is a frozen copy of an older policy, which gives the
learner a curriculum of progressively stronger opponents while avoiding the
degenerate feedback loop of always playing itself.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from agents.ppo_agent import PPOAgent


class OpponentPool:
    def __init__(self, network: torch.nn.Module, device: str | torch.device = "cpu",
                 pad_to: int = 24, pool_size: int = 10, greedy: bool = False):
        self.pool_size = pool_size
        self.pool: list[PPOAgent] = []
        self._template = network
        self.device = torch.device(device)
        self.pad_to = pad_to
        self.greedy = greedy

    def _snapshot(self, network: torch.nn.Module) -> PPOAgent:
        net = copy.deepcopy(self._template)
        net.load_state_dict(copy.deepcopy(network.state_dict()))
        net.eval()
        net.to(self.device)
        return PPOAgent(net, device=self.device, pad_to=self.pad_to, greedy=self.greedy)

    def add(self, network: torch.nn.Module) -> None:
        """Snapshot the current policy and add it to the pool (FIFO)."""
        self.pool.append(self._snapshot(network))
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)

    def sample(self) -> PPOAgent:
        return self.pool[int(np.random.randint(len(self.pool)))]

    def __len__(self) -> int:
        return len(self.pool)
