"""PyTorch policy-value network for Generals.io (CleanRL-style PPO).

Input:  (B, 15, H, W) padded observation tensor + (B, H, W, 4) valid-move mask.
Output: policy logits over 9*H*W actions, and a scalar state value.

Action encoding (matches generals-bots):
  9 channels -> ch 0-3: move UP/DOWN/LEFT/RIGHT with "all-but-one" army
               ch 4-7: same directions with "split" (send half)
               ch 8:   pass (always valid)
  Index = ch * (H * W) + row * W + col
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PolicyValueNetwork(nn.Module):
    def __init__(
        self,
        input_channels: int = 15,
        hidden_channels: tuple[int, ...] = (32, 32, 32, 16),
        grid_size: int = 24,
        value_hidden: int = 64,
    ):
        super().__init__()
        c0, c1, c2, c3 = hidden_channels
        self.grid_size = grid_size

        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, c0, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c0, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1), nn.ReLU(),
        )
        # 9 output channels: 4 dirs (full) + 4 dirs (split) + pass
        self.policy_head = nn.Conv2d(c3, 9, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Conv2d(c3, 4, kernel_size=1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(grid_size * grid_size * 4, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, obs: torch.Tensor, valid_moves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits (B, 9HW), value (B, 1)).

        Args:
            obs: (B, C, H, W) float observation tensor.
            valid_moves: (B, H, W, 4) bool mask of legal moves (True = legal).
        """
        x = self.backbone(obs)
        logits = self.policy_head(x)                       # (B, 9, H, W)
        B, _, H, W = logits.shape

        # Mask illegal moves with a large negative penalty; pass is always legal.
        mask_penalty = (1.0 - valid_moves.permute(0, 3, 1, 2).float()) * -1e9
        pass_channel = torch.zeros((B, 1, H, W), device=logits.device)
        combined = torch.cat([mask_penalty, mask_penalty, pass_channel], dim=1)
        logits = (logits + combined).reshape(B, 9 * H * W)  # (B, 9HW)

        v = self.value_head(x).reshape(B, -1)
        value = self.value_fc(v)                            # (B, 1)
        return logits, value


def decode_actions(action_idx: np.ndarray, grid_size: int) -> np.ndarray:
    """Convert sampled action indices (9HW-encoded) to generals actions.

    Returns an (..., 5) int array with columns [pass, row, col, direction, split].
    """
    action_idx = np.asarray(action_idx)
    n_cells = grid_size * grid_size
    ch = action_idx // n_cells
    pos = action_idx % n_cells
    rows = pos // grid_size
    cols = pos % grid_size
    is_pass = (ch == 8).astype(np.int32)
    is_split = ((ch >= 4) & (ch < 8)).astype(np.int32)
    direction = (ch % 4).astype(np.int32)
    return np.stack([is_pass, rows, cols, direction, is_split], axis=-1)


def encode_action(action: np.ndarray, grid_size: int) -> np.ndarray:
    """Inverse of decode_actions. action: (..., 5) [pass, row, col, dir, split]."""
    action = np.asarray(action)
    is_pass = action[..., 0].astype(bool)
    is_split = action[..., 4].astype(bool)
    direction = action[..., 3]
    # channel: pass -> 8, split -> 4 + dir, full -> dir
    ch = np.where(is_pass, 8, np.where(is_split, direction + 4, direction)).astype(np.int64)
    pos = action[..., 1].astype(np.int64) * grid_size + action[..., 2].astype(np.int64)
    return ch * (grid_size * grid_size) + pos
