"""In-network memory augmentation for the policy network.

Mirrors the paper *Artificial Generals Intelligence* (arXiv:2507.06825), official
code Zerov3 `selfplay/network.py`, but with a shorter move-history stack
(`history_size=4` instead of 7) and memory state stored as a *plain dict* rather
than registered buffers (so it never enters `state_dict` and `load_state_dict`
is unaffected by the batch size).

A raw 15-channel generals-bots observation is folded into 31 channels:

  - 23 base/memory channels: armies, armies*owned, armies*opponent, armies*neutral,
    seen (explored), enemy_seen, revealed generals/cities/mountains, current
    neutral/owned/opponent/fog/structures, timestep, timestep%50, priority,
    land/army totals, and last-enemy-army-seen recency + value.
  - 8 move-history channels: own-side (army_stack) and enemy-side (enemy_stack)
    per-cell army deltas over the last `history_size` steps.

The `mem` dict is the persistent state: it is mutated in place by
`augment_observation` and must be reset at the start of a new game. Resets are
detected automatically via the timestep channel (index 13 == 0), exactly like the
paper, or explicitly by calling `zero_mem` / clearing the dict.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d

BUFFER_KEYS = [
    "army_stack",
    "enemy_stack",
    "last_army",
    "last_enemy_army",
    "cities",
    "generals",
    "mountains",
    "seen",
    "enemy_seen",
    "last_enemy_army_seen_value",
    "last_enemy_army_seen_timestep",
]

# The two history stacks are (B, history_size, H, W); everything else is (B, H, W).
_STACK_KEYS = ("army_stack", "enemy_stack")


def zero_mem(mem: dict, B: int, H: int, W: int, device, history_size: int = 4) -> dict:
    """Allocate (or reallocate) zeroed memory tensors for a batch of `B` samples."""
    for key in BUFFER_KEYS:
        if key in _STACK_KEYS:
            mem[key] = torch.zeros((B, history_size, H, W), dtype=torch.float32, device=device)
        else:
            mem[key] = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    return mem


def reset_histories(obs: torch.Tensor, mem: dict) -> None:
    """Zero every batch sample whose timestep channel (13) is 0 — a fresh game."""
    new_game = obs[:, 13, 0, 0] == 0
    if not new_game.any():
        return
    for key in BUFFER_KEYS:
        mem[key][new_game] = 0


def augment_observation(obs: torch.Tensor, mem: dict, history_size: int = 4) -> torch.Tensor:
    """Augment a raw padded observation with memory and move-history channels.

    Args:
        obs: (B, 15, H, W) raw generals-bots observation tensor (padded grid).
        mem: persistent state dict mutated in place.
        history_size: length of each side's move-history stack.

    Returns:
        (B, 31, H, W) augmented observation tensor.
    """
    armies = obs[:, 0]
    generals = obs[:, 1]
    cities = obs[:, 2]
    mountains = obs[:, 3]
    neutral_cells = obs[:, 4]
    owned_cells = obs[:, 5]
    opponent_cells = obs[:, 6]
    fog_cells = obs[:, 7]
    structures_in_fog = obs[:, 8]
    owned_land_count = obs[:, 9]
    owned_army_count = obs[:, 10]
    opponent_land_count = obs[:, 11]
    opponent_army_count = obs[:, 12]
    timestep = obs[:, 13]
    priority = obs[:, 14]

    reset_histories(obs, mem)

    current_army = armies * owned_cells
    current_enemy_army = armies * opponent_cells

    # Shift the history stacks and push the newest per-cell army delta.
    army_stack = mem["army_stack"]
    enemy_stack = mem["enemy_stack"]
    army_stack[:, 1:, :, :] = army_stack[:, :-1, :, :].clone()
    enemy_stack[:, 1:, :, :] = enemy_stack[:, :-1, :, :].clone()
    army_stack[:, 0, :, :] = current_army - mem["last_army"]
    enemy_stack[:, 0, :, :] = current_enemy_army - mem["last_enemy_army"]
    mem["last_army"] = current_army
    mem["last_enemy_army"] = current_enemy_army

    # Persistent exploration / sight memory (3x3 dilation of owned / opponent).
    mem["seen"] = torch.maximum(mem["seen"], max_pool2d(owned_cells, 3, 1, 1))
    mem["enemy_seen"] = torch.maximum(mem["enemy_seen"], max_pool2d(opponent_cells, 3, 1, 1))
    mem["cities"] = torch.maximum(mem["cities"], cities)
    mem["generals"] = torch.maximum(mem["generals"], generals)
    mem["mountains"] = torch.maximum(mem["mountains"], mountains)

    # Last place we saw enemy army, and how long ago / how strong it was.
    mem["last_enemy_army_seen_value"] = torch.where(
        current_enemy_army > 0, current_enemy_army, mem["last_enemy_army_seen_value"]
    )
    mem["last_enemy_army_seen_value"] += 0.01
    mem["last_enemy_army_seen_timestep"] = torch.where(
        current_enemy_army > 0, torch.zeros_like(mem["last_enemy_army_seen_timestep"]),
        mem["last_enemy_army_seen_timestep"],
    )

    ones = torch.ones_like(armies)
    channels = torch.stack(
        [
            armies,
            current_army,
            current_enemy_army,
            armies * neutral_cells,
            mem["seen"],
            mem["enemy_seen"],
            mem["generals"],
            mem["cities"],
            mem["mountains"],
            neutral_cells,
            owned_cells,
            opponent_cells,
            fog_cells,
            structures_in_fog,
            timestep * ones,
            (timestep % 50) * ones / 50,
            priority * ones,
            owned_land_count * ones,
            owned_army_count * ones,
            opponent_land_count * ones,
            opponent_army_count * ones,
            mem["last_enemy_army_seen_timestep"],
            mem["last_enemy_army_seen_value"],
        ],
        dim=1,
    )
    stacks = torch.cat([army_stack, enemy_stack], dim=1)
    return torch.cat([channels, stacks], dim=1)


class MemoryAugmenter(nn.Module):
    """Thin module wrapper that owns a `mem` dict for the *learner* network path.

    Memory is a plain dict attribute, NOT registered buffers: it never appears in
    `state_dict`, so `load_state_dict` and `OpponentPool._snapshot` are unaffected.
    A single network must always be called with the same batch size (per-sample
    indices must stay stable across steps); if the batch/device ever changes the
    state is reallocated fresh.
    """

    def __init__(self, history_size: int = 4):
        super().__init__()
        self.history_size = history_size
        self._mem: dict = {}

    def augment_observation(self, obs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = obs.shape
        m = self._mem
        if (
            not m
            or m["last_army"].shape[0] != B
            or m["last_army"].device != obs.device
        ):
            zero_mem(m, B, H, W, obs.device, self.history_size)
        return augment_observation(obs, m, self.history_size)
