"""Environment smoke test for the Generals.io RL project.

Verifies:
  1. The generals-bots gymnasium environment imports and runs.
  2. Action encode/decode round-trips.
  3. The policy-value network forward pass works.
  4. A random-policy game reaches termination/truncation.
  5. The reward-shaping function returns sensible values.

Usage:
    python verify_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402
from agents.generals_env import GeneralsEnv  # noqa: E402
from agents.network import PolicyValueNetwork, decode_actions, encode_action  # noqa: E402
from agents.rewards import composite_reward  # noqa: E402
from generals.agents.random_agent import RandomAgent  # noqa: E402
from generals.core.grid import GridFactory  # noqa: E402


def test_encode_decode():
    for idx in [0, 100, 8 * 24 * 24, 8 * 24 * 24 + 5, 5 * 24 * 24 + 100]:
        a = decode_actions(np.array([idx]), 24)[0]
        assert int(encode_action(a, 24)) == idx, (idx, a)
    print("[OK] action encode/decode round-trip")


def test_network():
    net = PolicyValueNetwork(input_channels=config.INPUT_CHANNELS, grid_size=config.PAD_TO)
    obs = np.zeros((2, config.INPUT_CHANNELS, config.PAD_TO, config.PAD_TO), np.float32)
    mask = np.zeros((2, config.PAD_TO, config.PAD_TO, 4), np.bool_)
    mask[:, :, :, 0] = True
    logits, value = net(torch.from_numpy(obs), torch.from_numpy(mask))
    assert logits.shape == (2, 9 * config.PAD_TO * config.PAD_TO)
    assert value.shape == (2, 1)
    print(f"[OK] network forward ({config.INPUT_CHANNELS}ch): logits "
          f"{tuple(logits.shape)}, value {tuple(value.shape)}")


def test_memory_aug():
    from agents.memory_aug import MemoryAugmenter

    P = config.PAD_TO
    aug = MemoryAugmenter(history_size=config.MEMORY_HISTORY_SIZE)
    # Sample 0 starts a new game (timestep 0); sample 1 is mid-game (timestep 5).
    obs0 = np.zeros((2, 15, P, P), np.float32)
    obs0[:, 5, P // 2, P // 2] = 1    # owned cell
    obs0[:, 0, P // 2, P // 2] = 10   # army on it
    obs0[0, 13] = 0
    obs0[1, 13] = 5

    aug1 = aug.augment_observation(torch.from_numpy(obs0))
    assert aug1.shape == (2, config.INPUT_CHANNELS, P, P)
    # seen (ch 4) marks the owned cell for both samples.
    assert aug1[0, 4, P // 2, P // 2] == 1.0
    assert aug1[1, 4, P // 2, P // 2] == 1.0
    # army_stack[0] (ch 23) at the owned cell = current_army - last(0) = 10.
    assert aug1[0, 23, P // 2, P // 2] == 10.0
    assert aug1[1, 23, P // 2, P // 2] == 10.0

    # Second call: sample 0 resets again (timestep 0) -> delta = 10 - 0;
    # sample 1 persists (timestep 6) -> last_army=10, no change -> delta = 0.
    obs1 = obs0.copy()
    obs1[0, 13] = 0
    obs1[1, 13] = 6
    aug2 = aug.augment_observation(torch.from_numpy(obs1))
    assert aug2[0, 23, P // 2, P // 2] == 10.0
    assert aug2[1, 23, P // 2, P // 2] == 0.0
    # seen persists across steps for the mid-game sample.
    assert aug2[1, 4, P // 2, P // 2] == 1.0
    print("[OK] memory augmentation: shape, reset-on-timestep-0, persistence")


def test_env():
    env = GeneralsEnv(
        opponent=RandomAgent(),
        grid_factory=GridFactory(mode="uniform", min_grid_dims=(6, 6), max_grid_dims=(6, 6)),
        pad_observations_to=config.PAD_TO,
        truncation=200,
    )
    obs, info = env.reset(seed=42)
    assert obs.shape == (15, config.PAD_TO, config.PAD_TO)
    assert info["masks"].shape == (config.PAD_TO, config.PAD_TO, 4)

    net = PolicyValueNetwork(input_channels=config.INPUT_CHANNELS, grid_size=config.PAD_TO)
    done = False
    steps = 0
    term = trunc = False
    while not done and steps < 400:
        with torch.no_grad():
            obs_aug = net.augment_observation(torch.from_numpy(env.last_obs[None]))
            logits, _ = net(obs_aug, torch.from_numpy(env.last_mask[None]))
        idx = torch.distributions.Categorical(logits=logits).sample().numpy()
        act0 = decode_actions(idx, config.PAD_TO)[0]
        act1 = env.opponent.act(env.opponent_observation())
        _, _, term, trunc, _info0 = env.step(act0, act1)
        done = term or trunc
        steps += 1

    # reward-shaping sanity: both players pass -> potentials unchanged -> ~0
    prior = env.last_obs
    pass_action = decode_actions(np.array([8 * config.PAD_TO * config.PAD_TO]), config.PAD_TO)[0]
    env.step(pass_action, pass_action)
    r = composite_reward(prior[None], env.last_obs[None], np.array([False]))
    assert abs(float(r[0])) < 1e-5, f"expected ~0 shaping for pass, got {r[0]}"
    env.close()

    print(f"[OK] env: game ended terminated={term} truncated={trunc} after {steps} steps; "
          f"pass reward shaping ~0 ({float(r[0]):+.6f})")


if __name__ == "__main__":
    test_encode_decode()
    test_network()
    test_memory_aug()
    test_env()
    print("\nAll environment checks passed.")
