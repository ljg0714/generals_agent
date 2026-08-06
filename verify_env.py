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
    net = PolicyValueNetwork(input_channels=15, grid_size=config.PAD_TO)
    obs = np.zeros((2, 15, config.PAD_TO, config.PAD_TO), np.float32)
    mask = np.zeros((2, config.PAD_TO, config.PAD_TO, 4), np.bool_)
    mask[:, :, :, 0] = True
    logits, value = net(torch.from_numpy(obs), torch.from_numpy(mask))
    assert logits.shape == (2, 9 * config.PAD_TO * config.PAD_TO)
    assert value.shape == (2, 1)
    print(f"[OK] network forward: logits {tuple(logits.shape)}, value {tuple(value.shape)}")


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

    net = PolicyValueNetwork(input_channels=15, grid_size=config.PAD_TO)
    done = False
    steps = 0
    term = trunc = False
    while not done and steps < 400:
        logits, _ = net(torch.from_numpy(env.last_obs[None]), torch.from_numpy(env.last_mask[None]))
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
    test_env()
    print("\nAll environment checks passed.")
