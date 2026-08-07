"""Supervised pretraining (behavior cloning) for the policy network.

Loads a dataset produced by train/generate_data.py and trains the policy to
imitate the demonstrator with cross-entropy over the masked 9*H*W action space.
The network already applies a -1e9 penalty to illegal moves in forward(), so
the loss only pushes probability onto legal actions.

Usage:
    python train/pretrain.py --data checkpoints/dataset.npz --out checkpoints/pretrained.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.network import PolicyValueNetwork


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="checkpoints/dataset.npz")
    p.add_argument("--out", default="checkpoints/pretrained.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    data = np.load(args.data)
    obs, masks, actions = data["obs"], data["masks"], data["actions"]
    P = int(data.get("pad_to", config.PAD_TO))
    print(f"Loaded {len(obs)} samples (pad_to={P}) on {device}")

    # train/val split
    n = len(obs)
    perm = np.random.default_rng(0).permutation(n)
    n_val = max(1, int(0.1 * n))
    tr, va = perm[: n - n_val], perm[n - n_val:]

    def make_loader(idx, shuffle):
        ds = TensorDataset(torch.from_numpy(obs[idx]), torch.from_numpy(masks[idx]),
                           torch.from_numpy(actions[idx]))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

    loader = make_loader(tr, shuffle=True)
    val_loader = make_loader(va, shuffle=False)

    net = PolicyValueNetwork(
        input_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        grid_size=P,
        value_hidden=config.VALUE_HIDDEN,
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        tot_loss = tot_acc = cnt = 0.0
        for ob, mk, ac in loader:
            ob, mk, ac = ob.to(device), mk.to(device), ac.to(device)
            logits, _ = net(ob, mk)
            loss = F.cross_entropy(logits, ac)
            acc = (logits.argmax(dim=1) == ac).float().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(ob)
            tot_acc += acc.item() * len(ob)
            cnt += len(ob)

        net.eval()
        val_acc = vcnt = 0.0
        with torch.no_grad():
            for ob, mk, ac in val_loader:
                ob, mk, ac = ob.to(device), mk.to(device), ac.to(device)
                logits, _ = net(ob, mk)
                val_acc += (logits.argmax(dim=1) == ac).float().sum().item()
                vcnt += len(ob)
        val_acc /= vcnt

        print(f"epoch {epoch}: loss {tot_loss / cnt:.4f} train_acc {tot_acc / cnt:.4f} "
              f"val_acc {val_acc:.4f} ({time.time() - t0:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": net.state_dict()}, args.out)

    print(f"Best val_acc {best_acc:.4f}; weights saved to {args.out}")


if __name__ == "__main__":
    main()
