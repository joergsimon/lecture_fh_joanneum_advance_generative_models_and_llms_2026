#!/usr/bin/env python3
"""Render several fixed seeds to choose a legible classroom trajectory."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_models import TinyNoisePredictor, diffusion_schedule  # noqa: E402


@torch.no_grad()
def main() -> None:
    checkpoint = torch.load(ROOT / "checkpoints" / "diffusion_mnist_tiny.pt", map_location="cpu", weights_only=True)
    model = TinyNoisePredictor()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    schedule = diffusion_schedule(checkpoint["steps"])
    betas, alphas, alpha_bars = schedule["betas"], schedule["alphas"], schedule["alpha_bars"]
    finals = []
    for seed in range(16):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn((1, 1, 28, 28), generator=generator)
        for t_index in reversed(range(len(betas))):
            t = torch.tensor([t_index])
            eps_hat = model(x, t)
            mean = (x - betas[t_index] / torch.sqrt(1 - alpha_bars[t_index]) * eps_hat) / torch.sqrt(alphas[t_index])
            x = mean if t_index == 0 else mean + torch.sqrt(betas[t_index]) * torch.randn(x.shape, generator=generator)
        finals.append(x.clamp(-1, 1))
    fig, axes = plt.subplots(4, 4, figsize=(7, 7))
    for seed, (ax, image) in enumerate(zip(axes.flat, finals)):
        ax.imshow(image[0, 0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(f"seed {seed}")
        ax.axis("off")
    fig.tight_layout()
    output = ROOT / "fallbacks" / "05_diffusion_seed_grid.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
