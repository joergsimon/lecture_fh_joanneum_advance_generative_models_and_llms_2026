#!/usr/bin/env python3
"""Train the small offline checkpoints and create compact demo data.

Run this only when checkpoints need to be regenerated. The classroom notebooks
load the committed checkpoints and never train during the lecture.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demo_models import (  # noqa: E402
    ConditionalTinyNoisePredictor,
    TinyNoisePredictor,
    TinyVAE,
    diffusion_schedule,
    load_idx_images,
    load_idx_labels,
)


def find_mnist_raw() -> Path:
    candidates = [ROOT / "data" / "MNIST" / "raw"]
    override = os.environ.get("LECTURE2_MNIST_RAW")
    if override:
        candidates.insert(0, Path(override).expanduser())
    required = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No uncompressed MNIST IDX files were found. Expected "
        f"{', '.join(required)} in one of:\n{searched}\n"
        "Copy the files to data/MNIST/raw or set LECTURE2_MNIST_RAW."
    )


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_training_data(limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    raw = find_mnist_raw()
    images = load_idx_images(raw / "train-images-idx3-ubyte").astype(np.float32) / 255.0
    labels = load_idx_labels(raw / "train-labels-idx1-ubyte").astype(np.int64)
    if limit is not None:
        images, labels = images[:limit], labels[:limit]
    return torch.from_numpy(images[:, None]), torch.from_numpy(labels)


def make_demo_samples(images: torch.Tensor, labels: torch.Tensor) -> None:
    selected = []
    selected_labels = []
    for digit in range(10):
        idx = int(torch.where(labels == digit)[0][2])
        selected.append(images[idx].numpy())
        selected_labels.append(digit)
    np.savez_compressed(
        ROOT / "data" / "mnist_demo_samples.npz",
        images=np.stack(selected),
        labels=np.asarray(selected_labels),
    )


def train_vae(images: torch.Tensor, device: torch.device, epochs: int) -> dict[str, list[float]]:
    torch.manual_seed(7)
    model = TinyVAE(latent_dim=2).to(device)
    loader = DataLoader(TensorDataset(images), batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"loss": [], "reconstruction": [], "kl": []}
    model.train()
    for epoch in range(epochs):
        totals = np.zeros(3, dtype=np.float64)
        seen = 0
        for (x,) in loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            reconstruction = nn.functional.binary_cross_entropy(recon, x, reduction="sum") / len(x)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
            loss = reconstruction + 0.7 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), reconstruction.item(), kl.item()]) * len(x)
            seen += len(x)
        values = totals / seen
        history["loss"].append(float(values[0]))
        history["reconstruction"].append(float(values[1]))
        history["kl"].append(float(values[2]))
        print(f"VAE epoch {epoch + 1:02d}/{epochs}: loss={values[0]:.3f}, recon={values[1]:.3f}, kl={values[2]:.3f}")
    torch.save(
        {"state_dict": model.cpu().state_dict(), "latent_dim": 2, "history": history, "seed": 7},
        ROOT / "checkpoints" / "vae_mnist_latent2.pt",
    )
    return history


def train_diffusion(images: torch.Tensor, device: torch.device, epochs: int) -> dict[str, list[float]]:
    torch.manual_seed(11)
    model = TinyNoisePredictor().to(device)
    loader = DataLoader(TensorDataset(images), batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    schedule = diffusion_schedule(100)
    alpha_bars = schedule["alpha_bars"].to(device)
    history = {"mse": []}
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for (x0,) in loader:
            x0 = x0.to(device) * 2.0 - 1.0
            t = torch.randint(0, len(alpha_bars), (len(x0),), device=device)
            epsilon = torch.randn_like(x0)
            a_bar = alpha_bars[t][:, None, None, None]
            xt = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * epsilon
            prediction = model(xt, t)
            loss = nn.functional.mse_loss(prediction, epsilon)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x0)
            seen += len(x0)
        mse = total / seen
        history["mse"].append(float(mse))
        print(f"Diffusion epoch {epoch + 1:02d}/{epochs}: noise MSE={mse:.4f}")
    torch.save(
        {"state_dict": model.cpu().state_dict(), "steps": 100, "history": history, "seed": 11},
        ROOT / "checkpoints" / "diffusion_mnist_tiny.pt",
    )
    return history


def train_cfg_diffusion(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    epochs: int,
    condition_dropout: float = 0.15,
) -> dict[str, list[float]]:
    """Train one model for conditional and unconditional CFG predictions."""
    torch.manual_seed(19)
    model = ConditionalTinyNoisePredictor().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=256, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    schedule = diffusion_schedule(100)
    alpha_bars = schedule["alpha_bars"].to(device)
    history = {"mse": []}
    model.train()
    for epoch in range(epochs):
        total = 0.0
        seen = 0
        for x0, y in loader:
            x0, y = x0.to(device) * 2.0 - 1.0, y.to(device)
            drop = torch.rand(len(y), device=device) < condition_dropout
            labels_used = torch.where(drop, torch.full_like(y, model.null_label), y)
            t = torch.randint(0, len(alpha_bars), (len(x0),), device=device)
            epsilon = torch.randn_like(x0)
            a_bar = alpha_bars[t][:, None, None, None]
            xt = a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * epsilon
            prediction = model(xt, t, labels_used)
            loss = nn.functional.mse_loss(prediction, epsilon)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x0)
            seen += len(x0)
        mse = total / seen
        history["mse"].append(float(mse))
        print(f"CFG diffusion epoch {epoch + 1:02d}/{epochs}: noise MSE={mse:.4f}")
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "steps": 100,
            "history": history,
            "seed": 19,
            "num_classes": 10,
            "null_label": 10,
            "condition_dropout": condition_dropout,
        },
        ROOT / "checkpoints" / "diffusion_mnist_cfg.pt",
    )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-epochs", type=int, default=8)
    parser.add_argument("--diffusion-epochs", type=int, default=6)
    parser.add_argument("--diffusion-limit", type=int, default=30_000)
    parser.add_argument("--cfg-epochs", type=int, default=8)
    parser.add_argument("--cfg-limit", type=int, default=30_000)
    parser.add_argument("--train-cfg", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    args = parser.parse_args()

    ROOT.joinpath("checkpoints").mkdir(exist_ok=True)
    ROOT.joinpath("data").mkdir(exist_ok=True)
    images, labels = load_training_data()
    make_demo_samples(images, labels)
    device = choose_device()
    print(f"Training on {device}")
    started = time.perf_counter()
    metadata: dict[str, object] = {"device": str(device)}
    if not args.skip_vae:
        metadata["vae"] = train_vae(images, device, args.vae_epochs)
    if not args.skip_diffusion:
        metadata["diffusion"] = train_diffusion(images[: args.diffusion_limit], device, args.diffusion_epochs)
    if args.train_cfg:
        metadata["cfg_diffusion"] = train_cfg_diffusion(
            images[: args.cfg_limit], labels[: args.cfg_limit], device, args.cfg_epochs
        )
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    (ROOT / "checkpoints" / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished in {metadata['elapsed_seconds']} seconds")


if __name__ == "__main__":
    main()
