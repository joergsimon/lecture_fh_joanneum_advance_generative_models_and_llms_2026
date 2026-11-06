"""Small, dependency-light models used by the Lecture 2 demonstrations."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn


def load_idx_images(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def load_idx_labels(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    if len(data) != count:
        raise ValueError(f"Expected {count} labels, found {len(data)}")
    return data


class TinyVAE(nn.Module):
    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.flatten(1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * torch.randn_like(sigma)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def sinusoidal_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=t.device, dtype=torch.float32)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = t.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class TinyNoisePredictor(nn.Module):
    """A compact time-conditioned convolutional epsilon predictor for MNIST."""

    def __init__(self, time_dim: int = 32) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.in_conv = nn.Conv2d(1, 32, 3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.out_conv = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        h = torch.nn.functional.silu(self.in_conv(x))
        h = self.block1(h) + emb[:, :, None, None]
        h = self.block2(h)
        return self.out_conv(h)


class ConditionalTinyNoisePredictor(nn.Module):
    """Small class-conditional U-Net with one null label for CFG."""

    def __init__(self, time_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.null_label = num_classes
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.class_embedding = nn.Embedding(num_classes + 1, 64)
        self.input_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.output_block = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        condition = self.time_mlp(sinusoidal_embedding(t, self.time_dim)) + self.class_embedding(labels)
        skip = self.input_block(x)
        h = self.down(skip)
        h = self.middle(h + condition[:, :, None, None])
        h = self.up(h)
        return self.output_block(torch.cat([h, skip], dim=1))


class MaskedConv2d(nn.Conv2d):
    """Raster-order masked convolution for single-channel PixelCNN."""

    def __init__(self, mask_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be A or B")
        mask = torch.ones_like(self.weight)
        center_h, center_w = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        mask[:, :, center_h + 1 :, :] = 0
        mask[:, :, center_h, center_w + (1 if mask_type == "B" else 0) :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(
            x,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class SmallPixelCNN(nn.Module):
    """A compact 16-level grayscale PixelCNN for Fashion-MNIST."""

    def __init__(self, levels: int = 16, hidden: int = 64, depth: int = 7) -> None:
        super().__init__()
        if depth < 3:
            raise ValueError("depth must be at least 3: Mask A input, hidden/output projection, logits")
        self.levels = levels
        self.hidden = hidden
        self.depth = depth
        layers: list[nn.Module] = [MaskedConv2d("A", 1, hidden, 7, padding=3), nn.ReLU()]
        # One Mask-A convolution + (depth - 3) Mask-B convolutions + two
        # 1x1 projections gives exactly ``depth`` convolutional layers.
        for _ in range(depth - 3):
            layers.extend([MaskedConv2d("B", hidden, hidden, 3, padding=1), nn.ReLU()])
        layers.extend([nn.Conv2d(hidden, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, levels, 1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AffineCoupling(nn.Module):
    """RealNVP affine coupling layer with an explicit forward and inverse."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden: int = 64) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2 * dim),
        )

    def _scale_shift(self, masked: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_scale, shift = self.net(masked).chunk(2, dim=-1)
        active = 1.0 - self.mask
        return 1.5 * torch.tanh(log_scale) * active, shift * active

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = x * self.mask
        log_scale, shift = self._scale_shift(masked)
        y = masked + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        return y, log_scale.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = y * self.mask
        log_scale, shift = self._scale_shift(masked)
        x = masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        return x, -log_scale.sum(dim=-1)


class RealNVP(nn.Module):
    def __init__(self, dim: int = 2, layers: int = 6, hidden: int = 64) -> None:
        super().__init__()
        masks = []
        base = torch.tensor([(i % 2) for i in range(dim)], dtype=torch.float32)
        for index in range(layers):
            masks.append(base if index % 2 == 0 else 1.0 - base)
        self.dim = dim
        self.layers = nn.ModuleList([AffineCoupling(dim, mask, hidden) for mask in masks])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(len(z), device=z.device)
        x = z
        for layer in self.layers:
            x, contribution = layer(x)
            log_det = log_det + contribution
        return x, log_det

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(len(x), device=x.device)
        z = x
        for layer in reversed(self.layers):
            z, contribution = layer.inverse(z)
            log_det = log_det + contribution
        return z, log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, inverse_log_det = self.inverse(x)
        base_log_prob = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
        return base_log_prob + inverse_log_det



def diffusion_schedule(steps: int = 100) -> dict[str, torch.Tensor]:
    # With only 100 pedagogical steps, beta_end=0.1 is needed for the final
    # alpha_bar to approach zero. A 1000-step DDPM commonly uses a smaller
    # endpoint because it compounds noise over ten times as many transitions.
    betas = torch.linspace(1e-4, 0.1, steps)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alpha_bars": alpha_bars}
