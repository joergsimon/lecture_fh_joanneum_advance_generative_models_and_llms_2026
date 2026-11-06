"""Reproducible PixelCNN training profiles used by the notebook and CLI.

The ``h100`` profile is intentionally explicit: it records the architectural,
optimisation, precision, data-loading, checkpointing, and sampling choices in
the saved checkpoint.  Training and publishing remain separate operations.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from demo_models import SmallPixelCNN, load_idx_images, load_idx_labels


@dataclass(frozen=True)
class PixelCNNProfile:
    name: str
    description: str
    levels: int
    hidden: int
    depth: int
    epochs: int
    limit: int | None
    validation_examples: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    use_bfloat16: bool
    use_compile: bool
    channels_last: bool
    sample_count: int
    completion_count: int
    downsample: int = 1


PIXELCNN_PROFILES: dict[str, PixelCNNProfile] = {
    "smoke": PixelCNNProfile(
        name="smoke",
        description="CPU-safe pipeline verification; outputs are not model-quality evidence.",
        levels=16,
        hidden=16,
        depth=3,
        epochs=1,
        limit=512,
        validation_examples=64,
        batch_size=64,
        learning_rate=2e-3,
        weight_decay=0.0,
        num_workers=0,
        use_bfloat16=False,
        use_compile=False,
        channels_last=False,
        sample_count=1,
        completion_count=1,
        downsample=2,
    ),
    "local": PixelCNNProfile(
        name="local",
        description="Laptop/lab baseline for mechanism exploration.",
        levels=16,
        hidden=64,
        depth=7,
        epochs=8,
        limit=30_000,
        validation_examples=3_000,
        batch_size=128,
        learning_rate=2e-3,
        weight_decay=1e-4,
        num_workers=2,
        use_bfloat16=False,
        use_compile=False,
        channels_last=False,
        sample_count=4,
        completion_count=4,
    ),
    "h100": PixelCNNProfile(
        name="h100",
        description="Full Fashion-MNIST quality run for one NVIDIA H100 (80 GB).",
        levels=16,
        hidden=128,
        depth=15,
        epochs=50,
        limit=None,
        validation_examples=5_000,
        batch_size=1024,
        learning_rate=2e-3,
        weight_decay=1e-4,
        num_workers=8,
        use_bfloat16=True,
        use_compile=True,
        channels_last=True,
        sample_count=12,
        completion_count=6,
    ),
}


@dataclass
class PixelCNNData:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    test_labels: np.ndarray
    image_size: tuple[int, int]
    train_examples: int
    validation_examples: int
    test_examples: int


@dataclass
class PixelCNNTrainingResult:
    model: SmallPixelCNN
    data: PixelCNNData
    profile: PixelCNNProfile
    device: torch.device
    history: list[dict[str, float]]
    best_checkpoint: Path
    latest_checkpoint: Path
    best_validation_bpd: float
    test_bpd: float
    elapsed_seconds: float


def load_pixelcnn_for_demo(
    root: Path,
    checkpoint_path: Path | None = None,
    *,
    device: torch.device | None = None,
) -> PixelCNNTrainingResult:
    """Load a trained checkpoint on CUDA, MPS, or CPU without retraining.

    This is deliberately separate from the H100 training guard: a checkpoint
    trained on the H100 must remain usable on the lecturer's MacBook.
    """
    checkpoint_path = checkpoint_path or root / "checkpoints" / "pixelcnn_fashionmnist.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    profile = PixelCNNProfile(**checkpoint["profile"])
    device = device or choose_device()
    model = SmallPixelCNN(
        levels=checkpoint["config"]["levels"],
        hidden=checkpoint["config"]["hidden"],
        depth=checkpoint["config"]["depth"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    # Data loading for a live demo should be conservative even when the saved
    # training profile used many H100 workers and a very large batch.
    demo_profile = replace(
        profile,
        batch_size=min(profile.batch_size, 256),
        num_workers=0,
        use_compile=False,
        use_bfloat16=False,
        channels_last=False,
    )
    data = prepare_fashion_mnist(root, demo_profile, int(checkpoint.get("seed", 29)))
    history = list(checkpoint.get("history", []))
    return PixelCNNTrainingResult(
        model=model,
        data=data,
        profile=profile,
        device=device,
        history=history,
        best_checkpoint=checkpoint_path,
        latest_checkpoint=root / "checkpoints" / "pixelcnn_fashionmnist_latest.pt",
        best_validation_bpd=float(checkpoint.get("best_validation_bpd", float("nan"))),
        test_bpd=float(checkpoint.get("test_bpd", float("nan"))),
        elapsed_seconds=float(checkpoint.get("elapsed_seconds", 0.0)),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_profile(name: str, **overrides: Any) -> PixelCNNProfile:
    if name not in PIXELCNN_PROFILES:
        raise KeyError(f"Unknown profile {name!r}; choose from {sorted(PIXELCNN_PROFILES)}")
    allowed = {key: value for key, value in overrides.items() if value is not None}
    return replace(PIXELCNN_PROFILES[name], **allowed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(
            gpu_name=properties.name,
            compute_capability=f"{properties.major}.{properties.minor}",
            total_memory_gib=round(properties.total_memory / 2**30, 2),
        )
    elif device.type == "mps":
        report["gpu_name"] = "Apple Metal Performance Shaders"
    else:
        report["gpu_name"] = "CPU"
    return report


def _raw_fashion_mnist(root: Path) -> Path:
    candidates = [root / "data" / "FashionMNIST" / "raw"]
    override = os.environ.get("LECTURE2_FASHION_MNIST_RAW")
    if override:
        candidates.insert(0, Path(override).expanduser())
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise FileNotFoundError(
            "Fashion-MNIST is not cached and torchvision is unavailable. "
            "Install torchvision or copy the four uncompressed IDX files to data/FashionMNIST/raw."
        ) from exc
    FashionMNIST(root=root / "data", train=True, download=True)
    FashionMNIST(root=root / "data", train=False, download=True)
    return root / "data" / "FashionMNIST" / "raw"


def _quantise(images: np.ndarray, levels: int, downsample: int) -> tuple[torch.Tensor, torch.Tensor]:
    if downsample > 1:
        images = images[:, ::downsample, ::downsample]
    targets = torch.tensor(np.rint(images / 255 * (levels - 1)), dtype=torch.long)
    inputs = targets.float().unsqueeze(1) / (levels - 1) * 2 - 1
    return inputs, targets


def prepare_fashion_mnist(root: Path, profile: PixelCNNProfile, seed: int = 29) -> PixelCNNData:
    raw = _raw_fashion_mnist(root)
    train_images = load_idx_images(raw / "train-images-idx3-ubyte")
    train_labels = load_idx_labels(raw / "train-labels-idx1-ubyte")
    test_images = load_idx_images(raw / "t10k-images-idx3-ubyte")
    test_labels = load_idx_labels(raw / "t10k-labels-idx1-ubyte")
    if profile.limit is not None:
        train_images = train_images[: profile.limit]
        train_labels = train_labels[: profile.limit]

    all_inputs, all_targets = _quantise(train_images, profile.levels, profile.downsample)
    test_inputs, test_targets = _quantise(test_images, profile.levels, profile.downsample)
    validation_count = min(profile.validation_examples, max(1, len(all_inputs) // 5))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(all_inputs), generator=generator)
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    train_dataset = TensorDataset(all_inputs[train_indices], all_targets[train_indices])
    validation_dataset = TensorDataset(all_inputs[validation_indices], all_targets[validation_indices])
    test_dataset = TensorDataset(test_inputs, test_targets)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": profile.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": profile.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        drop_last=profile.name == "h100",
        generator=torch.Generator().manual_seed(seed + 1),
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return PixelCNNData(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        test_inputs=test_inputs,
        test_targets=test_targets,
        test_labels=test_labels,
        image_size=tuple(all_targets.shape[1:]),
        train_examples=len(train_dataset),
        validation_examples=len(validation_dataset),
        test_examples=len(test_dataset),
    )


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _bits_per_dimension(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_bfloat16: bool,
    channels_last: bool,
) -> float:
    # Current nightly PyTorch/MPS occasionally returns a nonsensical batched
    # reduction immediately after optimiser steps, while the same saved model
    # evaluates correctly on CPU.  Use an isolated CPU snapshot for metrics on
    # MPS.  CUDA/H100 evaluation stays on-device.
    evaluation_model = model
    evaluation_device = device
    evaluation_bfloat16 = use_bfloat16
    evaluation_channels_last = channels_last
    if device.type == "mps":
        torch.mps.synchronize()
        evaluation_model = copy.deepcopy(model).to("cpu")
        evaluation_device = torch.device("cpu")
        evaluation_bfloat16 = False
        evaluation_channels_last = False
    evaluation_model.eval()
    total_nll = 0.0
    pixels = 0
    with torch.inference_mode():
        for x, target in loader:
            x = x.to(evaluation_device, non_blocking=True)
            target = target.to(evaluation_device, non_blocking=True)
            if evaluation_channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with _autocast(evaluation_device, evaluation_bfloat16):
                logits = evaluation_model(x)
                loss = nn.functional.cross_entropy(logits.float(), target, reduction="sum")
            # Explicit host transfer avoids a rare asynchronous MPS scalar-read
            # anomaly observed during full-test evaluation.
            total_nll += loss.detach().cpu().item()
            pixels += target.numel()
    return total_nll / pixels / math.log(2)


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_pixelcnn(
    root: Path,
    profile: PixelCNNProfile,
    *,
    seed: int = 29,
    resume: str | Path = "none",
    enable_compile: bool | None = None,
    progress: bool = True,
) -> PixelCNNTrainingResult:
    seed_everything(seed)
    device = choose_device()
    if profile.name == "h100" and device.type != "cuda":
        raise RuntimeError("The h100 profile requires CUDA. Use profile='local' on this machine.")
    if profile.name == "h100":
        properties = torch.cuda.get_device_properties(device)
        memory_gib = properties.total_memory / 2**30
        if "H100" not in properties.name.upper() or memory_gib < 70:
            raise RuntimeError(
                "The explicit h100 profile requires an NVIDIA H100 with approximately 80 GB VRAM; "
                f"detected {properties.name} with {memory_gib:.1f} GiB. Select another profile or "
                "create a deliberate override instead of silently changing the experiment."
            )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data = prepare_fashion_mnist(root, profile, seed)
    base_model = SmallPixelCNN(
        levels=profile.levels,
        hidden=profile.hidden,
        depth=profile.depth,
    ).to(device)
    if profile.channels_last:
        base_model = base_model.to(memory_format=torch.channels_last)
    try:
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=profile.learning_rate,
            weight_decay=profile.weight_decay,
            fused=device.type == "cuda",
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            base_model.parameters(),
            lr=profile.learning_rate,
            weight_decay=profile.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=profile.epochs)

    prefix = {
        "smoke": "pixelcnn_fashionmnist_smoke",
        "local": "pixelcnn_fashionmnist_local",
        "h100": "pixelcnn_fashionmnist",
    }[profile.name]
    best_checkpoint = root / "checkpoints" / f"{prefix}.pt"
    latest_checkpoint = root / "checkpoints" / f"{prefix}_latest.pt"
    start_epoch = 0
    history: list[dict[str, float]] = []
    best_validation_bpd = float("inf")
    resume_path: Path | None = None
    if str(resume) == "auto" and latest_checkpoint.exists():
        resume_path = latest_checkpoint
    elif str(resume) not in {"none", "", "False"}:
        resume_path = Path(resume)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        base_model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        history = list(checkpoint.get("history", []))
        best_validation_bpd = float(checkpoint.get("best_validation_bpd", float("inf")))
        if progress:
            print(f"resuming {resume_path} at epoch {start_epoch + 1}")

    compile_requested = profile.use_compile if enable_compile is None else enable_compile
    training_model: nn.Module = base_model
    if compile_requested:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile; pass --no-compile.")
        training_model = torch.compile(base_model, mode="max-autotune")

    run_started = time.perf_counter()
    run_metadata = {
        "started_at": utc_now(),
        "profile": asdict(profile),
        "device": device_report(device),
        "seed": seed,
        "compile_enabled": compile_requested,
        "train_examples": data.train_examples,
        "validation_examples": data.validation_examples,
        "test_examples": data.test_examples,
        "image_size": list(data.image_size),
    }
    if progress:
        print(json.dumps(run_metadata, indent=2))

    for epoch in range(start_epoch, profile.epochs):
        epoch_started = time.perf_counter()
        training_model.train()
        total_nll = 0.0
        pixels = 0
        for step, (x, target) in enumerate(data.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if profile.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, profile.use_bfloat16):
                logits = training_model(x)
                loss = nn.functional.cross_entropy(logits.float(), target)
            loss.backward()
            nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
            optimizer.step()
            total_nll += loss.detach().cpu().item() * target.numel()
            pixels += target.numel()
            if progress and step % 50 == 0:
                print(f"epoch {epoch + 1:03d} step {step:04d}/{len(data.train_loader):04d} bpd={total_nll / pixels / math.log(2):.3f}")
        scheduler.step()
        train_bpd = total_nll / pixels / math.log(2)
        validation_bpd = _bits_per_dimension(
            base_model,
            data.validation_loader,
            device,
            profile.use_bfloat16,
            profile.channels_last,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch + 1,
                "train_bpd": train_bpd,
                "validation_bpd": validation_bpd,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": epoch_seconds,
            }
        )
        checkpoint_payload = {
            "state_dict": _cpu_state_dict(base_model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "profile": asdict(profile),
            "config": {
                "levels": profile.levels,
                "hidden": profile.hidden,
                "depth": profile.depth,
                "image_size": list(data.image_size),
            },
            "history": history,
            "best_validation_bpd": min(best_validation_bpd, validation_bpd),
            "dataset": "Fashion-MNIST",
            "quantisation": f"{profile.levels} uniform grayscale levels",
            "seed": seed,
            "run_metadata": run_metadata,
        }
        _atomic_torch_save(checkpoint_payload, latest_checkpoint)
        if validation_bpd < best_validation_bpd:
            best_validation_bpd = validation_bpd
            checkpoint_payload["best_validation_bpd"] = best_validation_bpd
            public_payload = {
                key: value
                for key, value in checkpoint_payload.items()
                if key not in {"optimizer_state_dict", "scheduler_state_dict"}
            }
            _atomic_torch_save(public_payload, best_checkpoint)
        elapsed = time.perf_counter() - run_started
        completed = epoch - start_epoch + 1
        estimated_total = elapsed / completed * (profile.epochs - start_epoch)
        if progress:
            print(
                f"epoch {epoch + 1:03d}/{profile.epochs}: train={train_bpd:.3f} bpd, "
                f"val={validation_bpd:.3f} bpd, {epoch_seconds:.1f}s; "
                f"estimated run total {estimated_total / 60:.1f} min"
            )

    best = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    base_model.load_state_dict(best["state_dict"])
    test_bpd = _bits_per_dimension(
        base_model,
        data.test_loader,
        device,
        profile.use_bfloat16,
        profile.channels_last,
    )
    elapsed_seconds = time.perf_counter() - run_started
    best["test_bpd"] = test_bpd
    best["elapsed_seconds"] = elapsed_seconds
    best["finished_at"] = utc_now()
    _atomic_torch_save(best, best_checkpoint)
    summary = {
        **run_metadata,
        "finished_at": best["finished_at"],
        "elapsed_seconds": elapsed_seconds,
        "best_validation_bpd": best_validation_bpd,
        "test_bpd": test_bpd,
        "best_checkpoint": str(best_checkpoint.relative_to(root)),
        "latest_checkpoint": str(latest_checkpoint.relative_to(root)),
        "history": history,
    }
    summary_path = root / "checkpoints" / f"{prefix}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if progress:
        print(json.dumps({"best_validation_bpd": best_validation_bpd, "test_bpd": test_bpd, "elapsed_minutes": elapsed_seconds / 60}, indent=2))
    return PixelCNNTrainingResult(
        model=base_model,
        data=data,
        profile=profile,
        device=device,
        history=history,
        best_checkpoint=best_checkpoint,
        latest_checkpoint=latest_checkpoint,
        best_validation_bpd=best_validation_bpd,
        test_bpd=test_bpd,
        elapsed_seconds=elapsed_seconds,
    )


@torch.inference_mode()
def sample_pixelcnn(
    model: nn.Module,
    device: torch.device,
    levels: int,
    shape: tuple[int, int],
    *,
    count: int,
    temperature: float = 0.9,
    seed: int = 30,
    prefix: torch.Tensor | None = None,
    known_rows: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    height, width = shape
    canvas = torch.zeros(count, 1, height, width, device=device)
    if prefix is not None and known_rows:
        canvas[:, :, :known_rows] = prefix[:count, :, :known_rows].to(device)
    model.eval()
    for row in range(known_rows, height):
        for column in range(width):
            logits = model(canvas)[:, :, row, column].float() / temperature
            probabilities = torch.softmax(logits, dim=1).cpu()
            value = torch.multinomial(probabilities, 1, generator=generator).squeeze(1).to(device)
            canvas[:, 0, row, column] = value.float() / (levels - 1) * 2 - 1
    return canvas.cpu()


def render_sampling_artifacts(result: PixelCNNTrainingResult, root: Path) -> list[Path]:
    output = root / "fallbacks"
    output.mkdir(parents=True, exist_ok=True)
    profile = result.profile
    shape = result.data.image_size
    samples = sample_pixelcnn(
        result.model,
        result.device,
        profile.levels,
        shape,
        count=profile.sample_count,
        temperature=0.9,
        seed=30,
    )
    cols = min(6, profile.sample_count)
    rows = math.ceil(profile.sample_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image in zip(axes.flat, samples):
        ax.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
    fig.suptitle(f"PixelCNN samples · {profile.name} profile · {result.test_bpd:.3f} test bpd")
    fig.tight_layout()
    sample_path = output / "06_pixelcnn_samples.png"
    fig.savefig(sample_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    count = profile.completion_count
    prefix = result.data.test_inputs[:1].repeat(count, 1, 1, 1)
    known_rows = shape[0] // 2
    completions = sample_pixelcnn(
        result.model,
        result.device,
        profile.levels,
        shape,
        count=count,
        temperature=0.85,
        seed=45,
        prefix=prefix,
        known_rows=known_rows,
    )
    fig, axes = plt.subplots(1, count + 1, figsize=(2 * (count + 1), 2.3))
    axes[0].imshow(prefix[0, 0], cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("source")
    for index, image in enumerate(completions, start=1):
        axes[index].imshow(image[0], cmap="gray", vmin=-1, vmax=1)
        axes[index].set_title(f"sample {index}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("One observed upper half; multiple sampled lower halves")
    fig.tight_layout()
    completion_path = output / "06_pixelcnn_completions.png"
    fig.savefig(completion_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [sample_path, completion_path]
