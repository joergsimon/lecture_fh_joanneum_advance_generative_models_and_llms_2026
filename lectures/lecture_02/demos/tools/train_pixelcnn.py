#!/usr/bin/env python3
"""Train the Lecture 2 PixelCNN with an explicit reproducible profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelcnn_training import (  # noqa: E402
    PIXELCNN_PROFILES,
    device_report,
    choose_device,
    render_sampling_artifacts,
    resolve_profile,
    train_pixelcnn,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a masked-convolution PixelCNN on Fashion-MNIST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=sorted(PIXELCNN_PROFILES), default="local")
    parser.add_argument("--resume", default="none", help="none, auto, or a checkpoint path")
    parser.add_argument("--epochs", type=int, help="Override the selected profile")
    parser.add_argument("--batch-size", type=int, help="Override the selected profile")
    parser.add_argument("--limit", type=int, help="Override dataset limit; omit for the profile default")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile even if the profile enables it")
    parser.add_argument("--skip-samples", action="store_true", help="Train and evaluate without generating final sample grids")
    parser.add_argument("--show-profile", action="store_true", help="Print the resolved profile and device, then exit")
    args = parser.parse_args()

    profile = resolve_profile(
        args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if args.show_profile:
        print(json.dumps({"profile": asdict(profile), "device": device_report(choose_device())}, indent=2))
        return
    result = train_pixelcnn(
        ROOT,
        profile,
        resume=args.resume,
        enable_compile=False if args.no_compile else None,
    )
    if not args.skip_samples:
        paths = render_sampling_artifacts(result, ROOT)
        for path in paths:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
