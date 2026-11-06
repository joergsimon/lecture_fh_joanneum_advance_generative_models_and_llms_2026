#!/usr/bin/env python3
"""Publish the full PixelCNN checkpoint and teaching assets to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import tempfile
import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face repository, e.g. username/fh-joanneum-pixelcnn-fashionmnist")
    parser.add_argument("--private", action="store_true", help="Create or keep the repository private")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: python -m pip install huggingface_hub") from exc

    checkpoint = ROOT / "checkpoints" / "pixelcnn_fashionmnist.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} does not exist. Run 06_pixelcnn_real_images.ipynb in full mode before publishing."
        )
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    profile_name = checkpoint_data.get("profile", {}).get("name")
    if profile_name != "h100":
        raise RuntimeError(
            f"Refusing to publish {checkpoint}: expected profile 'h100', found {profile_name!r}. "
            "Train and inspect the canonical H100 checkpoint first."
        )

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pixelcnn-release-") as temporary:
        release = Path(temporary)
        shutil.copy2(checkpoint, release / checkpoint.name)
        shutil.copy2(ROOT / "demo_models.py", release / "demo_models.py")
        shutil.copy2(ROOT / "pixelcnn_training.py", release / "pixelcnn_training.py")
        shutil.copy2(ROOT / "MODEL_CARD_PIXELCNN.md", release / "README.md")
        summary = ROOT / "checkpoints" / "pixelcnn_fashionmnist_training_summary.json"
        if summary.exists():
            shutil.copy2(summary, release / summary.name)
        for name in ["06_pixelcnn_samples.png", "06_pixelcnn_completions.png"]:
            source = ROOT / "fallbacks" / name
            if source.exists():
                shutil.copy2(source, release / name)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=release,
            commit_message="Publish FH JOANNEUM PixelCNN teaching checkpoint",
        )
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
