#!/usr/bin/env python3
"""Execute every notebook and retain an executed copy for classroom fallback."""

from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "01_autoregressive_4_pixel.ipynb",
    "02_vae_latent_walk.ipynb",
    "03_flow_2d.ipynb",
    "04_ebm_two_well.ipynb",
    "05_diffusion_snapshots.ipynb",
    "06_pixelcnn_real_images.ipynb",
]


def main() -> None:
    required = [
        ROOT / "checkpoints" / "vae_mnist_latent2.pt",
        ROOT / "checkpoints" / "diffusion_mnist_tiny.pt",
        ROOT / "data" / "mnist_demo_samples.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing prepared assets:\n" + "\n".join(missing))
    output = ROOT / "executed"
    output.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment["LECTURE2_QUICK"] = "1"
    environment.setdefault("IPYTHONDIR", str(ROOT / ".runtime" / "ipython"))
    environment.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".runtime" / "jupyter"))
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
    for variable in ("IPYTHONDIR", "JUPYTER_CONFIG_DIR", "MPLCONFIGDIR"):
        Path(environment[variable]).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lecture2-current-kernel-") as temporary:
        kernel_name = "lecture2-current-python"
        kernel_dir = Path(temporary) / "kernels" / kernel_name
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Lecture 2 verification (current Python)",
                    "language": "python",
                },
                indent=2,
            )
            + "\n"
        )
        environment["JUPYTER_PATH"] = temporary + os.pathsep + environment.get("JUPYTER_PATH", "")
        for filename in NOTEBOOKS:
            print(f"Executing {filename} with {sys.executable}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--ExecutePreprocessor.timeout=600",
                    f"--ExecutePreprocessor.kernel_name={kernel_name}",
                    "--output-dir",
                    str(output),
                    str(ROOT / filename),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
            )
    print("All six notebooks executed successfully.")


if __name__ == "__main__":
    main()
