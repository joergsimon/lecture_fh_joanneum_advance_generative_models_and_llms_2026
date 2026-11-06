# Lecture 2 live demonstrations

These notebooks support the mechanisms shown in Lecture 2. They are deliberately small, offline-first, and independent of cloud APIs.

## Recommended order

1. `01_autoregressive_4_pixel.ipynb` — exact likelihood versus sequential sampling.
2. `02_vae_latent_walk.ipynb` — encoding, reparameterisation, and latent interpolation.
3. `03_flow_2d.ipynb` — train a 2D RealNVP, animate forward/inverse maps, then test an 8D flow.
4. GAN Lab in the browser — <https://poloclub.github.io/ganlab/>.
5. `04_ebm_two_well.ipynb` — understand the EBM objective, then animate persistent Langevin particles.
6. `05_diffusion_snapshots.ipynb` — forward corruption, reverse sampling, classifier-free guidance, and guided refinement.
7. `06_pixelcnn_real_images.ipynb` — train and save a real-image PixelCNN on Fashion-MNIST.

## Environment setup

No pre-existing or lecturer-specific environment is assumed. Start in the course repository and change to the demo directory:

```bash
cd lectures/lecture_02/demos
```

### Option A: uv (recommended)

```bash
uv venv --python 3.11
source .venv/bin/activate          # macOS or Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -r requirements-lock.txt
python -m jupyter lab
```

### Option B: Conda or Miniforge

Create the supplied environment before activating it:

```bash
conda env create -f environment.yml
conda activate lecture2-generative-demos
python -m jupyter lab
```

To update an environment that you already created from this file:

```bash
conda env update -f environment.yml --prune
```

### Option C: standard Python virtual environment

macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m jupyter lab
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m jupyter lab
```

You do not need to install a global Jupyter kernel. Start Jupyter with `python -m jupyter lab` from the selected environment and choose **Python 3 (ipykernel)** if Jupyter asks for a kernel.

Verify the environment before opening the notebooks:

```bash
python -c "import torch, torchvision, numpy, matplotlib; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('MPS', torch.backends.mps.is_available())"
python tools/verify_demos.py
```

`requirements-lock.txt` records the exact versions used for the final clean-environment validation. `requirements.txt` remains the shorter compatibility specification if the lock file does not support a future platform.

The verifier creates temporary Jupyter configuration under `.runtime/`, selects the current environment's Python kernel explicitly, enables the CPU-safe quick profile, and stores executed copies in `executed/`.

Notebooks 1–5 need no network connection after the environment has been created. The VAE and diffusion notebooks load compact checkpoints from `checkpoints/`; they do not train during class. The PixelCNN notebook needs one Fashion-MNIST download unless `data/FashionMNIST/raw` already contains the dataset.

Every trainable notebook chooses `cuda`, then Apple `mps`, then `cpu`. PixelCNN has three explicit profiles: `smoke`, `local`, and `h100`. Students can set `LECTURE2_QUICK=1` for the CPU-safe smoke test; the default notebook profile is `local`. The `h100` profile is the prepared lecturer job and requires CUDA. PyTorch uses Apple's Metal backend through `mps`; a separate MLX install is not required.

### Prepared H100 PixelCNN run

The complete configuration and operational checklist are in `PIXELCNN_H100_RUN.md`. From this directory:

```bash
python tools/train_pixelcnn.py --profile h100 --show-profile
bash tools/run_pixelcnn_h100.sh
```

The H100 profile uses the full Fashion-MNIST dataset, 128 hidden channels, 15 layers, batch size 1024, 50 epochs, BF16, channels-last tensors, `torch.compile`, validation-selected best weights, a resumable latest checkpoint, and final test evaluation. A planning estimate is 5–20 minutes; the trainer reports a machine-specific estimate after epoch 1. Training does not publish anything.

To use the resulting H100 checkpoint later on the presentation Mac without retraining, start Jupyter with `LECTURE2_PIXELCNN_MODE=load`. For a short live sample, also set `LECTURE2_PIXELCNN_SAMPLE_COUNT=4` and `LECTURE2_PIXELCNN_COMPLETION_COUNT=4`.

## Student pathway

Each notebook follows the same pattern: **question → mechanism → executable example → interpretation → limitation → small task**. Students should first run the fixed-seed version, predict one change, then alter exactly one parameter and record what changed. The notebooks distinguish a mechanism demonstration from evidence of model quality.

Suggested low-cost exploration:

- Autoregression: change one observed pixel and identify only the affected factors.
- VAE: compare interpolation through a dense and a sparse latent region.
- Flow: change the coupling-layer count; compare fit and inverse error.
- EBM: change the Langevin step size and noise; inspect mixing rather than only final samples.
- Diffusion: compare guidance strengths from the same starting noise.
- PixelCNN: use quick mode locally; use the published full checkpoint for sampling and completion.

## Interactive instrument

`intuition/index.html` runs the same mechanisms in the browser: exact likelihood and raster sampling, a live two-dimensional VAE, the scale-and-shear maps plus a training RealNVP, Langevin on the double well and a learned energy, and diffusion’s forward draw, noise predictor, reverse chain, and guidance.

The plate is WebGL. The numbers in the margin are the model’s actual factors, log-determinants, energies, and noise errors. Pause leaves a real intermediate state on screen. Space pauses; keys 1 to 5 switch demonstrations.

Open the file from Jupyter, or from this directory:

```bash
python -m http.server 8765
```

Then visit <http://localhost:8765/intuition/>. Training inside the page is a mechanism demonstration, not a claim about sample quality.

## Animation principle

Where a process changes over time, the notebooks should animate real intermediate states rather than only moving illustrative shapes. The flow animation records outputs after actual coupling layers and their inverses; the EBM animation records learned energy curves and persistent particles; the diffusion sequence uses real forward and reverse states. Every animation should also expose static checkpoints so it can be paused, explained, and exported meaningfully to PDF. If an animation uses interpolation only for intuition, label it explicitly as interpolation rather than implying that it is a model operation.

## Reliability

- Execute `tools/verify_demos.py` before class. Verification automatically uses a reduced PixelCNN smoke test.
- Use the fixed seed first, then change it deliberately.
- Every notebook writes a rendered fallback to `fallbacks/`.
- Executed notebook copies are stored in `executed/` after verification.
- The models demonstrate mechanisms. A single trajectory or image is not evidence of general model quality.

## Regenerating checkpoints

Checkpoint training is not part of the classroom workflow. To regenerate it from the prepared data on another machine:

```bash
python tools/train_checkpoints.py
```

Place the four uncompressed MNIST IDX files in `data/MNIST/raw`, or set `LECTURE2_MNIST_RAW` to the directory containing them. The script reports the expected filenames when they are missing.

## Publishing the PixelCNN weights

After the H100 run has finished and its likelihoods and sample grids have been inspected, authenticate with Hugging Face and publish the checkpoint:

```bash
python tools/publish_huggingface.py joergsimon/fh-joanneum-pixelcnn-fashionmnist
```

The command uploads the full checkpoint, architecture code, model card, and sample figures. It refuses to publish the smoke-test checkpoint.

Publishing is deliberately separate from training: first inspect the full checkpoint and its sample figures, then run the upload command with the chosen repository ID.
