# PixelCNN H100 training run

This is the prepared lecturer run for producing the canonical teaching checkpoint. It trains locally on the selected H100 machine; it does not upload or publish anything.

This profile is not the student setup. Use the cluster's supported CUDA/PyTorch environment, or create a fresh environment according to `README.md` and install the PyTorch build recommended for that cluster. Do not activate a lecturer-specific environment name copied from another machine.

## Explicit profile

- GPU: one NVIDIA H100 with 80 GB VRAM
- Dataset: complete Fashion-MNIST training split, with a seeded 55,000/5,000 train/validation split
- Final evaluation: untouched 10,000-image Fashion-MNIST test split
- Image representation: 28×28 grayscale, quantised to 16 categorical levels
- Architecture: Mask-A 7×7 input convolution, Mask-B 3×3 hidden convolutions, 128 hidden channels, 15 layers in total
- Objective: categorical negative log-likelihood at every pixel, reported as bits per dimension
- Optimiser: fused AdamW when supported, learning rate 0.002, weight decay 0.0001
- Schedule: cosine decay over 50 epochs
- Batch size: 1024; incomplete final training batch is dropped to avoid recompilation
- Arithmetic: BF16 autocast, TF32 enabled where PyTorch uses float32 matrix operations
- Layout: channels-last tensors and cuDNN benchmark mode
- Compilation: `torch.compile(mode="max-autotune")`
- Data loading: eight persistent workers with pinned host memory
- Stability: gradient norm clipped to 1.0
- Reproducibility seed: 29
- Outputs: resumable latest checkpoint, validation-selected canonical checkpoint, JSON training summary, unconditional samples, and conditional completions

The planning estimate is approximately 5–20 minutes on an H100, but small 28×28 convolutions can be dominated by framework and compilation overhead. After epoch 1, the trainer prints an estimate based on the actual machine. Treat that measurement as authoritative.

## Preflight

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
python tools/train_pixelcnn.py --profile h100 --show-profile
```

The last command must report `device: cuda` and the expected H100 before starting the job.

## Start or resume

From `lecture 2/demos`:

```bash
bash tools/run_pixelcnn_h100.sh
```

Equivalent direct command:

```bash
python tools/train_pixelcnn.py --profile h100 --resume auto
```

If compilation is unavailable in the installed PyTorch build:

```bash
python tools/train_pixelcnn.py --profile h100 --resume auto --no-compile
```

For a remote session, run the command inside `tmux` or the cluster's persistent job mechanism. Do not rely on an open browser notebook for the long run.

## Outputs to inspect

- `checkpoints/pixelcnn_fashionmnist.pt`: best validation checkpoint; this is the publication candidate
- `checkpoints/pixelcnn_fashionmnist_latest.pt`: resumable checkpoint with optimiser and scheduler state
- `checkpoints/pixelcnn_fashionmnist_training_summary.json`: full profile, hardware report, epoch history, elapsed time, validation BPD, and test BPD
- `fallbacks/06_pixelcnn_samples.png`: unconditional samples
- `fallbacks/06_pixelcnn_completions.png`: multiple lower-half completions for one shared prefix

Before publication, check that validation BPD improved smoothly, the test BPD is close to validation BPD, samples cover multiple clothing categories, and conditional completions respect the observed upper half. A low likelihood alone is not sufficient evidence of visually useful generation.

## Use the checkpoint on the presentation Mac

The H100 requirement applies only to training. The saved weights are portable to MPS or CPU:

```bash
LECTURE2_PIXELCNN_MODE=load \
LECTURE2_PIXELCNN_SAMPLE_COUNT=4 \
LECTURE2_PIXELCNN_COMPLETION_COUNT=4 \
python -m jupyter lab 06_pixelcnn_real_images.ipynb
```

The notebook then loads `checkpoints/pixelcnn_fashionmnist.pt` and does not enter the training loop. Cached sample figures remain the fastest classroom fallback.
