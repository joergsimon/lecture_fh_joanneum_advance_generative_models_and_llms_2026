# Lecture 2 classroom runbook

## Before students enter

1. Set up and activate one of the environments documented in `README.md`. No lecturer-specific environment is required.
2. Run `python tools/verify_demos.py` from this directory.
3. Start Jupyter Lab with `python -m jupyter lab` and open notebooks 01–06.
4. Open GAN Lab in a separate browser tab.
5. Restart each notebook kernel and clear transient variables.

## Live sequence

### Autoregression

- Ask students to calculate `0.8 × 0.75 × 0.7 × 0.9`.
- Run exact likelihood.
- Ask whether “exact likelihood” implies deterministic generation.
- Run two random seeds and change the second pixel.

### VAE

- Show μ and σ before decoding.
- Use the numerical reparameterisation cell.
- Ask where the interpolation between digits 1 and 7 becomes ambiguous.

### GAN Lab

- Use two separated target modes.
- Pause after discriminator-only and generator-only updates.
- Ask for separate fidelity and coverage judgments.

### EBM

- Start at `x=0`, where the gradient vanishes.
- Compare no noise, low noise, and Langevin noise.
- Ask whether the chains cover both wells after burn-in.

### Diffusion

- Pause at the five forward timesteps.
- Separate the known forward process from the learned noise predictor.
- Reveal the fixed reverse trajectory one snapshot at a time.
- Compare unconditional sampling, classifier-free guidance, and guided refinement.

### PixelCNN extension

- Train beforehand with `bash tools/run_pixelcnn_h100.sh`; the job resumes automatically from `checkpoints/pixelcnn_fashionmnist_latest.pt`.
- Inspect `checkpoints/pixelcnn_fashionmnist_training_summary.json`, `fallbacks/06_pixelcnn_samples.png`, and `fallbacks/06_pixelcnn_completions.png` before class.
- Load the validation-selected `checkpoints/pixelcnn_fashionmnist.pt` in class; do not train the H100 profile live.
- Contrast exact image likelihood with stochastic sequential samples.
- Time a full sample and connect the delay to one model call per pixel.
- Use lower-half completion to make the conditional distribution visible.

## Live Moodle checkpoint

- Activity: `Lecture 2 live check: Autoregressive models`.
- Keep it ungraded and available to both lab and remote students.
- Let students answer after the four-pixel blackboard calculation and discuss the result before revealing the explanation.
- The exact question text and answer key are in `MOODLE_LIVE_ACTIVITY.md`.

## Fallbacks

If a kernel or browser fails, use the PNG files in `fallbacks/`. The relevant lecture slides contain the same conceptual sequence.
