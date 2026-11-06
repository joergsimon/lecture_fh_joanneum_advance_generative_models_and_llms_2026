#!/usr/bin/env python3
"""Build the six Lecture 2 notebooks from readable cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def notebook(title: str, purpose: str, cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n{purpose}\n\n"
            "**Classroom pattern:** predict first, run one cell, inspect the intermediate state, then change one parameter. "
            "All random seeds are fixed for the first run."
        ),
        nbf.v4.new_markdown_cell(
            "## Before you run\n\n"
            "- Use the environment from `README.md`; no lecturer-specific Conda environment is required.\n"
            "- Set `LECTURE2_QUICK=1` before starting Jupyter for the CPU-safe verification path.\n"
            "- CUDA is selected first, then Apple MPS, then CPU. A separate MLX installation is not required.\n"
            "- Read the prediction prompt before each experiment. The fixed-seed result is a mechanism check, not a quality benchmark.\n"
            "- Every generated file stays inside `fallbacks/` or `checkpoints/`. No cell uploads or publishes anything."
        ),
    ]
    for kind, source in cells:
        nb.cells.append(nbf.v4.new_markdown_cell(source) if kind == "md" else nbf.v4.new_code_cell(source))
    return nb


SETUP = """from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "demo_models.py").exists():
    candidate = ROOT / "lecture 2" / "demos"
    if candidate.exists():
        ROOT = candidate
sys.path.insert(0, str(ROOT))
(ROOT / "fallbacks").mkdir(exist_ok=True)
print(f"Demo root: {ROOT}")"""


NOTEBOOKS: dict[str, nbf.NotebookNode] = {}


NOTEBOOKS["01_autoregressive_4_pixel.ipynb"] = notebook(
    "Four-pixel autoregressive model",
    "Make chain-rule factorisation, exact likelihood evaluation, and sequential stochastic sampling executable.",
    [
        ("code", SETUP + "\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nnp.set_printoptions(precision=3, suppress=True)"),
        ("md", "## 1. One conditional distribution per position\n\nThe functions below return the probability that the next binary pixel equals 1. The prefix determines the conditional."),
        ("code", """def p_one(prefix):
    if len(prefix) == 0:
        return 0.80
    if len(prefix) == 1:
        return 0.25 if prefix[0] == 1 else 0.55
    if len(prefix) == 2:
        return 0.70 if prefix == [1, 0] else 0.40
    if len(prefix) == 3:
        table = {(1, 0, 1): 0.90, (1, 1, 1): 0.45, (1, 0, 0): 0.30}
        return table.get(tuple(prefix), 0.50)
    raise ValueError("This demo contains four pixels")

def probability_of_value(value, prefix):
    p1 = p_one(prefix)
    return p1 if value == 1 else 1.0 - p1"""),
        ("md", "## 2. Exact likelihood for a fixed observed image\n\nPredict the product before running the cell. Exact means that every normalised factor can be evaluated without Monte Carlo approximation."),
        ("code", """observed = [1, 0, 1, 1]
factors = [probability_of_value(value, observed[:i]) for i, value in enumerate(observed)]
joint = float(np.prod(factors))
nll = -float(np.log(joint))
print("observed image:", observed)
print("factors:       ", factors)
print(f"p(x) = {' × '.join(f'{p:.2f}' for p in factors)} = {joint:.3f}")
print(f"negative log-likelihood = {nll:.3f}")"""),
        ("md", "## 3. Sequential sampling\n\nAt generation time the model samples one pixel, appends it to the context, and asks the next conditional question."),
        ("code", """def sample_image(seed):
    rng = np.random.default_rng(seed)
    prefix, trace = [], []
    for position in range(4):
        p1 = p_one(prefix)
        value = int(rng.random() < p1)
        prefix.append(value)
        trace.append({"position": position + 1, "p_one": p1, "sample": value, "state": prefix.copy()})
    return prefix, trace

sample_a, trace_a = sample_image(4)
sample_b, trace_b = sample_image(8)
print("seed 4:", sample_a)
print("seed 8:", sample_b)
for row in trace_a:
    print(row)"""),
        ("code", """fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))
axes[0].imshow(np.full((2, 2), np.nan), vmin=0, vmax=1, cmap="Blues")
axes[0].set_title("start")
for ax, row in zip(axes[1:], trace_a):
    canvas = np.full(4, np.nan)
    canvas[:len(row["state"])] = row["state"]
    ax.imshow(canvas.reshape(2, 2), vmin=0, vmax=1, cmap="Blues")
    ax.set_title(f"sample x{row['position']}\\np(1)={row['p_one']:.2f}")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("One random draw extends the observed prefix")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "01_autoregressive_sampling.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 4. Intervention: change the second pixel\n\nThe first two factors remain local to their positions, while later conditionals can change because their context changed."),
        ("code", """for prefix in ([1, 0], [1, 1]):
    print(f"prefix {prefix}: p(x3=1 | prefix)={p_one(list(prefix)):.2f}")
print("This is dependency through the conditional context, not a change to the chain rule itself.")"""),
    ],
)


NOTEBOOKS["02_vae_latent_walk.ipynb"] = notebook(
    "VAE latent walk",
    "Use a prepared two-dimensional MNIST VAE to expose encoding, reparameterisation, decoding, and latent interpolation.",
    [
        ("code", SETUP + "\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nfrom demo_models import TinyVAE\ntorch.manual_seed(7)"),
        ("md", "## 1. Load the small offline checkpoint\n\nNo training occurs during class. The checkpoint was trained locally on MNIST with a two-dimensional latent space."),
        ("code", """checkpoint = torch.load(ROOT / "checkpoints" / "vae_mnist_latent2.pt", map_location="cpu", weights_only=True)
model = TinyVAE(latent_dim=checkpoint["latent_dim"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
demo = np.load(ROOT / "data" / "mnist_demo_samples.npz")
images = torch.tensor(demo["images"], dtype=torch.float32)
labels = demo["labels"]
print("training loss, last epoch:", round(checkpoint["history"]["loss"][-1], 3))
print("available fixed examples:", labels.tolist())"""),
        ("md", "## 2. Encoder output and reconstruction\n\nThe encoder produces μ and log σ². The decoder reconstructs a sample z. For a stable live comparison, this cell decodes μ directly."),
        ("code", """with torch.no_grad():
    mu, logvar = model.encode(images)
    recon = model.decode(mu)

fig, axes = plt.subplots(2, 10, figsize=(13, 3.1))
for i in range(10):
    axes[0, i].imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(str(labels[i]))
    axes[0, i].axis("off"); axes[1, i].axis("off")
axes[0, 0].set_ylabel("data"); axes[1, 0].set_ylabel("decode μ")
fig.suptitle("A two-dimensional bottleneck forces a smooth but imperfect representation")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "02_vae_reconstructions.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 3. Reparameterisation with concrete numbers"),
        ("code", """index = 7
epsilon = torch.tensor([[-0.5, 0.8]])
sigma = torch.exp(0.5 * logvar[index:index+1])
z = mu[index:index+1] + sigma * epsilon
print("digit:", labels[index])
print("mu:     ", mu[index].numpy().round(3))
print("sigma:  ", sigma[0].numpy().round(3))
print("epsilon:", epsilon[0].numpy())
print("z:      ", z[0].numpy().round(3))"""),
        ("md", "## 4. Latent interpolation\n\nPredict where identity becomes ambiguous. Linear interpolation may pass through a low-density region even when both endpoints reconstruct well."),
        ("code", """left, right = 1, 7
weights = torch.linspace(0, 1, 9)[:, None]
z_path = (1 - weights) * mu[left] + weights * mu[right]
with torch.no_grad():
    decoded = model.decode(z_path)

fig, axes = plt.subplots(1, 9, figsize=(12, 1.8))
for i, ax in enumerate(axes):
    ax.imshow(decoded[i, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"α={weights[i,0]:.2f}", fontsize=8)
    ax.axis("off")
fig.suptitle(f"Decode the line between digit {labels[left]} and digit {labels[right]}")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "02_vae_latent_walk.png", dpi=180, bbox_inches="tight")
plt.show()
print("endpoint coordinates:", mu[left].numpy().round(2), "→", mu[right].numpy().round(2))"""),
    ],
)


NOTEBOOKS["03_flow_2d.ipynb"] = notebook(
    "Two-dimensional normalising flow",
    "Make invertibility, density correction, and the Jacobian determinant visible with two explicit transformations.",
    [
        ("code", SETUP + "\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nrng = np.random.default_rng(12)"),
        ("md", "## 1. Base samples and two invertible maps\n\nThe first map doubles the horizontal axis. The second adds a shear with determinant one."),
        ("code", """z = rng.normal(size=(1200, 2))

def scale_forward(points, scale=2.0):
    out = points.copy(); out[:, 0] *= scale
    return out

def scale_inverse(points, scale=2.0):
    out = points.copy(); out[:, 0] /= scale
    return out

def shear_forward(points, strength=0.9):
    out = points.copy(); out[:, 1] += strength * np.tanh(points[:, 0] / 2)
    return out

def shear_inverse(points, strength=0.9):
    out = points.copy(); out[:, 1] -= strength * np.tanh(points[:, 0] / 2)
    return out

x1 = scale_forward(z)
x2 = shear_forward(x1)"""),
        ("md", "## 2. Geometry and density\n\nA local horizontal expansion by 2 divides density by 2. The shear changes shape but has determinant 1."),
        ("code", """fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
for ax, points, title in zip(axes, [z, x1, x2], ["z ~ N(0,I)", "scale x-axis by 2", "invertible shear"]):
    ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.25)
    ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
    ax.set_title(title); ax.set_xlabel("coordinate 1")
axes[0].set_ylabel("coordinate 2")
fig.suptitle("Forward maps bend samples; the Jacobian tracks local volume")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "03_flow_warp.png", dpi=180, bbox_inches="tight")
plt.show()

log_pz_at_zero = -np.log(2 * np.pi)
log_det_scale = np.log(2.0)
print(f"log p_z(0) = {log_pz_at_zero:.3f}")
print(f"log |det J_scale| = {log_det_scale:.3f}")
print(f"log p_x(f(0)) = {log_pz_at_zero - log_det_scale:.3f}")
print("density ratio after the expansion:", np.exp(-log_det_scale))"""),
        ("md", "## 3. Inverse check\n\nExact reversible sampling requires that the numerical inverse returns every point."),
        ("code", """recovered = scale_inverse(shear_inverse(x2))
error = np.max(np.abs(recovered - z))
print(f"maximum reconstruction error: {error:.3e}")
assert error < 1e-10"""),
        ("md", "## 4. Train a RealNVP rather than hand-designing the warp\n\nA RealNVP stacks affine coupling layers. Each layer leaves part of the vector unchanged and uses it to scale and shift the remaining coordinates. This triangular Jacobian makes the determinant cheap."),
        ("code", """import os
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display
from demo_models import RealNVP

torch.manual_seed(12)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
QUICK_MODE = os.getenv("LECTURE2_QUICK", "0") == "1"
artifact_suffix = "_smoke" if QUICK_MODE else ""
print("device:", device, "quick mode:", QUICK_MODE)

def make_two_moons(count=2000, noise=0.08, seed=12):
    generator = np.random.default_rng(seed)
    first = count // 2
    angle_a = generator.uniform(0, np.pi, first)
    angle_b = generator.uniform(0, np.pi, count - first)
    moon_a = np.c_[np.cos(angle_a), np.sin(angle_a)]
    moon_b = np.c_[1 - np.cos(angle_b), 0.45 - np.sin(angle_b)]
    points = np.vstack([moon_a, moon_b]) + generator.normal(0, noise, (count, 2))
    points = (points - points.mean(0)) / points.std(0)
    return torch.tensor(points, dtype=torch.float32)

target = make_two_moons(2400 if not QUICK_MODE else 600)
flow = RealNVP(dim=2, layers=6 if not QUICK_MODE else 4, hidden=64 if not QUICK_MODE else 32).to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=2e-3)
steps = 1500 if not QUICK_MODE else 180
fixed_z = torch.randn(800, 2, device=device)
frames, losses, frame_steps = [], [], []

for step in range(steps + 1):
    indices = torch.randint(0, len(target), (256,))
    batch = target[indices].to(device)
    loss = -flow.log_prob(batch).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    if step % max(steps // 12, 1) == 0 or step == steps:
        with torch.no_grad():
            generated, _ = flow(fixed_z)
        frames.append(generated.cpu().numpy())
        frame_steps.append(step)

print(f"initial NLL {losses[0]:.3f}; final NLL {losses[-1]:.3f}")"""),
        ("md", "## 5. Source, target, and learned samples\n\nThe base density stays Gaussian. Training changes the invertible map so its pushforward approaches the two-moon target."),
        ("code", """base = fixed_z.detach().cpu().numpy()
target_np = target.numpy()
final_np = frames[-1]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True, sharey=True)
for ax, points, title, color in zip(
    axes,
    [base, target_np, final_np],
    ["source z ~ N(0,I)", "target data", "trained flow samples"],
    ["tab:purple", "0.45", "tab:blue"],
):
    ax.scatter(points[:, 0], points[:, 1], s=7, alpha=0.35, c=color)
    ax.set_title(title); ax.set_xlabel("dimension 1")
axes[0].set_ylabel("dimension 2")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / f"03_realnvp_trained{artifact_suffix}.png", dpi=180, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(target_np[:, 0], target_np[:, 1], s=5, alpha=0.14, c="black", label="target")
moving = ax.scatter([], [], s=8, alpha=0.45, c="tab:blue", label="flow samples")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(loc="upper right")
def update(frame_index):
    moving.set_offsets(frames[frame_index])
    ax.set_title(f"RealNVP training step {frame_steps[frame_index]}")
    return (moving,)
animation = FuncAnimation(fig, update, frames=len(frames), interval=650, blit=False)
gif_path = ROOT / "fallbacks" / f"03_realnvp_training{artifact_suffix}.gif"
animation.save(gif_path, writer=PillowWriter(fps=2))
plt.close(fig)
display(HTML(animation.to_jshtml()))
print("saved", gif_path)"""),
        ("md", "## 6. Forward and inverse after training\n\nSampling uses z → x. Likelihood evaluation uses x → z and adds the inverse log-determinant."),
        ("code", """with torch.no_grad():
    test_z = torch.randn(128, 2, device=device)
    test_x, forward_logdet = flow(test_z)
    recovered_z, inverse_logdet = flow.inverse(test_x)
print("maximum inverse error:", float((test_z - recovered_z).abs().max().cpu()))
print("maximum log-det cancellation error:", float((forward_logdet + inverse_logdet).abs().max().cpu()))

# Record the actual output after every learned coupling layer, then apply the
# exact inverse layers in reverse order. This is not an interpolation: every
# frame is a real intermediate representation in the trained network.
with torch.no_grad():
    roundtrip = fixed_z[:500].clone()
    roundtrip_frames = [roundtrip.cpu().numpy()]
    roundtrip_labels = ["source z"]
    for layer_index, layer in enumerate(flow.layers, start=1):
        roundtrip, _ = layer(roundtrip)
        roundtrip_frames.append(roundtrip.cpu().numpy())
        roundtrip_labels.append(f"forward coupling {layer_index}")
    for reverse_index, layer in enumerate(reversed(flow.layers), start=1):
        roundtrip, _ = layer.inverse(roundtrip)
        roundtrip_frames.append(roundtrip.cpu().numpy())
        label = f"inverse coupling {reverse_index}"
        if reverse_index == len(flow.layers):
            label += " — recovered z"
        roundtrip_labels.append(label)

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(target_np[:, 0], target_np[:, 1], s=5, alpha=0.10, c="black", label="target data")
roundtrip_marks = ax.scatter([], [], s=9, alpha=0.50, c="tab:blue", label="tracked samples")
ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(loc="upper right")
def update_roundtrip(frame_index):
    roundtrip_marks.set_offsets(roundtrip_frames[frame_index])
    ax.set_title(roundtrip_labels[frame_index])
    return (roundtrip_marks,)
roundtrip_animation = FuncAnimation(fig, update_roundtrip, frames=len(roundtrip_frames), interval=900, blit=False)
roundtrip_path = ROOT / "fallbacks" / f"03_realnvp_roundtrip{artifact_suffix}.gif"
roundtrip_animation.save(roundtrip_path, writer=PillowWriter(fps=1))
plt.close(fig)
display(HTML(roundtrip_animation.to_jshtml()))
print("saved", roundtrip_path)"""),
        ("md", "## 7. Why a flow does not compress dimensionality\n\nEvery flow layer is a bijection between spaces of equal dimension. Its Jacobian must be square and nonsingular. A VAE may map x ∈ R⁷⁸⁴ to z ∈ R² because its encoder is not required to be invertible. A normalising flow over x ∈ R⁷⁸⁴ needs z ∈ R⁷⁸⁴."),
        ("code", """flow_8d = RealNVP(dim=8, layers=8, hidden=96).to(device)
z_8d = torch.randn(32, 8, device=device)
x_8d, _ = flow_8d(z_8d)
z_recovered, _ = flow_8d.inverse(x_8d)
print("source shape:", tuple(z_8d.shape))
print("target shape:", tuple(x_8d.shape))
print("square Jacobian for one example: 8 × 8")
print("inverse error:", float((z_8d - z_recovered).abs().max().detach().cpu()))
print("A 2D bottleneck would destroy invertibility; use a VAE when compression is the goal.")"""),
    ],
)


NOTEBOOKS["04_ebm_two_well.ipynb"] = notebook(
    "Two-well energy-based model",
    "Show that evaluating an energy is easy while obtaining representative samples depends on Langevin step size, noise, burn-in, and mixing.",
    [
        ("code", SETUP + "\n\nimport numpy as np\nimport matplotlib.pyplot as plt"),
        ("md", "## 1. Energy, gradient, and the two target modes"),
        ("code", """def energy(x):
    return (x**2 - 1.0)**2

def grad_energy(x):
    return 4.0 * x * (x**2 - 1.0)

grid = np.linspace(-2.2, 2.2, 700)
plt.figure(figsize=(8, 3.2))
plt.plot(grid, energy(grid), lw=2)
plt.scatter([-1, 1], [0, 0], s=70, label="low-energy modes")
plt.scatter([0], [1], s=70, label="barrier / zero gradient")
plt.ylim(-0.1, 4); plt.xlabel("x"); plt.ylabel("E(x)")
plt.legend(); plt.title("E(x) = (x² − 1)²")
plt.tight_layout(); plt.show()
print("∇E(0) =", grad_energy(np.array([0.0]))[0])"""),
        ("md", "## 2. One Langevin transition\n\nDrift moves downhill in energy. Noise prevents the procedure from becoming pure optimisation."),
        ("code", """def langevin(initial, steps=400, eta=0.02, noise_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(initial, dtype=float).copy()
    trajectory = [x.copy()]
    for _ in range(steps):
        drift = -0.5 * eta * grad_energy(x)
        noise = noise_scale * np.sqrt(eta) * rng.normal(size=x.shape)
        x = x + drift + noise
        trajectory.append(x.copy())
    return np.asarray(trajectory)

x0 = 1.50
eta = 0.10
gradient = 1.20
realised_noise = 0.05
after_drift = x0 - eta / 2 * gradient
after_noise = after_drift + realised_noise
print(f"current {x0:.2f} → drift {after_drift:.2f} → noise {after_noise:.2f}")"""),
        ("md", "## 3. Chains and coverage\n\nPredict whether each noise setting will cross the central barrier."),
        ("code", """settings = [(0.0, "no noise"), (0.45, "low noise"), (1.0, "Langevin noise")]
initial = np.linspace(-1.8, 1.8, 24)
fig, axes = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]})
for column, (scale, label) in enumerate(settings):
    trajectories = langevin(initial, steps=600, eta=0.02, noise_scale=scale, seed=20 + column)
    for chain in range(len(initial)):
        axes[0, column].plot(trajectories[:, chain], alpha=0.35, lw=0.8)
    axes[0, column].set_title(label)
    axes[0, column].set_xlabel("step"); axes[0, column].set_ylabel("state x")
    axes[0, column].axhline(-1, color="tab:green", ls="--", lw=1)
    axes[0, column].axhline(1, color="tab:green", ls="--", lw=1)
    samples = trajectories[200:].reshape(-1)
    axes[1, column].hist(samples, bins=45, density=True, color="tab:blue", alpha=0.75)
    axes[1, column].set_xlabel("visited state"); axes[1, column].set_ylabel("density")
fig.suptitle("Noise changes exploration and empirical mode coverage")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "04_ebm_langevin.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 4. What EBM training is trying to do\n\nThe model is pθ(x) = exp(−Eθ(x)) / Zθ. For the negative log-likelihood,\n\n∇θ L = E_data[∇θEθ(x)] − E_model[∇θEθ(x)].\n\nGradient descent therefore has two opposing effects:\n\n- **positive phase:** lower Eθ on observed data;\n- **negative phase:** raise Eθ on samples currently preferred by the model.\n\nThe exact negative phase is an expectation under pθ and therefore requires sampling. Contrastive divergence replaces it with short MCMC chains. In the code below, `mean(E(data)) − mean(E(negative particles))` is the direct approximation of these two terms."),
        ("code", """states = np.array([-1.0, 0.0, 1.0])
energies_before = np.array([0.8, 0.2, 0.9])
prob_before = np.exp(-energies_before); prob_before /= prob_before.sum()
energies_after = np.array([0.5, 0.8, 0.6])  # lower data energies at ±1, raise model-favoured x=0
prob_after = np.exp(-energies_after); prob_after /= prob_after.sum()
print("state             ", states)
print("probability before", prob_before.round(3), "← model wrongly prefers x=0")
print("probability after ", prob_after.round(3), "← mass moves toward the data modes")
print("The partition function changes automatically when any energy changes.")"""),
        ("md", "## 5. Train a one-dimensional neural EBM\n\nData come from two Gaussian modes. Persistent negative particles approximate samples from the current model. Each update lowers the data energy and raises the negative-particle energy."),
        ("code", """import os
import torch
from torch import nn
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML, display

torch.manual_seed(23)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
QUICK_MODE = os.getenv("LECTURE2_QUICK", "0") == "1"
artifact_suffix = "_smoke" if QUICK_MODE else ""

class EnergyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def draw_data(count):
    modes = torch.randint(0, 2, (count, 1), device=device) * 2 - 1
    return 1.5 * modes.float() + 0.18 * torch.randn(count, 1, device=device)

def update_negative(model, particles, steps=12, step_size=0.025, noise=1.0):
    x = particles.detach()
    for _ in range(steps):
        x.requires_grad_(True)
        gradient = torch.autograd.grad(model(x).sum(), x)[0]
        x = (x - 0.5 * step_size * gradient + noise * step_size**0.5 * torch.randn_like(x)).detach()
        x.clamp_(-3.0, 3.0)
    return x

energy_model = EnergyNet().to(device)
optimizer = torch.optim.Adam(energy_model.parameters(), lr=2e-3)
negative_particles = torch.empty(256, 1, device=device).uniform_(-3, 3)
training_steps = 800 if not QUICK_MODE else 140
grid_t = torch.linspace(-3, 3, 400, device=device)[:, None]
snapshots, snapshot_steps, losses = [], [], []

for step in range(training_steps + 1):
    data_batch = draw_data(256)
    negative_particles = update_negative(energy_model, negative_particles)
    positive_energy = energy_model(data_batch)
    negative_energy = energy_model(negative_particles)
    contrastive_loss = positive_energy.mean() - negative_energy.mean()
    regularizer = 1e-3 * (positive_energy.square().mean() + negative_energy.square().mean())
    loss = contrastive_loss + regularizer
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach().cpu()))
    if step % max(training_steps // 12, 1) == 0 or step == training_steps:
        with torch.no_grad():
            curve = energy_model(grid_t).cpu().numpy()
        snapshots.append((curve, negative_particles.detach().cpu().numpy().copy()))
        snapshot_steps.append(step)

print("device:", device)
print(f"contrastive objective: {losses[0]:.3f} → {losses[-1]:.3f}")"""),
        ("md", "## 6. Watch the energy landscape and negative particles co-evolve\n\nThe absolute energy level is arbitrary. What matters is the relative landscape: low near the two data modes, higher between and outside them."),
        ("code", """grid_np = grid_t.cpu().numpy().ravel()
data_for_plot = draw_data(1000).cpu().numpy().ravel()
fig, ax = plt.subplots(figsize=(7.5, 4.2))
curve_line, = ax.plot([], [], lw=2.5, color="tab:blue", label="learned energy")
particle_marks = ax.scatter([], [], marker="|", s=130, color="tab:orange", alpha=0.65, label="negative particles")
ax.hist(data_for_plot, bins=55, density=True, alpha=0.18, color="tab:green", label="data density")
ax.set_xlim(-3, 3); ax.set_ylim(-1.5, 5.0); ax.set_xlabel("x"); ax.set_ylabel("energy / scaled density")
ax.legend(loc="upper right")
def update_ebm(frame_index):
    curve, particles = snapshots[frame_index]
    curve = curve - curve.min()
    curve_line.set_data(grid_np, curve)
    particle_marks.set_offsets(np.c_[particles.ravel(), np.full(len(particles), 0.12)])
    ax.set_title(f"EBM training step {snapshot_steps[frame_index]}")
    return curve_line, particle_marks
ebm_animation = FuncAnimation(fig, update_ebm, frames=len(snapshots), interval=650, blit=False)
ebm_gif = ROOT / "fallbacks" / f"04_ebm_training{artifact_suffix}.gif"
ebm_animation.save(ebm_gif, writer=PillowWriter(fps=2))
plt.close(fig)
display(HTML(ebm_animation.to_jshtml()))
print("saved", ebm_gif)

final_curve, final_particles = snapshots[-1]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(grid_np, final_curve - final_curve.min(), lw=2.5, label="learned Eθ(x)")
ax.hist(data_for_plot, bins=55, density=True, alpha=0.25, label="data")
ax.scatter(final_particles.ravel(), np.full(len(final_particles), 0.08), marker="|", s=90, alpha=0.45, label="model particles")
ax.set_xlabel("x"); ax.set_ylabel("relative energy / density"); ax.legend()
ax.set_title("Training lowers energy near data and moves model particles toward both modes")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / f"04_ebm_trained{artifact_suffix}.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 7. Where the approximation can fail\n\nIf the negative chains do not mix, training raises energy only where those chains happen to visit. The model may keep unseen low-energy regions. This is why sampling quality and the learning objective cannot be assessed separately for an EBM."),
    ],
)


NOTEBOOKS["05_diffusion_snapshots.ipynb"] = notebook(
    "Diffusion forward and reverse snapshots",
    "Separate known forward corruption, noise-prediction training, and iterative reverse sampling using a tiny prepared MNIST model.",
    [
        ("code", SETUP + "\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nfrom demo_models import ConditionalTinyNoisePredictor, TinyNoisePredictor, diffusion_schedule\ntorch.manual_seed(11)"),
        ("md", "## 1. Load fixed data, schedule, and checkpoint\n\nThe checkpoint is intentionally small and pedagogical. It demonstrates the mechanics, not state-of-the-art image quality."),
        ("code", """checkpoint = torch.load(ROOT / "checkpoints" / "diffusion_mnist_tiny.pt", map_location="cpu", weights_only=True)
model = TinyNoisePredictor()
model.load_state_dict(checkpoint["state_dict"])
model.eval()
schedule = diffusion_schedule(checkpoint["steps"])
betas, alphas, alpha_bars = schedule["betas"], schedule["alphas"], schedule["alpha_bars"]
demo = np.load(ROOT / "data" / "mnist_demo_samples.npz")
x0 = torch.tensor(demo["images"][8:9], dtype=torch.float32) * 2.0 - 1.0
print("last training noise MSE:", round(checkpoint["history"]["mse"][-1], 4))"""),
        ("md", "## 2. The known forward process\n\nFor any selected t, q(xₜ|x₀) can be sampled directly. The same fixed ε makes the signal-to-noise progression easy to compare."),
        ("code", """timesteps = [0, 15, 35, 60, 99]
generator = torch.Generator().manual_seed(5)
epsilon = torch.randn(x0.shape, generator=generator)
forward = []
for t in timesteps:
    a_bar = alpha_bars[t]
    xt = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * epsilon
    forward.append(xt)

fig, axes = plt.subplots(1, len(timesteps), figsize=(10, 2.2))
for ax, t, xt in zip(axes, timesteps, forward):
    ax.imshow(xt[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t={t}\\nᾱ={alpha_bars[t]:.2f}")
    ax.axis("off")
fig.suptitle("Forward corruption is known; no neural network is required")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "05_diffusion_forward.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 3. The training target at one random timestep\n\nThe network receives xₜ and t, then predicts the injected ε."),
        ("code", """t = torch.tensor([60])
a_bar = alpha_bars[t][:, None, None, None]
xt = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * epsilon
with torch.no_grad():
    epsilon_hat = model(xt, t)
print("epsilon-prediction MSE for this example:", torch.mean((epsilon_hat - epsilon) ** 2).item())"""),
        ("md", "## 4. Iterative reverse sampling\n\nStart from Gaussian noise. Each step predicts noise, removes part of it, and adds the variance required by the reverse transition. One trajectory is illustrative evidence, not a quality evaluation."),
        ("code", """@torch.no_grad()
def reverse_sample(seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 1, 28, 28), generator=generator)
    capture = {99, 75, 50, 25, 0}
    snapshots = {}
    for t_index in reversed(range(len(betas))):
        t = torch.tensor([t_index])
        eps_hat = model(x, t)
        alpha = alphas[t_index]
        alpha_bar = alpha_bars[t_index]
        beta = betas[t_index]
        mean = (x - (beta / torch.sqrt(1 - alpha_bar)) * eps_hat) / torch.sqrt(alpha)
        if t_index > 0:
            noise = torch.randn(x.shape, generator=generator)
            previous_alpha_bar = alpha_bars[t_index - 1]
            posterior_variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
            x = mean + torch.sqrt(posterior_variance) * noise
        else:
            x = mean
        if t_index in capture:
            snapshots[t_index] = x.clamp(-1, 1).clone()
    return snapshots

reverse = reverse_sample()
shown = [99, 75, 50, 25, 0]
fig, axes = plt.subplots(1, len(shown), figsize=(10, 2.2))
for ax, t, image in zip(axes, shown, [reverse[t] for t in shown]):
    ax.imshow(image[0, 0].numpy(), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"after t={t}")
    ax.axis("off")
fig.suptitle("The learned reverse transition gradually exposes structure")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "05_diffusion_reverse.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 5. Classifier-free guidance\n\nA second model was trained with class labels randomly removed. At sampling time it predicts noise twice:\n\nε_cfg = ε_uncond + w(ε_cond − ε_uncond).\n\nThe difference points toward features associated with the requested digit. Larger guidance usually improves class compliance but can reduce diversity and amplify artifacts."),
        ("code", """cfg_checkpoint = torch.load(ROOT / "checkpoints" / "diffusion_mnist_cfg.pt", map_location="cpu", weights_only=True)
cfg_model = ConditionalTinyNoisePredictor(num_classes=cfg_checkpoint["num_classes"])
cfg_model.load_state_dict(cfg_checkpoint["state_dict"])
cfg_model.eval()
print("CFG training MSE:", round(cfg_checkpoint["history"]["mse"][-1], 4))
print("null label:", cfg_model.null_label, "condition dropout:", cfg_checkpoint["condition_dropout"])

@torch.no_grad()
def guided_reverse(initial, start_t, digit, guidance_scale=3.0, seed=31, capture=False):
    generator = torch.Generator().manual_seed(seed)
    x = initial.clone()
    snapshots = {}
    label = torch.tensor([digit])
    null = torch.tensor([cfg_model.null_label])
    for t_index in reversed(range(start_t + 1)):
        t = torch.tensor([t_index])
        eps_unconditional = cfg_model(x, t, null)
        eps_conditional = cfg_model(x, t, label)
        eps_guided = eps_unconditional + guidance_scale * (eps_conditional - eps_unconditional)
        alpha, alpha_bar, beta = alphas[t_index], alpha_bars[t_index], betas[t_index]
        mean = (x - beta / torch.sqrt(1 - alpha_bar) * eps_guided) / torch.sqrt(alpha)
        if t_index == 0:
            x = mean
        else:
            previous_alpha_bar = alpha_bars[t_index - 1]
            posterior_variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
            x = mean + torch.sqrt(posterior_variance) * torch.randn(x.shape, generator=generator)
        if capture and t_index in {start_t, 75, 50, 25, 0}:
            snapshots[t_index] = x.clamp(-1, 1).clone()
    return x.clamp(-1, 1), snapshots"""),
        ("md", "## 6. Compare guidance strengths from the same initial noise\n\nEvery column starts from exactly the same initial noise and reuses the same random noise at every reverse step. Changes can therefore be attributed to guidance rather than the random path. We use the DDPM posterior variance, which is smaller than simply adding √βₜ noise."),
        ("code", """target_digit = 3
guidance_scales = [0.0, 1.0, 2.0, 3.0]
initial_generator = torch.Generator().manual_seed(27)
initial_noise = torch.randn((1, 1, 28, 28), generator=initial_generator)
guided_results = [guided_reverse(initial_noise, 99, target_digit, scale, seed=91)[0] for scale in guidance_scales]

fig, axes = plt.subplots(1, len(guidance_scales), figsize=(8.5, 2.3))
for ax, scale, image in zip(axes, guidance_scales, guided_results):
    ax.imshow(image[0, 0], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"w={scale:g}")
    ax.axis("off")
fig.suptitle(f"Same noise, requested digit {target_digit}: guidance changes class evidence")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "05_diffusion_cfg_scales.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 7. Guided refinement from an intermediate noise level\n\nStarting from pure noise tests unconditional generation. Starting from a partly corrupted real image tests guided denoising or image-to-image refinement. The latter is easier because some source structure remains; it should not be reported as pure generation."),
        ("code", """source_digit = 3
source = torch.tensor(demo["images"][source_digit:source_digit+1], dtype=torch.float32) * 2 - 1
start_t = 60
refine_generator = torch.Generator().manual_seed(8)
refine_noise = torch.randn(source.shape, generator=refine_generator)
noised = alpha_bars[start_t].sqrt() * source + (1 - alpha_bars[start_t]).sqrt() * refine_noise
refined, refine_trace = guided_reverse(noised, start_t, source_digit, guidance_scale=3.0, seed=52, capture=True)

fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
for ax, image, title in zip(axes, [source, noised, refined], ["source", f"noised to t={start_t}", "guided reverse"]):
    ax.imshow(image[0, 0], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(title); ax.axis("off")
fig.suptitle("Guided refinement preserves information that pure generation never received")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / "05_diffusion_guided_refinement.png", dpi=180, bbox_inches="tight")
plt.show()"""),
    ],
)


NOTEBOOKS["06_pixelcnn_real_images.ipynb"] = notebook(
    "Train a PixelCNN on Fashion-MNIST",
    "Move from the four-pixel blackboard example to a real-image autoregressive model with exact likelihood, masked convolutions, GPU or Apple-Metal acceleration, saved weights, unconditional sampling, and conditional completion.",
    [
        ("code", SETUP + "\n\nfrom dataclasses import asdict\nimport json\nimport os\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport torch\nfrom torch import nn\nfrom pixelcnn_training import load_pixelcnn_for_demo, resolve_profile, sample_pixelcnn, train_pixelcnn\ntorch.manual_seed(29)"),
        ("md", "## 1. Choose an explicit training profile\n\nThe profile is part of the experiment, not a hidden machine-specific detail. `smoke` verifies the pipeline, `local` is a laptop/lab baseline, and `h100` is the prepared full-quality run for one 80 GB NVIDIA H100. The H100 profile uses the complete 28×28 Fashion-MNIST training split, a 128-channel/15-layer model, batch size 1024, 50 epochs, BF16, channels-last tensors, `torch.compile`, validation, resumable latest checkpoints, and a separate best checkpoint.\n\nFor a long H100 job, the terminal command is safer than leaving a browser kernel open:\n\n```bash\npython tools/train_pixelcnn.py --profile h100 --resume auto\n```\n\nAfterwards, set `LECTURE2_PIXELCNN_MODE=load` to use the H100 checkpoint on the presentation Mac without retraining. Optional `LECTURE2_PIXELCNN_SAMPLE_COUNT` and `LECTURE2_PIXELCNN_COMPLETION_COUNT` variables keep live sampling short. Training never publishes anything. Inspect the saved checkpoint and samples first."),
        ("code", """QUICK_MODE = os.getenv("LECTURE2_QUICK", "0") == "1"
PROFILE_NAME = os.getenv("LECTURE2_PIXELCNN_PROFILE", "smoke" if QUICK_MODE else "local")
profile = resolve_profile(PROFILE_NAME)
resume = os.getenv("LECTURE2_PIXELCNN_RESUME", "auto" if PROFILE_NAME == "h100" else "none")
MODE = os.getenv("LECTURE2_PIXELCNN_MODE", "train")
checkpoint_override = os.getenv("LECTURE2_PIXELCNN_CHECKPOINT")
artifact_suffix = "_smoke" if PROFILE_NAME == "smoke" else "_local" if PROFILE_NAME == "local" else ""
print(json.dumps(asdict(profile), indent=2))
print("mode:", MODE, "resume policy:", resume)"""),
        ("md", "## 2. Train, validate, and checkpoint\n\nFashion-MNIST contains 28×28 grayscale images of clothing. The shared trainer performs a seeded train/validation split, evaluates bits per dimension after every epoch, writes an atomic `latest` checkpoint for resuming, and updates the canonical checkpoint only when validation likelihood improves. After the final epoch it evaluates the untouched 10,000-image test split. The first completed epoch gives a machine-specific runtime estimate; this is more reliable than assuming that every H100 software stack has identical throughput."),
        ("code", """if MODE == "load":
    checkpoint_path = Path(checkpoint_override) if checkpoint_override else ROOT / "checkpoints" / "pixelcnn_fashionmnist.pt"
    result = load_pixelcnn_for_demo(ROOT, checkpoint_path)
elif MODE == "train":
    result = train_pixelcnn(ROOT, profile, resume=resume, progress=True)
else:
    raise ValueError("LECTURE2_PIXELCNN_MODE must be 'train' or 'load'")
model = result.model
device = result.device
data = result.data
profile = result.profile
LEVELS = profile.levels
inputs = data.test_inputs
targets = data.test_targets
labels_np = data.test_labels
CHECKPOINT_PATH = result.best_checkpoint
print({
    "best_checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    "best_validation_bpd": round(result.best_validation_bpd, 4),
    "test_bpd": round(result.test_bpd, 4),
    "elapsed_minutes": round(result.elapsed_seconds / 60, 2),
})

fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
for ax, image, label in zip(axes.flat, inputs[:16], labels_np[:16]):
    ax.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
    ax.set_title(str(label)); ax.axis("off")
fig.suptitle("Fashion-MNIST examples; labels are shown only for reference, not used by PixelCNN")
fig.tight_layout(); plt.show()"""),
        ("md", "## 3. What the masked-convolution objective computed\n\nAt every spatial position, the model predicts one of 16 intensity values. Mask A blocks the current pixel and all future pixels. Later Mask B layers may see the current hidden state but still cannot see future pixels. Training evaluates all observed targets in parallel. The validation and test numbers above are normalised negative log-likelihoods in bits per dimension; lower is better."),
        ("md", "## 4. Exact likelihood on a complete image\n\nExact means the normalised categorical probability for every observed pixel can be evaluated and summed. It does not mean that generation is deterministic or that the most likely image will look most plausible."),
        ("code", """model.eval()
with torch.no_grad():
    example_x = inputs[:1].to(device)
    example_target = targets[:1].to(device)
    logits = model(example_x)
    nll_nats = nn.functional.cross_entropy(logits, example_target, reduction="sum")
    bits_per_dimension = nll_nats / example_target.numel() / np.log(2)
print(f"image NLL: {float(nll_nats.cpu()):.2f} nats")
print(f"normalised: {float(bits_per_dimension.cpu()):.3f} bits/dimension")"""),
        ("md", "## 5. Sequential sampling\n\nGeneration is intentionally slow: each sampled pixel becomes context for the next model call. Training used one forward pass per batch; a 28×28 sample requires 784 sequential model evaluations. The number of parallel sample chains changes throughput, not the dependency depth."),
        ("code", """sample_count = int(os.getenv("LECTURE2_PIXELCNN_SAMPLE_COUNT", profile.sample_count))
samples = sample_pixelcnn(
    model,
    device,
    LEVELS,
    tuple(targets.shape[1:]),
    count=sample_count,
    temperature=0.9,
    seed=30,
)
fig, axes = plt.subplots(1, sample_count, figsize=(2.2 * sample_count, 2.3), squeeze=False)
for ax, sample in zip(axes.flat, samples):
    ax.imshow(sample[0], cmap="gray", vmin=-1, vmax=1); ax.axis("off")
fig.suptitle("SMOKE PROFILE: pipeline test, not model-quality evidence" if PROFILE_NAME == "smoke" else "Unconditional sequential samples")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / f"06_pixelcnn_samples{artifact_suffix}.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 6. Conditional completion\n\nThe top half is observed. PixelCNN samples the lower half from p(x_missing | x_observed). Different seeds can yield different valid completions for the same prefix."),
        ("code", """completion_count = int(os.getenv("LECTURE2_PIXELCNN_COMPLETION_COUNT", profile.completion_count))
prefix = inputs[:1].repeat(completion_count, 1, 1, 1)
known_rows = prefix.shape[-2] // 2
completions = sample_pixelcnn(
    model,
    device,
    LEVELS,
    tuple(targets.shape[1:]),
    count=completion_count,
    temperature=0.85,
    seed=45,
    prefix=prefix,
    known_rows=known_rows,
)
fig, axes = plt.subplots(1, completion_count + 1, figsize=(2.2 * (completion_count + 1), 2.4))
axes[0].imshow(prefix[0, 0], cmap="gray", vmin=-1, vmax=1); axes[0].set_title("original")
for index, image in enumerate(completions, start=1):
    axes[index].imshow(image[0], cmap="gray", vmin=-1, vmax=1); axes[index].set_title(f"completion {index}")
for ax in axes: ax.axis("off")
fig.suptitle("SMOKE PROFILE: completion mechanics only" if PROFILE_NAME == "smoke" else "One observed prefix, multiple sampled futures")
fig.tight_layout()
fig.savefig(ROOT / "fallbacks" / f"06_pixelcnn_completions{artifact_suffix}.png", dpi=180, bbox_inches="tight")
plt.show()"""),
        ("md", "## 7. Student experiments\n\n1. Compare exact image likelihood with visual plausibility.\n2. Change sampling temperature while keeping the seed fixed.\n3. Change the known prefix height.\n4. Measure sampling time as image resolution grows.\n5. Explain why training is parallel but sampling is sequential."),
        ("md", "## 8. Inspect first; publish separately\n\nOnly the H100 profile writes the canonical `checkpoints/pixelcnn_fashionmnist.pt`. The `local` and `smoke` profiles use different filenames and cannot overwrite it. Inspect validation/test likelihood, samples, completions, and the recorded profile before publishing. Publication is a separate explicit command and is never triggered by this notebook:\n\n```bash\npython tools/publish_huggingface.py joergsimon/fh-joanneum-pixelcnn-fashionmnist\n```"),
    ],
)


REFLECTIONS = {
    "01_autoregressive_4_pixel.ipynb": (
        "Explain why exact likelihood evaluation does not make sampling deterministic. "
        "Then change one conditional probability and identify exactly which observed-image factors and sampled prefixes can change.\n\n"
        "**Optional extension:** replace the hand-written table with a tiny masked neural network and compare its learned conditionals with the table."
    ),
    "02_vae_latent_walk.ipynb": (
        "Identify where randomness enters, and explain why decoding the mean is useful for comparison but is not a posterior sample. "
        "Find an interpolation that crosses a sparse latent region and record the visual failure.\n\n"
        "**Optional extension:** compare linear interpolation with spherical interpolation and measure reconstruction error along both paths."
    ),
    "03_flow_2d.ipynb": (
        "Relate every visible expansion or contraction to the sign of the log-determinant. "
        "Explain why a small inverse error is necessary but does not prove that the learned density matches the data.\n\n"
        "**Optional extension:** add coupling layers, then compare held-out negative log-likelihood, inverse error, and training time."
    ),
    "04_ebm_two_well.ipynb": (
        "Separate the energy model from the sampler. Explain how poor mixing can corrupt both generated samples and the negative phase used for learning.\n\n"
        "**Optional extension:** start all persistent particles in one well and test which step-size and noise combinations recover both modes."
    ),
    "05_diffusion_snapshots.ipynb": (
        "Distinguish forward corruption, the supervised noise target, unconditional reverse sampling, classifier-free guidance, and guided refinement. "
        "State which outputs count as pure generation.\n\n"
        "**Optional extension:** hold the initial noise fixed and plot class compliance and diversity across guidance strengths."
    ),
    "06_pixelcnn_real_images.ipynb": (
        "Compare bits per dimension with visual plausibility and sampling time. Explain why a model can assign exact likelihoods while still producing weak samples.\n\n"
        "**Optional extension:** compare the `local` and supplied full checkpoint on held-out likelihood, completion diversity, and wall-clock sampling cost."
    ),
}

for filename, reflection in REFLECTIONS.items():
    NOTEBOOKS[filename].cells.append(
        nbf.v4.new_markdown_cell(
            "## Interpretation check and optional extension\n\n" + reflection +
            "\n\nRecord the changed parameter, the fixed controls, the observed evidence, and one limitation."
        )
    )


def main() -> None:
    for filename, nb in NOTEBOOKS.items():
        nbf.write(nb, ROOT / filename)
        print("wrote", filename)


if __name__ == "__main__":
    main()
