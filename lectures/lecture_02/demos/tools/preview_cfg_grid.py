"""Render a diagnostic grid for the Lecture 2 class-conditional diffusion model."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo_models import ConditionalTinyNoisePredictor, diffusion_schedule  # noqa: E402


def device_for_run() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    device = device_for_run()
    checkpoint = torch.load(ROOT / "checkpoints" / "diffusion_mnist_cfg.pt", map_location="cpu", weights_only=True)
    model = ConditionalTinyNoisePredictor(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    schedule = diffusion_schedule(checkpoint["steps"])
    betas = schedule["betas"].to(device)
    alphas = schedule["alphas"].to(device)
    alpha_bars = schedule["alpha_bars"].to(device)

    labels = torch.arange(10, device=device)
    null = torch.full_like(labels, model.null_label)
    generator = torch.Generator().manual_seed(27)
    shared_noise = torch.randn((1, 1, 28, 28), generator=generator).repeat(10, 1, 1, 1).to(device)

    for guidance_scale in [1.0, 2.0, 3.0]:
        x = shared_noise.clone()
        step_generator = torch.Generator().manual_seed(91)
        for t_index in reversed(range(len(alpha_bars))):
            t = torch.full((len(x),), t_index, device=device, dtype=torch.long)
            eps_u = model(x, t, null)
            eps_c = model(x, t, labels)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            alpha = alphas[t_index]
            alpha_bar = alpha_bars[t_index]
            beta = betas[t_index]
            mean = (x - beta / torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha)
            if t_index == 0:
                x = mean
            else:
                previous = alpha_bars[t_index - 1]
                posterior_variance = beta * (1 - previous) / (1 - alpha_bar)
                shared_step_noise = torch.randn((1, 1, 28, 28), generator=step_generator).repeat(10, 1, 1, 1).to(device)
                x = mean + torch.sqrt(posterior_variance) * shared_step_noise

        fig, axes = plt.subplots(2, 5, figsize=(8.2, 3.6))
        for digit, ax in enumerate(axes.ravel()):
            ax.imshow(x[digit, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"requested {digit}")
            ax.axis("off")
        fig.suptitle(f"Same initial noise · classifier-free guidance w={guidance_scale:g}")
        fig.tight_layout()
        path = ROOT / "fallbacks" / f"05_cfg_digit_grid_w{guidance_scale:g}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
