# Advanced Generative Methods and Large Language Models

Student-facing code, notebooks, practical exercises, and supporting material for the 2026 FH JOANNEUM course **Advanced Generative Methods and Large Language Models**.

The course connects probabilistic generative modelling with modern language models and agentic engineering. The recurring case study is **CampusCritters** and its LLM-based assistant **Scout**. Most practical work runs locally or from supplied checkpoints and replay data. Moodle remains the authoritative source for dates, assessment instructions, submissions, and reviewed slide handouts.

## Course map

| Lecture | Date | Main focus | Materials |
| --- | --- | --- | --- |
| [1](lectures/lecture_01/) | 30 Oct 2026 | Motivation, agents, traces, reliability, and course contract | Course orientation |
| [2](lectures/lecture_02/) | 6 Nov 2026 | Autoregressive models, VAEs, flows, GANs, energy-based models, and diffusion | [Six executable notebooks](lectures/lecture_02/demos/) |
| [3](lectures/lecture_03/) | 13 Nov 2026 | Tokenisation, transformers, attention, KV caches, decoding, and efficiency | [Transformer laboratory](labs/lecture3_transformer_lab/) |
| [4](lectures/lecture_04/) | 20 Nov 2026 | Text mining, LSA/LDA, embeddings, prompting, retrieval, RAG, and typed tools | [Retrieval practical](lectures/lecture_04/practical/) |
| [5](lectures/lecture_05/) | 27 Nov 2026 | SFT, LoRA/QLoRA/DoRA, preference optimisation, tool-use training, and adaptation | [Adapter evaluation laboratory](labs/lecture5_adapter_evaluation/) |
| [6](lectures/lecture_06/) | 4 Dec 2026 | Agent harnesses, state, memory, permissions, traces, and security | [Harness repair laboratory](labs/lecture6_harness_repair/) |
| [7](lectures/lecture_07/) | 11 Dec 2026 | Held-out evaluation, repeated-run reliability, utility/security, cost, and system evidence | [System evaluation laboratory](labs/lecture7_system_evaluation/) |

This is the first course offering. Topic emphasis, tools, practical details, and deadlines may change with advance notice.

## Lecture 2 notebooks

Lecture 2 is the first live-coding session. Its notebooks use one comparison frame for every model family: **How is the distribution represented? What is optimised? How do we sample? What can fail?**

1. [Four-pixel autoregression](lectures/lecture_02/demos/01_autoregressive_4_pixel.ipynb): exact likelihood and sequential sampling.
2. [VAE latent walk](lectures/lecture_02/demos/02_vae_latent_walk.ipynb): encoding, reparameterisation, reconstruction, and interpolation.
3. [Two-dimensional normalising flow](lectures/lecture_02/demos/03_flow_2d.ipynb): training, forward and inverse maps, animation, and the fixed-dimensionality constraint.
4. [Two-well energy-based model](lectures/lecture_02/demos/04_ebm_two_well.ipynb): positive and negative phases, Langevin drift, noise, and mixing.
5. [Diffusion snapshots](lectures/lecture_02/demos/05_diffusion_snapshots.ipynb): forward corruption, reverse sampling, classifier-free guidance, and refinement.
6. [PixelCNN on Fashion-MNIST](lectures/lecture_02/demos/06_pixelcnn_real_images.ipynb): masked convolutions, conditional completion, sampling, and the prepared H100 training profile.

See the [Lecture 2 demo guide](lectures/lecture_02/demos/README.md) for setup, checkpoints, fallbacks, and the classroom run order.

## Practical work

The repository contains four small **Scout Missions** under [`assignments/`](assignments/) and guided in-class laboratories under [`labs/`](labs/). The core paths are deliberately modest in compute and can be extended into more ambitious experiments.

Git stores versioned code and evidence. Moodle records the official submission, deadline, student identity, repository URL, exact commit hash, short evidence summary, and AI-use declaration. Do not submit coursework merely by opening a pull request here.

## Quick start

The recommended route uses `uv`; it can install Python 3.11 as well as the
course environment. Install it first if necessary:

```bash
# macOS with Homebrew
brew install uv

# Windows with WinGet
winget install --id=astral-sh.uv -e
```

Linux users and anyone who prefers another installation method should use the
[official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
Then clone the repository:

```bash
git clone https://github.com/joergsimon/lecture_fh_joanneum_advance_generative_models_and_llms_2026.git
cd lecture_fh_joanneum_advance_generative_models_and_llms_2026
```

For Lecture 2, create the supplied Python environment and start Jupyter:

```bash
cd lectures/lecture_02/demos
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python tools/verify_demos.py
python -m jupyter lab
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Conda,
Miniforge, standard `venv`, CUDA, and Apple Silicon instructions are in
[SETUP.md](SETUP.md). Common problems and their fixes are collected in
[TROUBLESHOOTING.md](TROUBLESHOOTING.md). Code selects CUDA first, then Apple
MPS, then CPU. No paid API is required.

## Responsible use

- Never commit API keys, passwords, private data, proprietary documents, or unreviewed model outputs containing sensitive information.
- Record the model, runtime, decoding settings, seed where applicable, and relevant environment details.
- Document approved AI assistance using [`common/ai_use_declaration.md`](common/ai_use_declaration.md).
- Treat a successful demonstration as an example, not as evidence of general reliability. Use held-out cases and repeated runs where the claim requires them.

## Repository status

Materials are released incrementally. Clean notebooks and small teaching checkpoints are versioned here. Large datasets, large model weights, slide production files, videos, private review notes, and Moodle exports are intentionally excluded.
