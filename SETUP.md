# Local setup

## Baseline

- Python 3.11
- Git
- `uv` (recommended), Conda/Miniforge, or the standard Python virtual environment
- Optional: Ollama or LM Studio for local language-model inference

No paid API is required. Python is the supported training and fine-tuning baseline. Python or TypeScript may be used for inference and harness work when the required interfaces, traces, and evaluations remain equivalent.

## Lecture 2 environment

```bash
cd lectures/lecture_02/demos
uv venv --python 3.11
source .venv/bin/activate          # macOS or Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m jupyter lab
```

For an exact reproduction of the validated environment, use `requirements-lock.txt`. For a more portable installation on a future platform, use `requirements.txt`.

Verify the environment:

```bash
python -c "import torch, torchvision, numpy, matplotlib; print('PyTorch', torch.__version__); print('CUDA', torch.cuda.is_available()); print('MPS', torch.backends.mps.is_available())"
python tools/verify_demos.py
```

## Lightweight replay paths

These exercises use the Python standard library:

```bash
python assignments/mission1_sampling/run_sampling.py --replay
python labs/lecture3_transformer_lab/transformer_lab.py
python lectures/lecture_04/practical/scout_retrieval_lab.py
python assignments/mission3_change_one_thing/evaluate.py
python assignments/mission4_scout_on_a_leash/harness.py --variant controlled
python assignments/mission4_scout_on_a_leash/evaluate_replays.py
```

## Local language models

If an exercise uses Ollama or LM Studio, record the exact model identifier, quantisation, runtime version, and decoding settings. Use a small quantised model that fits the available computer. Supplied replay outputs and checkpoints provide the low-compute fallback.

## Repository hygiene

Do not commit downloaded datasets, large model weights, `.env` files, credentials, private data, or notebook autosaves. Run the repository check before committing:

```bash
python scripts/check_repository.py
```
