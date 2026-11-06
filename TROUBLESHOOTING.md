# Troubleshooting

Start by confirming that the terminal is inside the repository and that the
course environment is active:

```bash
pwd
python --version
python -c "import sys; print(sys.executable)"
```

The Python version should be 3.11 and the executable should point into the
environment created for this repository.

## `uv`, `git`, or `python` is not found

Close and reopen the terminal after installing a command-line tool. On Windows,
also try a new PowerShell window. Re-run `git --version` and `uv --version`.
Installation methods are documented in [SETUP.md](SETUP.md).

## PowerShell blocks environment activation

Activation is convenient but not required. Use the environment's interpreter
directly:

```powershell
.venv\Scripts\python.exe -m jupyter lab
```

## Jupyter uses the wrong Python

Stop Jupyter and start it from the intended environment:

```bash
python -c "import sys; print(sys.executable)"
python -m jupyter lab
```

In the notebook, run:

```python
import sys
print(sys.executable)
```

Both paths should refer to the same environment.

## `ModuleNotFoundError`

Install dependencies through the environment's Python, not a global `pip`:

```bash
cd lectures/lecture_02/demos
python -m pip install -r requirements.txt
```

With `uv`, while `.venv` is active:

```bash
cd lectures/lecture_02/demos
uv pip install -r requirements.txt
```

## The exact lock file does not install

The lock captures the validated environment but can be too strict for a
different operating system or newer Python package index. Use the portable
specification instead:

```bash
uv pip install -r requirements.txt
```

## `CUDA False` on an NVIDIA computer

Confirm the NVIDIA driver with `nvidia-smi`. Then use the official
[PyTorch installation selector](https://docs.pytorch.org/get-started/locally/)
to install the build matching the operating system and supported CUDA runtime.
Do not infer the correct package from the CUDA number alone.

## `MPS False` on a Mac

MPS requires a supported Apple Silicon/macOS/PyTorch combination. The notebooks
also run on CPU. Restart Jupyter after changing the environment and verify that
the kernel uses the intended Python. A separate MLX installation does not make
PyTorch report `MPS True`.

## Fashion-MNIST cannot download

Only the PixelCNN notebook requires this download. Check the network or proxy,
or copy an existing torchvision Fashion-MNIST directory to:

```text
lectures/lecture_02/demos/data/FashionMNIST/
```

The other five Lecture 2 notebooks can still run with the supplied data and
checkpoints.

## A notebook is slow or memory runs out

Restart the kernel, close other notebooks, and select the documented quick or
smoke profile. For the PixelCNN demo, launch Jupyter with `LECTURE2_QUICK=1`.
Full H100 training is a lecturer workflow and is not a student requirement.

## Animations do not appear in Jupyter

Run all preceding cells, wait for training or loading to complete, and check
the corresponding static image or GIF under
`lectures/lecture_02/demos/fallbacks/`. The fallback represents the same
teaching step and is safe to use when browser rendering fails.

## The demo verifier fails

From `lectures/lecture_02/demos`, run:

```bash
python tools/verify_demos.py
```

Read the first failing notebook and error, rather than later cascading errors.
If the error concerns an import, verify the environment. If it concerns data,
check the data section above. Do not commit the generated `executed/` directory.

## Git reports large or private files

Do not force-add virtual environments, downloaded datasets, API keys, `.env`
files, large model weights, Moodle exports, or executed notebooks. Run:

```bash
python scripts/check_repository.py
git status --short
```

If a secret was committed, removing the file in a later commit is not enough;
revoke the credential immediately and contact the lecturer before rewriting
history.
