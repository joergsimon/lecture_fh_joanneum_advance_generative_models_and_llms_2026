---
library_name: pytorch
license: mit
tags:
  - pixelcnn
  - autoregressive
  - fashion-mnist
  - education
---

# FH JOANNEUM teaching PixelCNN

Planned public repository: <https://huggingface.co/joergsimon/fh-joanneum-pixelcnn-fashionmnist>

This small PixelCNN supports the course **Advanced Generative Methods and Large Language Models** at FH JOANNEUM. It provides a concrete bridge from a four-variable chain-rule calculation to autoregressive image modelling.

## Model

- Dataset: Fashion-MNIST training split
- Representation: grayscale quantised to 16 uniform intensity levels
- Architecture: raster-order masked convolutions, Mask A followed by Mask B layers
- Objective: categorical negative log-likelihood at every pixel
- Sampling: sequential raster-order categorical sampling
- Reference training profile: one H100, 128 hidden channels, 15 layers, batch size 1024, 50 epochs, BF16, cosine-decayed AdamW

The checkpoint contains the `state_dict`, explicit training profile, architecture configuration, training and validation history in bits per dimension, test bits per dimension in the companion summary, dataset description, quantisation description, device metadata, and seed. The architecture is defined in `demo_models.py`; the training implementation is in `pixelcnn_training.py`.

## Intended use

- teaching exact autoregressive likelihood;
- visualising the difference between parallel training and sequential sampling;
- unconditional sampling and conditional lower-half completion;
- small student experiments with temperature and context length.

## Limitations

This is a compact teaching model, not a competitive image generator. Fashion-MNIST is small and grayscale. Samples can be ambiguous, and a visually plausible sample need not receive the best likelihood. Sampling is slow because every pixel requires a model evaluation. The checkpoint must not be used for safety-critical or production decisions.

## Reproducibility

Run `bash tools/run_pixelcnn_h100.sh` with the environment documented in `README.md`, then inspect the summary and generated figures. The `local` and `smoke` profiles use distinct filenames and cannot overwrite the canonical H100 checkpoint. Do not publish either reduced-profile checkpoint.
