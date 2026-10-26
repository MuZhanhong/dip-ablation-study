# Deep Image Prior Architecture Ablation Study

Study how architectural choices in Deep Image Prior (DIP) affect recovery of missing regions in synthetic grayscale images.

## Status
Scaffold now includes the configurable DIP hourglass ablation models, masked training loop, held-out-MSE checkpoint selection, and a CLI experiment runner. The original DIP vase sanity check, frequency-domain analysis, and full experimental run remain to be integrated and validated. Tests have been added but should be run locally before interpreting results.

## Experiment setup
- Image size: 64 × 64, grayscale.
- Image families: horizontal gradient; periodic Gaussian-blurred noise at scales ℓ ∈ {0, 1, 2, 4, 8}; diagonal sinusoidal stripes (30° orientation, period 8).
- Missing region: centered 16 × 16 square.
- Held-out validation: 5% of visible pixels, sampled with seed 0 and never inside the missing square.
- Primary model plan: 4-level DIP-style hourglass, bilinear upsampling, average pooling, 32-channel fixed random-noise input scaled by 0.1, no skip connections, no input jitter, Adam at learning rate 0.01, up to 3000 steps, logging every 50 steps.
- Ablation factors: upsampling on/off; 3 × 3 vs 1 × 1 filters; weight sharing on/off. Keep channel width consistent; consider 16 channels if memory is limiting.
- Model selection: held-out MSE. Report gap MSE, PSNR, SSIM, frequency-domain summaries, optimization trajectories, and image snapshots. Gap-oracle metrics are secondary and must not select the primary checkpoint.
- Baselines: grey fill, harmonic fill, fully connected network.
- Sanity check: reproduce the original DIP inpainting/vase notebook separately. Coordinate inputs are for this sanity check only, not the main experiment.

## Implementation notes

- All variants use average pooling in the encoder and have no encoder-to-decoder skip connections.
- In this implementation, `upsampling: on` means bilinear interpolation in the decoder; `off` replaces it with learned 2× transposed convolutions. This is the operational definition used by the runner and should be confirmed against the intended experimental definition before reporting results.
- Weight sharing reuses one encoder block across encoder scales and one separate decoder block across decoder scales; encoder and decoder do not share the same block.
- The model input is fixed random noise with the number of channels set by the model configuration. The output uses a sigmoid to constrain predictions to [0, 1].

## Ablation matrix
| ID | Upsampling | Filter size | Weight sharing |
|---|---|---|---|
| 1 | on | 3 × 3 | on |
| 2 | on | 3 × 3 | off |
| 3 | on | 1 × 1 | on |
| 4 | on | 1 × 1 | off |
| 5 | off | 3 × 3 | on |
| 6 | off | 3 × 3 | off |
| 7 | off | 1 × 1 | on |
| 8 | off | 1 × 1 | off |

## Layout
- src/synthetic_data.py: synthetic image generation.
- src/masks.py: central gap and held-out mask generation.
- src/metrics.py: masked metrics.
- src/models.py: eight configurable hourglass architecture ablations.
- src/training.py: masked optimization and held-out checkpoint selection.
- baselines/fill.py: grey and harmonic fill baselines.
- experiments/: configuration and runners.
- tests/: lightweight checks.

## Quick start
Python 3.10+ recommended.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest
    python experiments/run_smoke_test.py

The smoke test validates data, masks, and fill baselines. Model/training unit tests are in tests/test_models.py and tests/test_training.py.

Run a short single-model experiment first:

    python experiments/run_experiment.py --image-type gradient --ablation-id 1 --steps 10 --channels 8 --device cpu

Run all configured image types/scales and all eight ablations (this can take substantial time):

    python experiments/run_experiment.py

Each run writes metrics.csv, the best prediction and model state, plus periodic prediction arrays under outputs/.

## Reproducibility
- Keep random seeds explicit and save the configuration with experiment outputs.
- Use held-out MSE to select the primary checkpoint; do not select by ground-truth gap MSE.
- Save metrics and reconstructed images at regular intervals (planned: every 50 steps).
- Do not apply per-image min-max normalization to blurred-noise samples.
