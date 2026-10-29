# Deep Image Prior Architecture Ablation Study

Study how architectural choices in Deep Image Prior (DIP) affect recovery of missing regions in synthetic grayscale images.

## Status
The repository includes the configurable 17-layer DIP-style network builder, masked training loop, held-out-MSE checkpoint selection, logging, and tests. The original DIP vase sanity check and full experimental runs remain to be validated before interpreting final results.

## Experiment setup
- Image size: 64 × 64, grayscale.
- Image families: horizontal gradient; periodic Gaussian-blurred noise at scales ℓ ∈ {0, 1, 2, 4, 8}; diagonal sinusoidal stripes (30° orientation, period 8).
- Missing region: centered 16 × 16 square.
- Held-out validation: 5% of visible pixels, sampled with seed 0 and never inside the missing square.
- Primary model: 4-level, 17-layer DIP-style stack with batch normalization and LeakyReLU(0.2), bilinear upsampling, average pooling, 32-channel fixed random-noise input scaled by 0.1, no skip connections, no input jitter, Adam at learning rate 0.01, up to 3000 steps, logging every 50 steps.
- Ablation factors: pooling/upsampling on or off; 3 × 3 vs 1 × 1 main filters; spatially shared vs locally connected convolution weights. Keep channel width consistent; consider 16 channels if memory is limiting.
- Model selection: held-out MSE. Report gap MSE, PSNR, SSIM, frequency-domain summaries, optimization trajectories, and image snapshots. Gap-oracle metrics are secondary and must not select the primary checkpoint.
- Baselines: grey fill and harmonic fill only.
- Sanity check: reproduce the original DIP inpainting/vase notebook separately. Coordinate inputs are for this sanity check only, not the main experiment.

## Exact switch definitions
- **Upsampling on:** use average-pool downsampling and bilinear upsampling in the hourglass.
- **Upsampling off:** remove both pooling and upsampling. Keep the same layer stack, with every layer operating at 64 × 64.
- **Filter size 1 × 1:** replace the main 3 × 3 convolution layers with 1 × 1 convolutions; layers already 1 × 1 remain unchanged.
- **Weight sharing on:** use normal convolutions, sharing each filter spatially across pixel positions.
- **Weight sharing off:** replace each internal convolution with a locally connected layer with separate weights at every pixel position.
- **Final output layer:** every network, including the locally connected variants, ends with a normal spatially shared `nn.Conv2d(C, out_ch, 1)` and `nn.Sigmoid()`. The final output layer is never locally connected.

## Ablation matrix
| ID | Pooling + upsampling | Filter size | Spatial weight sharing |
|---|---|---|---|
| 1 | on | 3 × 3 | on |
| 2 | on | 3 × 3 | off |
| 3 | on | 1 × 1 | on |
| 4 | on | 1 × 1 | off |
| 5 | off (all layers at 64 × 64) | 3 × 3 | on |
| 6 | off (all layers at 64 × 64) | 3 × 3 | off |
| 7 | off (all layers at 64 × 64) | 1 × 1 | on |
| 8 | off (all layers at 64 × 64) | 1 × 1 | off |

## Layout
- `src/synthetic_data.py`: synthetic image generation.
- `src/masks.py`: central gap and held-out mask generation.
- `src/metrics.py`: masked metrics.
- `src/models.py`: eight configurable 17-layer architecture ablations.
- `src/training.py`: masked optimization and held-out checkpoint selection.
- `baselines/fill.py`: grey and harmonic fill baselines.
- `experiments/`: configuration and runners.
- `tests/`: lightweight checks.

## Quick start
Python 3.10+ recommended.

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest
    python experiments/run_smoke_test.py

Run a short single-model experiment:

    python experiments/run_experiment.py --image-type gradient --ablation-id 1 --steps 10 --channels 8 --device cpu

Run all configured image types/scales and all eight ablations (this can take substantial time):

    python experiments/run_experiment.py

Each run writes `metrics.csv`, the best prediction and model state, plus periodic prediction arrays under `outputs/`.

## Reproducibility
- The runner calls `torch.manual_seed(seed)` before constructing each model, so initial model weights are reproducible for a given seed.
- The fixed noise input is generated from the same seed separately in the training loop.
- Use held-out MSE to select the primary checkpoint; do not select by ground-truth gap MSE.
- Save metrics and reconstructed images at regular intervals (planned: every 50 steps).
- Do not apply per-image min-max normalization to blurred-noise samples.
