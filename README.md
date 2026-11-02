# Deep Image Prior Architecture Ablation Study

Study how architectural choices in Deep Image Prior (DIP) affect recovery of missing regions in synthetic grayscale images.

## Current status

The repository implements the eight-network ablation matrix, masked training, metrics, prediction snapshots, and tests. The current design follows Design v3. Pilot findings are preliminary; the main ablation sweeps have not yet been completed.

## Experiment setup

- Image size: 64 × 64, grayscale.
- Image families: horizontal gradient; periodic Gaussian-blurred noise at scales ℓ ∈ {0, 1, 2, 4, 8}; diagonal sinusoidal stripes (30° orientation, period 8).
- Missing region: centred 16 × 16 square.
- Held-out validation: 5% of visible pixels, sampled with seed 0 and never inside the missing square.
- Model: four-level DIP-style stack, batch normalization and LeakyReLU(0.2), fixed random-noise input scaled by 0.1, no skip connections, no input jitter, Adam at learning rate 0.01.
- Ablation factors: pooling/upsampling on or off; 3 × 3 vs 1 × 1 main filters; spatially shared vs locally connected internal convolution weights.
- Channel width and training steps are experiment-level settings and must be fixed consistently for a sweep. Confirm the final values against the current width test and team decision before launching the main sweep.
- Baselines: grey fill and harmonic fill.

## Exact switch definitions

- **Upsampling on:** use average-pool downsampling and bilinear upsampling in the hourglass.
- **Upsampling off:** remove both pooling and upsampling. Keep all layers operating at 64 × 64.
- **Filter size 1 × 1:** replace the main 3 × 3 convolution layers with 1 × 1 convolutions; layers already 1 × 1 remain unchanged.
- **Weight sharing on:** use normal convolutions, sharing each filter spatially across pixel positions.
- **Weight sharing off:** use locally connected internal convolution layers with separate weights at every spatial position.
- **Final output layer:** it follows the weight-sharing switch. Shared networks use a spatially shared 1 × 1 convolution; unshared networks use a locally connected output layer. Do not force a shared output layer onto the unshared variants.

## Evaluation and checkpoint selection

- **Shared networks:** select the checkpoint with the lowest held-out MSE.
- **Unshared networks:** score the final training step. A held-out pixel receives no location-specific training signal in an unshared network, just as a gap pixel does, so held-out MSE cannot provide a meaningful early-stopping signal.
- Report gap MSE as the primary reconstruction score, alongside held-out MSE, PSNR, SSIM, optimization trajectories, and prediction snapshots as appropriate. Gap MSE is an evaluation metric, not a checkpoint-selection criterion.
- Generate predictions in training mode, consistent with the current implementation and pilot protocol.
- Do not interpret held-out MSE for unshared networks as evidence that their gap reconstruction is improving or failing; inspect final-step gap metrics and predictions instead.

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
- `src/models.py`: eight configurable architecture ablations.
- `src/training.py`: masked optimization, checkpoint selection, and prediction saving.
- `baselines/fill.py`: grey and harmonic fill baselines.
- `experiments/`: configuration and runners.
- `.github/workflows/`: manual experiment workflows and automated tests.
- `tests/`: lightweight checks.

## Quick start

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python experiments/run_smoke_test.py
```

Run a short single-model experiment:

```bash
python experiments/run_experiment.py --image-type gradient --ablation-id 1 --steps 10 --channels 8 --device cpu
```

Override the model/noise seed with `--seed`:

```bash
python experiments/run_experiment.py --image-type gradient --ablation-id 6 --steps 2000 --channels 16 --seed 0 --device cpu
```

Each run writes metrics, predictions, and model state under its output directory. Keep each configuration and seed in a distinct output directory to avoid overwriting results.

## Reproducibility

- The runner seeds PyTorch and NumPy before constructing each model. The `--seed` option overrides model initialization and fixed-noise seed; synthetic image generation and held-out mask remain fixed by the configuration.
- Keep image, mask, optimizer, learning rate, channel width, and training-step budget fixed when comparing ablations.
- Use held-out MSE for checkpoint selection only on shared networks. Unshared networks are evaluated at the final step.
- Experiment workflows are manual-only; launch them deliberately from GitHub Actions after confirming the protocol.
- Do not apply per-image min-max normalization to blurred-noise samples.
