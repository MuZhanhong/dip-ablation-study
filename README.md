# Deep Image Prior Architecture Ablation Study

Study how architectural choices in Deep Image Prior (DIP) affect recovery of missing regions in synthetic grayscale images.

## Current status

The repository implements the eight-network ablation matrix, masked training, metrics, prediction snapshots, and tests. The current design follows Design v3. The 120-run main sweep has completed; follow-up image-seed generalization experiments are underway. See `results/comprehensive-target-aligned-report.md` for the analysis.

## Experiment setup

- Image size: 64 × 64, grayscale.
- Image families: horizontal gradient; periodic Gaussian-blurred noise at scales ℓ ∈ {0, 1, 2, 4, 8}; diagonal sinusoidal stripes (30° orientation, period 8).
- Missing region: centred 16 × 16 square.
- Held-out validation: 5% of visible pixels, sampled with seed 0 and never inside the missing square.
- Model: four-level DIP-style stack, batch normalization and LeakyReLU(0.2), fixed random-noise input scaled by 0.1, no skip connections, no input jitter, Adam at learning rate 0.01.
- Ablation factors: pooling/upsampling on or off; 3 × 3 vs 1 × 1 main filters; spatially shared vs locally connected internal convolution weights.
- Channel width and training steps are experiment-level settings and must be fixed consistently for a sweep. The main sweep used 16 channels and 2,000 steps.
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

Python 3.11 is used in the GitHub Actions runs.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
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

Use `--seed` to set the model initialization and input-noise seed:

```bash
python experiments/run_experiment.py --image-type gradient --ablation-id 6 --steps 2000 --channels 16 --seed 0 --device cpu
```

Each run writes metrics, predictions, and model state under its output directory. Keep each configuration and seed in a distinct output directory to avoid overwriting results.

## Reproducibility

- The runner seeds PyTorch and NumPy before constructing each model. `--seed` controls model initialization and input noise. `--image-seed` changes the synthetic target image independently; the held-out mask uses its configured seed.
- Keep image, mask, optimizer, learning rate, channel width, and training-step budget fixed when comparing ablations.
- Use held-out MSE for checkpoint selection only on shared networks. Unshared networks are evaluated at the final step.
- Experiment workflows are manual-only; launch them deliberately from GitHub Actions after confirming the protocol.
- Do not apply per-image min-max normalization to blurred-noise samples.

## Running experiments

The runner uses `experiments/config.yaml` unless a different config file is passed with `--config`. Command-line options override the channel width, number of steps, image type, and seeds for an individual run.

For blurred noise, `--seed` and `--image-seed` do different things. The first changes the network initialization and fixed input noise. The second generates a new target image. Keeping them separate lets us check whether a result holds across images rather than just across random initializations.

For example, to run Network 1 on a different blurred-noise image:

```bash
python experiments/run_experiment.py --image-type blurred_noise --blur-scale 4 --ablation-id 1 --channels 16 --steps 2000 --seed 0 --image-seed 2 --device cpu --output-dir outputs/generalization
```

A run produces a `metrics.csv` file, prediction arrays (`.npy`), and model checkpoints (`.pt`). The selected prediction is saved as `selected_prediction.npy`. For shared networks this comes from the best held-out checkpoint; for unshared networks it is the final prediction.

The output path includes the image type, blur scale where relevant, image seed when explicitly supplied, and ablation ID. Model seeds are **not** automatically included in the path, so use a separate `--output-dir` for each model seed when running several seeds locally.

## GitHub Actions

The workflows under `.github/workflows/` handle larger runs. Training workflows are started manually from the Actions tab; pushing a code change does not start a sweep. The test workflow runs on pushes and pull requests.

| Workflow | Purpose |
|---|---|
| `tests.yml` | Unit tests, data checks, and a short training run |
| `sweep.yml` | Main ablation sweep for one image family |
| `collect-results.yml` | Gather sweep artifacts without model checkpoints |
| `blurred-noise-generalization.yml` | Configurable image-seed and model-seed runs |
| `blurred-noise-phase2.yml` | Separate image seeds 2 and 3 into parallel jobs |

The phase-two workflow is set to 2,000 steps, 16 channels, model seed 0, and networks 1, 3, 5, 6, and 8. It covers image seeds 2 and 3; image seed 1 was tested in an earlier run. Each job uploads its metrics, selected prediction, and environment information.

## Reading the results

The main sweep covered eight architectures, three image families, and five model seeds. The analysis is in [the target-aligned report](results/comprehensive-target-aligned-report.md). Additional diagnostic notes are in the `results/` directory.

Gap MSE is useful for comparing reconstruction error, but it does not tell the whole story. A prediction can have a reasonable mean and low variance without reproducing the missing pattern. We therefore also compare predictions with the true gap using spatial correlation and inspect the reconstructed images. This matters particularly for locally connected networks, where the gap pixels do not receive a direct training signal.

The harmonic baseline is a useful reference for smooth images. The baseline used in the existing analysis has access to some pixels held out from network training, so those comparisons are not fully matched-information. A follow-up baseline should use only the pixels available during training.

## Notes for follow-up work

- Compare results across independently generated target images, not only model seeds.
- Check training curves before increasing the step budget. Some networks may still be improving at 2,000 steps.
- Keep the shared/unshared checkpoint rules consistent across comparisons.
- Record the workflow run, commit, environment, and seeds alongside each result.
- Keep exploratory experiments separate from the main sweep so that the original comparisons remain interpretable.

See [PROGRESS.md](PROGRESS.md) for the working plan and completed runs.
