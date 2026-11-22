# Deep Image Prior: Architecture Ablation Study

This project looks at how much of an image a neural network can reconstruct from its architecture alone. We use **Deep Image Prior (DIP)**-style networks to fill a missing square in small synthetic images, without pretraining on an image dataset.

The main question is which architectural choices help information from the visible part of an image reach the missing region. We vary spatial weight sharing, convolution kernel size, and pooling/upsampling, then compare the reconstructions against simple interpolation baselines.

## Where the project stands

The main sweep is complete: **8 architectures × 3 image families × 5 model seeds = 120 runs**, using 16 channels and 2,000 optimization steps per run. We have also started testing whether the blurred-noise findings carry over to independently generated images.

- [Full reconstruction analysis](results/comprehensive-target-aligned-report.md)
- [Development notes and next steps](PROGRESS.md)
- [GitHub Actions runs](../../actions)

The results below refer to the completed main sweep. They should not be taken as results from the image-generalization follow-up.

## What we test

Each image is 64 × 64 pixels, with a 16 × 16 square removed from its centre. Another 5% of the visible pixels are held out for validation. The network receives a fixed random-noise tensor as input and is fitted to the remaining visible pixels. The missing region is never included in the training loss.

We use three image families:

| Image | Description | What it tests |
|---|---|---|
| Gradient | Smooth horizontal change from black to white | Simple global structure |
| Blurred noise | Gaussian-smoothed random noise, with several blur scales | Spatial structure at different scales |
| Stripes | Sinusoidal stripes at 30°, period 8 | Continuation of a repeating pattern |

The formal comparison uses blurred noise at scale 4. Other blur scales are available in the configuration.

### Network variants

All eight networks use four levels, batch normalization, LeakyReLU, and a sigmoid output. There are no skip connections or input jitter. The three switches are:

- **Pooling/upsampling:** an average-pooling encoder and bilinear-upsampling decoder, or neither (all layers remain at full resolution).
- **Kernel size:** 3 × 3 or 1 × 1 in the main convolution layers.
- **Spatial weight sharing:** ordinary convolutions, or locally connected layers with separate weights at each spatial position.

| Network | Pooling/upsampling | Main kernel | Spatial sharing |
|---|---|---|---|
| N1 | On | 3 × 3 | Shared |
| N2 | On | 3 × 3 | Unshared |
| N3 | On | 1 × 1 | Shared |
| N4 | On | 1 × 1 | Unshared |
| N5 | Off | 3 × 3 | Shared |
| N6 | Off | 3 × 3 | Unshared |
| N7 | Off | 1 × 1 | Shared |
| N8 | Off | 1 × 1 | Unshared |

The final 1 × 1 output layer also follows the sharing switch. In particular, the unshared networks do **not** end with a shared convolution.

### Training and evaluation

The main sweep uses Adam with learning rate 0.01, a fixed noise input scaled by 0.1, and 2,000 steps. Metrics are recorded every 50 steps.

Checkpoint selection differs for a reason:

- **Shared networks (N1, N3, N5, N7):** select the checkpoint with the lowest held-out MSE.
- **Unshared networks (N2, N4, N6, N8):** use the final training step. Their held-out locations do not receive a location-specific training signal, so held-out error is not a reliable stopping criterion.

Gap MSE is calculated against the known target after training, not used to choose a checkpoint. Predictions are evaluated in training mode so batch normalization uses the same kind of statistics as during fitting.

## Main results

The table shows mean **gap MSE across five model seeds** for the three images used in the main sweep. Lower is better.

| Network | Gradient | Blurred noise (scale 4) | Stripes |
|---|---:|---:|---:|
| N1 | 0.000020 | 0.001071 | 0.047837 |
| N2 | 0.007282 | 0.058402 | 0.126526 |
| N3 | 0.000121 | 0.002235 | 0.103635 |
| N4 | 0.006951 | 0.046572 | 0.127021 |
| N5 | 0.001223 | 0.003515 | 0.061705 |
| N6 | 0.007186 | 0.046970 | 0.126218 |
| N7 | 0.033802 | 0.011050 | 0.166538 |
| N8 | 0.008296 | 0.021645 | 0.125977 |
| Constant grey (0.5) | 0.005354 | 0.010777 | 0.125000 |
| Harmonic fill | 0.000000 | 0.001072 | 0.119586 |

A few findings stand out:

1. **Weight sharing matters.** N1, N3, and N5 generally reconstruct the missing structure much better than their unshared counterparts.
2. **Low MSE does not necessarily mean the pattern was recovered.** On blurred noise, N8 has a lower MSE than N6, but both have almost no pixelwise alignment with the true gap (mean Pearson correlations about 0.035 and 0.051).
3. **Simple interpolation is a strong baseline for smooth images.** Harmonic filling reconstructs the linear gradient almost exactly and performs about as well as N1 on blurred noise.
4. **The stripes are harder.** N1 and N5 improve on harmonic interpolation, although the quality varies across model seeds.

These observations are descriptive. The main sweep changes model seeds but uses the same target image within each image family, so it does not yet show that the findings generalize across new images.

For spatial correlation, gradient error, prediction variance, and the full interpretation, see the [target-aligned results report](results/comprehensive-target-aligned-report.md).

## Getting started

The GitHub Actions experiments use **Python 3.11**. Install the dependencies from the repository root.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python experiments/run_smoke_test.py
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python experiments/run_smoke_test.py
```

The smoke test checks image generation and the baseline fills; it does not train all eight networks.

### Run one network

A quick CPU check:

```bash
python experiments/run_experiment.py --image-type gradient --ablation-id 1 --steps 10 --channels 8 --device cpu
```

A full-length blurred-noise run on a different target image:

```bash
python experiments/run_experiment.py --image-type blurred_noise --blur-scale 4 --ablation-id 1 --channels 16 --steps 2000 --seed 0 --image-seed 2 --device cpu --output-dir outputs/image_seed_2_model_seed_0
```

The runner reads `experiments/config.yaml` by default. Use `--config` for a different config, or override individual settings from the command line. Omitting `--ablation-id` runs all eight networks; omitting `--image-type` runs the configured image families and blur scales. This can take considerably longer than a single run.

**The two seeds have different roles:**

- `--seed` sets the model initialization and fixed input-noise seed.
- `--image-seed` changes the generated target image independently.
- The validation mask has its own seed in `experiments/config.yaml`.

The output folder includes the image type, blur scale, ablation ID, and image seed when explicitly supplied. **It does not automatically include the model seed.** Use a separate `--output-dir` for each model seed to avoid overwriting results.

### Output files

Each run saves files under its experiment directory:

| File | Contents |
|---|---|
| `metrics.csv` | Step-by-step training, held-out, and gap metrics |
| `selected_prediction.npy` | Prediction used for the reported result |
| `best_prediction.npy` | Prediction with lowest held-out MSE |
| `final_prediction.npy` | Prediction at the last step |
| `prediction_step_*.npy` | Saved prediction snapshots |
| `best_model_state.pt`, `final_model_state.pt` | PyTorch model states |

For unshared networks, `selected_prediction.npy` is the final prediction, not necessarily `best_prediction.npy`.

## Staged core study (new protocol)

The next core study is split into four manually started batches, one for each blurred-noise scale: **2, 3, 4, and 6**. Each batch covers eight independent image seeds and three model seeds. The workflow is in [`core-study.yml`](.github/workflows/core-study.yml).

For now, the workflow uses the existing shared networks **N1, N3, N5, and N7**. That gives 4 networks × 8 images × 3 model seeds = **96 runs per scale**, or **384 runs across four scales**. Tom's proposed replacement architecture configurations have not yet been specified in enough detail to implement safely. The workflow therefore represents a runnable staging protocol, **not a confirmed final architecture matrix**.

To start a batch, open [Core study - one blur scale](../../actions/workflows/core-study.yml), choose the scale, check the step count and width, and select **Run workflow**. Run a single batch first and review the outputs before launching the others. Each job processes three model seeds and uploads metrics, selected predictions, the commit SHA, and a dependency snapshot. It deliberately does not upload large model checkpoints.

The previous 120-run sweep remains unchanged. Do not combine its results with the new study without accounting for differences in image seeds, blur scales, and any later architecture changes.

## Running on GitHub Actions

Expensive training jobs are **manual-only**. Pushing changes runs the tests but does not start a new experiment sweep.

| Workflow | Use |
|---|---|
| [Tests and smoke check](../../actions/workflows/tests.yml) | Unit tests and short CPU training |
| [Staged core study](../../actions/workflows/core-study.yml) | One blurred-noise scale per batch; independent image and model seeds |
| [Main sweep](../../actions/workflows/sweep.yml) | Eight networks × five model seeds for one image family |
| [Collect results](../../actions/workflows/collect-results.yml) | Gather artifacts without large model weights |
| [Blurred-noise generalization](../../actions/workflows/blurred-noise-generalization.yml) | Choose image seeds, model seeds, and step count |
| [Phase 2: independent images](../../actions/workflows/blurred-noise-phase2.yml) | Parallel runs for image seeds 2 and 3 |

To launch a manual workflow, open it in the **Actions** tab, select **Run workflow**, check the inputs, and start the run. Download the uploaded artifacts after it finishes.

The phase-two workflow is configured for networks N1, N3, N5, N6, and N8, with 16 channels, 2,000 steps, model seed 0, and image seeds 2 and 3. Image seed 1 was run separately in [Actions run 43706330133](../../actions/runs/43706330133). The existence of the workflow does not by itself mean all its jobs have finished.

## Repository structure

```text
src/                 Models, synthetic images, masks, metrics, training
baselines/           Constant-grey and harmonic interpolation
experiments/         Configuration and command-line runners
tests/               Unit and smoke tests
.github/workflows/   CI and manually triggered experiments
results/             Analysis reports and architecture diagnostics
preliminary_experiments/  Earlier checks and pilot artifacts
PROGRESS.md          Project log and next steps
```

## Limitations and next steps

There are several important limits to the current comparison:

- **Independent images:** five model seeds on one target do not replace testing several target images. The image-seed follow-up addresses this.
- **Baseline information:** the harmonic fill in the existing report uses true pixels outside the gap, including the 5% held out from network training. A matched-information version is still needed for a strictly fair comparison.
- **Different checkpoint rules:** shared and unshared networks cannot use the same held-out stopping rule under the present architecture.
- **Metric interpretation:** gap MSE can reward low-variance predictions that do not follow the actual pattern. Correlation, gradient error, and the saved images should be considered together.
- **Compute and reproducibility:** keep the image, mask, optimizer, width, training budget, code revision, and environment documented for every comparison. CPU runs can be slow, especially for locally connected networks.

The immediate follow-up is to compare independently generated blurred-noise images, inspect convergence, and revisit the harmonic baseline using matched training information. Other possible extensions include different gap sizes, more detailed stripe analysis, and exploratory architecture variants; these are not part of the completed 120-run comparison.

## Further reading

- Ulyanov, Vedaldi, and Lempitsky (2018), *Deep Image Prior*.
- Heckel and Hand (2019), *Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks*.
- Rahaman et al. (2019), *On the Spectral Bias of Neural Networks*.

For project-specific decisions and experiment history, see [PROGRESS.md](PROGRESS.md).
