# Development Progress

Last updated: 2026-10-27

This file records the implementation and local validation completed so far. Results below are preliminary smoke/trial runs, not the full study.

## Completed

- Downloaded and extracted the repository locally on Windows.
- Created and activated a Python virtual environment using Python 3.12.7.
- Installed the packages in `requirements.txt`.
- Ran the test suite successfully: **19 tests passed**.
- Ran the CLI help check for `experiments/run_experiment.py`; it exposes options for image type, blur scale, ablation ID, steps, channel width, output directory, and device.
- Checked `experiments/config.yaml` against the planned setup:
  - seed 0; 64 × 64 grayscale images
  - centered 16 × 16 missing square
  - 5% held-out visible pixels with seed 0
  - 4-level model, fixed noise scale 0.1, no input jitter or encoder-decoder skip connections
  - average-pooling downsampling, Adam, learning rate 0.01, 3000 steps, logging every 50 steps
  - eight combinations of upsampling, filter size, and weight sharing
  - checkpoint selection by held-out MSE
- Completed a 10-step CPU smoke run for the gradient image, ablation 1, using 8 channels.
- Completed a 300-step CPU trial for the gradient image, ablation 1, using 8 channels. The run saved output locally under `outputs/gradient/ablation_1`.

## Preliminary 300-step trial

Command:

```powershell
python experiments/run_experiment.py --image-type gradient --ablation-id 1 --steps 300 --channels 8 --device cpu
```

Observed result:

- Best checkpoint step: **250**
- Best held-out MSE: **0.0006337648374027821** (approximately 0.000633765)
- Gap MSE at step 250: **0.00014235987327992916**
- PSNR at step 250: **38.4661**
- SSIM at step 250: **0.97735**

The trial's `metrics.csv` records metrics at steps 50, 100, 150, 200, 250, and 300. These figures are a short-run software check with 8 channels and 300 steps; they must not be reported as final study results or compared directly with the planned 32-channel, 3000-step runs.

## Current implementation limitations / items to verify

- Frequency-domain summaries are requested in the configuration, but the frequency-domain analysis is **not yet implemented/validated**.
- The fully connected baseline runner has not yet been implemented/validated.
- The separate original DIP vase/inpainting sanity check has not yet been integrated or run.
- The grey-fill and harmonic-fill baseline functions exist, but the full baseline evaluation workflow still needs to be checked.
- Confirm that the operational definition of `upsampling: false` (learned transposed convolution) matches the intended research protocol; `upsampling: true` uses bilinear interpolation.
- The full 32-channel, 3000-step experiment matrix has **not** been run yet.
- Local `outputs/` files are not included in this progress record; results should be preserved locally and added only if desired and appropriate.

## Next steps

1. Confirm the upsampling ablation definition against the research plan.
2. Implement and test frequency-domain summaries and the fully connected baseline workflow.
3. Add or document the separate original DIP vase sanity check.
4. Run and inspect the full planned configuration, keeping each run in a distinct output directory to avoid overwriting trial results.
5. Summarize all runs using held-out MSE for checkpoint selection; use gap metrics only for evaluation, never for selecting the primary checkpoint.
