# Preliminary Experiments

This folder collects exploratory runs performed before the formal ablation study. These results are for pipeline validation and debugging only; they must not be reported as final experimental findings.

## Available runs

- [`gradient/ablation_1/`](gradient/ablation_1/): CPU trial on the synthetic gradient image, 8 channels, 300 optimization steps.

## Current recorded result

- Best checkpoint selected by held-out MSE: step 250
- Best held-out MSE: 0.0006337648374027821
- Metrics: training MSE, held-out MSE, gap MSE, PSNR, and SSIM

The current repository records metrics only. Model checkpoints and image snapshots were not available during this upload. Formal runs (planned with 32 channels and 3000 steps) should be stored separately from this folder.