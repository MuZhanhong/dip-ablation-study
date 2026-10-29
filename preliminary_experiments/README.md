# Preliminary Experiments

This folder collects exploratory runs performed before the formal ablation study. These results are for pipeline validation and debugging only; they must not be reported as final experimental findings.

## Available runs

- [`gradient/ablation_1/`](gradient/ablation_1/): CPU trial on the synthetic gradient image, 8 channels, 300 optimization steps.
- [`channel_preruns_8_16/`](channel_preruns_8_16/): earlier 100-step CPU pre-runs for Ablation 6 at 8 and 16 channels.
- [`channel_preruns_300/`](channel_preruns_300/): 300-step CPU pre-runs for Ablation 6 at 8 and 16 channels; includes a summary table and machine-readable metrics.

## 300-step channel pre-run summary

- [Detailed report and artifact links](channel_preruns_300/README.md)
- [Summary metrics CSV](channel_preruns_300/summary_metrics.csv)
- GitHub Actions run: [40966313992](https://github.com/MuZhanhong/dip-ablation-study/actions/runs/40966313992)

The 300-step workflow completed successfully for both channel configurations, and both artifact archives were uploaded. The archives contain the detailed per-checkpoint metrics, predictions, model checkpoints, and resource reports; they are stored as GitHub Actions artifacts with a 14-day retention period rather than committed as large binary files to this repository.

## Earlier gradient trial

- Best checkpoint selected by held-out MSE: step 250
- Best held-out MSE: 0.0006337648374027821
- Metrics: training MSE, held-out MSE, gap MSE, PSNR, and SSIM

All runs in this directory are preliminary diagnostics. Formal runs (planned with 32 channels and 3000 steps) should be stored separately from this folder, and conclusions should be supported by repeated runs with multiple random seeds.
