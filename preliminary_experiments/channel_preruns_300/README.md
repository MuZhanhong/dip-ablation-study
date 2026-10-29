# Ablation 6: 300-Step Channel Pre-Runs

## Purpose

These runs validate the CPU training pipeline for Ablation 6 at 8 and 16 channels. They are preliminary diagnostics, not final ablation-study evidence.

## Run details

- Workflow: [channel-prerun-300.yml](../../../.github/workflows/channel-prerun-300.yml)
- GitHub Actions run: [Run 40966313992](https://github.com/MuZhanhong/dip-ablation-study/actions/runs/40966313992)
- Commit: `12f9ced212e50726bd904abafa9a8e5314360629`
- Status: both jobs completed successfully; both artifact uploads succeeded.
- Metrics were logged at six checkpoints over 300 optimization steps.
- The artifact ZIPs contain metrics, predictions, checkpoints, and resource reports. They are retained by GitHub Actions for 14 days from upload.

## Summary

| Channels | Best step | Best held-out MSE | Final train MSE | Final held-out MSE | Final gap MSE |
|---:|---:|---:|---:|---:|---:|
| 8 | 100 | 0.08920369297266006 | 0.0000006058254484742065 | 0.08960745483636856 | 0.006725386716425419 |
| 16 | 300 | 0.0923883318901062 | 0.00000242729743149539 | 0.0923883318901062 | 0.006101427599787712 |

See [summary_metrics.csv](summary_metrics.csv) for machine-readable values.

## Artifacts

- [8-channel artifact ZIP](https://github.com/MuZhanhong/dip-ablation-study/actions/runs/40966313992/artifacts/12937424588)
- [16-channel artifact ZIP](https://github.com/MuZhanhong/dip-ablation-study/actions/runs/40966313992/artifacts/12939035461)

Each archive includes the detailed `metrics.csv`, `summary.txt`, `resource_usage.txt`, predictions, and model checkpoints. The raw per-checkpoint CSVs and resource reports remain in the artifacts; this repository summary does not reconstruct or approximate those files.

## Interpretation and limitations

The 8-channel run achieved the lower best held-out MSE (0.08920 at step 100), while the 16-channel run's best observed value occurred at step 300 (0.09239). Both runs reached very low training MSE, with a non-zero train/held-out gap. This is a descriptive observation from a single preliminary run per configuration; it does not establish statistical significance or a final architecture recommendation.

These are CPU pre-runs with a short 300-step budget. Use multiple random seeds and the planned formal training protocol before drawing conclusions.
