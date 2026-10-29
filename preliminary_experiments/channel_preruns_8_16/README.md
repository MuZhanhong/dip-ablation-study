# Channel Pre-run Results: 8 vs 16 Channels

## Scope

These are preliminary CPU pre-runs for **Ablation 6**, using the channel-prerun workflow. Each run was configured for 100 optimization steps and recorded metrics at steps 50 and 100. These are resource/performance checks, not the formal 3,000-step experiments.

## Resource usage

| Channels | Configured steps | Logged metric steps | Peak RSS | Wall time | Exit code |
|---:|---:|---:|---:|---:|---:|
| 8 | 100 | 50, 100 | 1.54 GiB | 36.90 s | 0 |
| 16 | 100 | 50, 100 | 2.85 GiB | 126.13 s | 0 |

## Metrics

The full logged values are in [metrics_8_channels.csv](metrics_8_channels.csv) and [metrics_16_channels.csv](metrics_16_channels.csv).

### 8 channels

| Step | Train MSE | Held-out MSE | Gap MSE | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|
| 50 | 1.44920e-4 | 0.0892243 | 0.00711483 | 21.4784 | 0.31210 |
| 100 | 3.11152e-6 | 0.0892037 | 0.00689586 | 21.6141 | 0.31504 |

### 16 channels

| Step | Train MSE | Held-out MSE | Gap MSE | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|
| 50 | 1.29534e-4 | 0.0935125 | 0.00605685 | 22.1775 | 0.34116 |
| 100 | 6.83267e-6 | 0.0927081 | 0.00610957 | 22.1399 | 0.33877 |

## Preliminary observations

- Both runs exited successfully.
- From step 50 to 100, held-out MSE changed from 0.0892243 to 0.0892037 for 8 channels, and from 0.0935125 to 0.0927081 for 16 channels.
- SSIM changed from 0.31210 to 0.31504 for 8 channels, and from 0.34116 to 0.33877 for 16 channels.
- Training MSE became very small while held-out MSE remained around 0.09. This indicates a substantial train/held-out discrepancy that should be investigated; these short runs alone do not establish the cause.
- The two logged checkpoints are insufficient to establish long-run stability or to choose a final channel width. The 16-channel run also took substantially longer and used more peak memory in this environment.

## Recommended next step

Run a longer pre-run (for example, 300 steps) with more frequent metric logging and checkpoint selection based on held-out performance before deciding on the full 3,000-step experiments. Compare configurations under the same data split, seed, and runtime environment.

## Provenance and limitations

These values are transcribed from the supplied pre-run metrics and resource-usage records. They describe preliminary Ablation 6 runs only; they should not be presented as results for all eight ablations or as final experimental conclusions.
