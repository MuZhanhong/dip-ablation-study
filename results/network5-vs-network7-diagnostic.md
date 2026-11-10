# Network 5 vs Network 7: Training-Dynamics Diagnostic

Date: 2026-11-10

## Scope and protocol

This diagnostic examines why Network 7 selects its earliest available checkpoint and whether its worsening held-out performance is accompanied by worsening gap reconstruction.

The analysis uses the first formal sweep artifact (`sweep-results-no-weights`):
- 64 × 64 grayscale synthetic images: gradient, blurred noise (scale 4), and stripes.
- 16 model channels and 2,000 training steps.
- Five random seeds per network and image family (30 runs total for Networks 5 and 7).
- Metrics were recorded every 50 steps, beginning at step 50.
- Network 5 and Network 7 are both spatially shared and have upsampling disabled. They differ in the main filter size: 3 × 3 for Network 5 and 1 × 1 for Network 7.

For this diagnostic, trajectories are averaged across the five seeds at each recorded step. The table below reports means, not standard deviations. Checkpoint selection for both networks follows the shared-network rule: choose the checkpoint with minimum held-out MSE.

## Main finding

**Network 7 exhibits a consistent train–held-out divergence across all three image families.** Its training MSE falls substantially from step 50 to step 2,000, while both held-out MSE and gap MSE rise. This is consistent with the model fitting training-location noise-to-target associations that do not transfer to unseen pixel locations. It is more precise to describe this as a failure of spatial generalisation / a memorisation-like pattern than to claim that ordinary parameter overfitting has been proven.

Across all 15 Network 7 runs (3 image families × 5 seeds), the minimum held-out MSE occurs at the first recorded checkpoint, step 50. Therefore, the selected step is not a training stop: training continues to step 2,000, but later checkpoints have worse held-out scores.

### Five-seed mean trajectories

| Image family | Network | Train MSE, step 50 → 2,000 | Held-out MSE, step 50 → 2,000 | Gap MSE, step 50 → 2,000 |
|---|---:|---:|---:|---:|
| Gradient | 5 | 0.000126 → 0.000029 | 0.000208 → 0.000032 | 0.008205 → 0.001327 |
| Gradient | 7 | 0.062070 → 0.005344 | 0.112552 → 0.172813 | 0.033802 → 0.090735 |
| Blurred noise (scale 4) | 5 | 0.0000496 → 0.000000146 | 0.000070 → 0.000003 | 0.006806 → 0.003515 |
| Blurred noise (scale 4) | 7 | 0.007661 → 0.000994 | 0.013907 → 0.022297 | 0.011050 → 0.017697 |
| Stripes | 5 | 0.000767 → 0.000005 | 0.001208 → 0.000024 | 0.096430 → 0.060085 |
| Stripes | 7 | 0.085464 → 0.012582 | 0.169089 → 0.238571 | 0.166538 → 0.230924 |

For Network 7, from step 50 to step 2,000:
- Gradient: training MSE decreases by 91.4%, while held-out MSE increases by 53.5% and gap MSE increases by 168.4%.
- Blurred noise: training MSE decreases by 87.0%, while held-out MSE increases by 60.3% and gap MSE increases by 60.2%.
- Stripes: training MSE decreases by 85.3%, while held-out MSE increases by 41.1% and gap MSE increases by 38.7%.

Network 5 shows the contrasting pattern: held-out MSE decreases over training for all three image families, and its gap MSE is also lower at step 2,000 than at step 50. This supports the interpretation that the 3 × 3 filters provide a useful spatial inductive bias in this no-upsampling, shared-weight setting.

## Architectural interpretation

Network 5 and Network 7 form a direct filter-size comparison:
- Both disable pooling and upsampling, so all layers remain at 64 × 64.
- Both use spatially shared weights.
- Network 5 uses 3 × 3 main filters.
- Network 7 uses 1 × 1 filters throughout the main convolutional stack (the final output layer is also 1 × 1).

A 1 × 1 convolution does not mix neighbouring pixel locations. With fixed random-noise input, the input vector at one location does not encode that location's position in the target image. The model therefore has no local convolutional mechanism to use neighbouring visible pixels to infer the target value at a held-out or missing location. It can reduce training loss by fitting associations at supervised locations, without learning a spatial rule that transfers to other locations.

Batch normalization in training mode does aggregate channel statistics across spatial positions, so locations are not mathematically wholly independent. However, this global statistic does not provide the local neighbourhood mixing supplied by a 3 × 3 convolution and does not encode the arrangement of nearby target pixels.

The trajectory pattern is therefore consistent with insufficient spatial inductive bias in Network 7, rather than merely a need for more training steps.

## What this does and does not establish

Supported by the current sweep:
1. Network 7's held-out MSE is minimised at step 50 in all 15 runs.
2. Its training MSE decreases while held-out and gap MSE worsen from step 50 to step 2,000.
3. Network 5 has substantially better gap reconstruction and improving held-out trajectories in the same no-upsampling/shared setting.
4. The direct Network 5 vs Network 7 comparison is evidence that filter size matters when upsampling is off and weights are spatially shared.

Not established:
- The curves alone do not prove a unique causal mechanism or rule out all optimisation effects.
- This comparison does not establish that 3 × 3 filters are always superior under every architecture or image distribution.
- The saved prediction snapshots are qualitative evidence only. The sweep artifact used here does not include the target images or masks needed to independently re-evaluate the visual reconstructions.

## Implications for reporting

- Retain the corrected checkpoint rule: shared networks use minimum held-out MSE; unshared networks are evaluated at the final step.
- Do not describe Network 7 as having stopped at step 50. It trained for 2,000 steps; step 50 is only its selected checkpoint.
- Report its poor generalisation using the train, held-out, and gap trajectories together.
- Treat the result as evidence for the importance of local spatial mixing in this experimental setting, not as a universal claim about 1 × 1 convolutions.

## Recommended follow-up

No immediate rerun is required to establish the observed trajectory. The next analysis should compare the other matched pairs in the eight-network matrix and test whether the filter-size effect persists when upsampling is enabled (Network 1 vs 3). If an additional diagnostic is later approved, preserve the fixed-noise and mask protocol and inspect per-location predictions and loss maps rather than changing the main sweep retrospectively.
