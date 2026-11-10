# Unshared Networks: Kernel-Size Diagnostic

## Scope

This note compares the two unshared-kernel contrasts in the first formal sweep:

- Network 2 vs Network 4: upsampling enabled, unshared spatial weights, 3x3 vs 1x1 kernels.
- Network 6 vs Network 8: upsampling disabled, unshared spatial weights, 3x3 vs 1x1 kernels.

Each network was evaluated on Gradient, Blurred Noise, and Stripes using five random seeds, 16 channels, and 2,000 training steps. The reported values are mean gap MSE across seeds, with standard deviation (SD).

The comparisons are controlled for the listed architectural factor: within each pair, the kernel size changes while the upsampling and weight-sharing settings remain fixed.

## Results

### Upsampling enabled: Network 2 vs Network 4

| Image | Network 2 (3x3) | Network 4 (1x1) | Lower mean gap MSE |
|---|---:|---:|---|
| Gradient | 0.007282 ± 0.000339 | 0.006951 ± 0.000676 | Network 4 |
| Blurred Noise | 0.058402 ± 0.005179 | 0.046572 ± 0.007156 | Network 4 |
| Stripes | 0.126526 ± 0.000936 | 0.127021 ± 0.001840 | Network 2 |

Network 4 has lower mean gap MSE on Gradient and Blurred Noise. Its advantage on Blurred Noise is the clearest of these contrasts (approximately 20.3% lower mean gap MSE than Network 2). The Stripes means are nearly identical, with Network 2 marginally lower.

### Upsampling disabled: Network 6 vs Network 8

| Image | Network 6 (3x3) | Network 8 (1x1) | Lower mean gap MSE |
|---|---:|---:|---|
| Gradient | 0.007186 ± 0.000192 | 0.008296 ± 0.000830 | Network 6 |
| Blurred Noise | 0.046970 ± 0.006799 | 0.021645 ± 0.002395 | Network 8 |
| Stripes | 0.126218 ± 0.002959 | 0.125977 ± 0.002696 | Network 8 |

Network 6 performs better on Gradient, while Network 8 performs better on Blurred Noise and is marginally better on Stripes. The largest difference is on Blurred Noise, where Network 8 has approximately 53.9% lower mean gap MSE than Network 6.

## Interpretation

Across the two unshared comparisons, a 3x3 kernel does not produce a consistent advantage over a 1x1 kernel. The direction of the difference depends on image family and upsampling setting. In particular, the 1x1 unshared networks have lower mean gap MSE on Blurred Noise in both comparisons.

This differs from the shared comparison Network 1 vs Network 3, where the 3x3 network had lower mean gap MSE across all three image families. Taken together, the descriptive results suggest that local spatial mixing and spatial weight sharing may interact: the benefit associated with a larger local kernel is clearer in the shared architecture than in the unshared architecture.

This is a hypothesis suggested by the observed pattern, not a demonstrated interaction effect. The sample contains only five seeds per condition, and the tables report descriptive means and SDs rather than inferential tests. Small differences, especially for Stripes, should not be treated as meaningful without further analysis.

## Important evaluation caveat

The formal sweep uses different checkpoint rules by architecture:

- Shared networks are scored at the checkpoint with the lowest held-out MSE.
- Unshared networks are scored at the final training step because held-out locations do not receive location-specific training signals and held-out MSE is not a valid early-stopping criterion for them.

Consequently, these results compare the architectures under the project's predefined evaluation protocol; they should not be interpreted as a comparison under an identical checkpoint-selection rule.

## Limitations

- Five random seeds provide only a limited view of run-to-run variability.
- The current summary does not include paired seed-wise difference tests or confidence intervals.
- This note does not add new experiments or code changes.
- The comparison concerns synthetic 64x64 image families and should not be generalized directly to natural-image inpainting.

## Suggested next comparison

The next controlled contrasts should isolate the effect of upsampling within unshared networks:

- Network 2 vs Network 6: 3x3 kernel, unshared weights, upsampling enabled vs disabled.
- Network 4 vs Network 8: 1x1 kernel, unshared weights, upsampling enabled vs disabled.

These comparisons can test whether the multiscale upsampling pathway helps unshared networks, and whether its effect depends on kernel size. The same five-seed summaries should be supplemented with seed-level trajectories and prediction inspection before making stronger claims.
