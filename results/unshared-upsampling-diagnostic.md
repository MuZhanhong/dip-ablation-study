# Upsampling Diagnostic for Unshared Networks

## Scope

This note examines the effect of enabling the upsampling pathway in unshared networks, using two controlled architectural contrasts:

- Network 2 vs Network 6: unshared 3x3 kernels, upsampling enabled vs disabled.
- Network 4 vs Network 8: unshared 1x1 kernels, upsampling enabled vs disabled.

The first formal sweep used 16 channels, 2,000 training steps, and five random seeds per condition. Values below are mean gap MSE ± standard deviation across seeds. Lower gap MSE is better.

## Results

### Unshared 3x3 networks: Network 2 vs Network 6

| Image | Network 2 (upsampling on) | Network 6 (upsampling off) | Lower mean gap MSE |
|---|---:|---:|---|
| Gradient | 0.007282 ± 0.000339 | 0.007186 ± 0.000192 | Network 6 |
| Blurred Noise | 0.058402 ± 0.005179 | 0.046970 ± 0.006799 | Network 6 |
| Stripes | 0.126526 ± 0.000936 | 0.126218 ± 0.002959 | Network 6 |

Network 6 has slightly lower mean gap MSE on all three image families. The difference is negligible for Gradient and Stripes, while the reduction for Blurred Noise is approximately 19.6%.

### Unshared 1x1 networks: Network 4 vs Network 8

| Image | Network 4 (upsampling on) | Network 8 (upsampling off) | Lower mean gap MSE |
|---|---:|---:|---|
| Gradient | 0.006951 ± 0.000676 | 0.008296 ± 0.000830 | Network 4 |
| Blurred Noise | 0.046572 ± 0.007156 | 0.021645 ± 0.002395 | Network 8 |
| Stripes | 0.127021 ± 0.001840 | 0.125977 ± 0.002696 | Network 8 |

The effect is mixed. Upsampling helps on Gradient, but disabling it is associated with lower mean gap MSE on Blurred Noise (approximately 53.5% lower) and a very small improvement on Stripes.

## Interpretation

There is no uniform benefit from enabling upsampling in these unshared architectures.

- For unshared 3x3 networks, disabling upsampling gives slightly lower mean gap MSE across all three image families, although only the Blurred Noise difference is substantial in magnitude.
- For unshared 1x1 networks, the direction depends on image family: upsampling helps Gradient, while no upsampling performs better on Blurred Noise and marginally on Stripes.

The contrast between the 3x3 and 1x1 results suggests that the effect of upsampling may depend on kernel size. In particular, the upsampling pathway does not consistently rescue the limited spatial generalization observed in unshared networks.

These are descriptive patterns, not proof of a statistically significant interaction. Only five seeds were run per condition, and no paired seed-wise inferential analysis is reported here.

## Relationship to other comparisons

This result complements the kernel-size diagnostic:

- Among shared networks with upsampling enabled, Network 1 (3x3) outperformed Network 3 (1x1) on all three image families.
- Among unshared networks, the kernel-size comparison did not show a consistent 3x3 advantage.
- Within unshared networks, enabling upsampling did not consistently improve reconstruction.

Together, these results are consistent with the possibility that spatial weight sharing is important for turning local and multiscale architectural structure into generalization to unseen locations. This remains an interpretation to test, not an established causal conclusion.

## Evaluation protocol and limitations

Shared networks are selected using the best held-out-MSE checkpoint. Unshared networks are evaluated at the final training step because held-out pixels receive no location-specific training signal; held-out MSE is therefore not a valid early-stopping criterion for these networks. All comparisons here follow that predefined protocol.

The results are based on five seeds, synthetic image families, and aggregate mean/SD summaries. The present analysis does not include seed-paired differences, confidence intervals, full training-trajectory inspection, or a new experiment. Small differences should not be over-interpreted.

## Recommended next step

The main architectural comparisons are now covered at the level of aggregate gap-MSE summaries. The next useful step is a robustness and interaction analysis using the existing seed-level sweep data:

1. Compute seed-matched differences for the controlled pairs (N1/N3, N2/N4, N5/N7, N6/N8, N2/N6, and N4/N8).
2. Report uncertainty for each paired contrast and avoid relying only on separate means and SDs.
3. Inspect selected checkpoints and prediction images for the pairs where the aggregate differences are largest, especially the Blurred Noise cases.
4. Update the project-level results summary only after these checks, keeping the checkpoint-selection caveat explicit.

No new training run is required for this next analysis if the existing per-seed artifacts contain the necessary metrics and predictions.
