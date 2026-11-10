# Architecture interaction and blurred-noise gap diagnostics

## Data and evaluation

First formal sweep: 3 image families, 8 networks, 5 paired seeds, width 16, 2,000 training steps. Shared networks use the checkpoint minimizing held-out MSE; unshared networks use their final step. No new training was performed.

For a fixed upsampling setting, define the 3×3 advantage as `gap_MSE(1×1) − gap_MSE(3×3)`. The interaction contrast is `3×3 advantage(shared) − 3×3 advantage(unshared)`, calculated per matched seed. Positive values mean the 3×3 kernel advantage is larger with spatial weight sharing. Confidence intervals are descriptive 95% t intervals (n=5), unadjusted for multiplicity; do not interpret them as definitive factorial-model inference.

## Interaction contrasts

| Image | Upsampling | Mean interaction | 95% interval | Positive seeds |
|---|---|---:|---|---:|
| gradient-na | on | +0.000431 | [-0.000517, +0.001379] | 3/5 |
| gradient-na | off | +0.031468 | [+0.025589, +0.037348] | 5/5 |
| blurred_noise-4 | on | +0.012994 | [-0.000734, +0.026721] | 4/5 |
| blurred_noise-4 | off | +0.032859 | [+0.025022, +0.040696] | 5/5 |
| stripes-na | on | +0.055303 | [-0.002279, +0.112885] | 4/5 |
| stripes-na | off | +0.105073 | [+0.087132, +0.123014] | 5/5 |

All three no-upsampling contrasts are positive for every seed. The interaction is less consistent when upsampling is enabled. This pattern is compatible with an interplay between shared local filtering and spatial generalization, but checkpoint selection differs between shared and unshared networks and confounds any strict causal interpretation.

## N6 vs N8: blurred-noise prediction statistics

At the final step, N6 (unshared 3×3, no upsampling) has mean gap MSE 0.046970; N8 (unshared 1×1, no upsampling) has mean gap MSE 0.021645. Over five seeds, the mean within-gap pixel standard deviation is 0.194315 for N6 and 0.106589 for N8, while the average gap pixel means are 0.497695 and 0.499827, respectively. For the surrounding 32×32 ring excluding the gap, the mean predicted standard deviations are 0.105941 and 0.107873.

Thus N8's predictions in the gap are markedly less variable than N6's; N8 is **not** simply constant, as its gap range and SD are nonzero. A lower-variance prediction close to 0.5 can have lower MSE on blurred noise even if it reconstructs less spatial detail. This is a plausible explanation, **not** a demonstrated mechanism: the archive does not include verified target arrays/masks in this analysis, so we have not computed target-aligned gap correlation, target variance, or a matched constant-mean baseline from target pixels.

## Implications and next steps

1. Inspect target-aligned prediction error maps and compare gap variance/correlation with ground truth once the exact target-generation code and mask are verified.
2. Consider reporting structural diagnostics (spatial correlation, power spectra, edge alignment) alongside gap MSE to separate smooth averaging from genuine texture recovery.
3. Treat small-n confidence intervals cautiously; replicate key contrasts with more seeds if computationally feasible.
4. Keep checkpoint protocol explicit in every comparison and avoid attributing the shared/unshared interaction solely to the architectural switch.

## Files

The analysis generated `kernel_sharing_interaction_by_seed.csv` (30 matched-seed interaction contrasts), `kernel_sharing_interaction_summary.csv` (six mean contrasts and descriptive intervals), and `blurred_noise_gap_prediction_statistics.csv` (10 final prediction summaries). These local artifacts are not included in this report commit.
