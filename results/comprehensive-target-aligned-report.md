# Target-aligned reconstruction assessment across all eight architectures

**Formal comprehensive results report — 10 November 2026**

## Executive summary

We reconstructed the deterministic ground-truth images and assessed all 120 selected predictions (eight networks × three image families × five seeds). Every recomputed gap MSE agrees with the corresponding recorded metric to within 2×10⁻⁶; no discrepancies were found. The spatially shared networks N1, N3, and N5 achieve both lower gap MSE and substantially stronger target-aligned pixel correlation than the unshared networks. Low MSE alone can nevertheless be misleading: on blurred noise, N8 beats N6 on MSE but neither recovers target-aligned variation (mean Pearson r ≈ 0.035 and 0.051 respectively). The simple harmonic baseline is especially competitive on smooth images, and exactly reconstructs the linear gradient up to floating-point precision.

## Dataset, evaluation and verification

- Source: GitHub Actions collector run 41651125146, artifact `sweep-results-no-weights` (ID 13237378454), containing the original `selected_prediction.npy` and `metrics.csv` files.
- Inputs: deterministic 64×64 images from `src/synthetic_data.py` at repository configuration seed 0: horizontal gradient; periodic FFT-Gaussian blurred noise at blur scale 4, base 0.5, amplitude 0.1; diagonal sinusoidal stripes at angle 30° and period 8.
- Region: central 16×16 square, rows and columns 24–39 inclusive, matching `src/masks.py`.
- Model outputs: archived `selected_prediction.npy` for each network/seed, rather than reconstructing checkpoints or choosing a new step. Shared networks (N1, N3, N5, N7) use best-heldout checkpoint; unshared networks (N2, N4, N6, N8) use final step.
- Metrics: mean squared pixel error on the 256 gap pixels; Pearson correlation across the same 256 pixel pairs; prediction standard deviation and mean; finite-difference gradient MSE on the 16×16 gap as a complementary spatial-structure diagnostic. Correlation for constant baselines is undefined, not zero.
- Baselines: constant 0.5, global mean of all non-gap pixels, mean of the one-pixel boundary ring, and a four-neighbor discrete Laplace harmonic interpolation with all pixels outside the gap held fixed to the true image. The latter is an **oracle-style image-boundary baseline** using observed pixels; it is not a trained neural network and is independent of the 5% held-out validation split.
- Validation: all 120 recalculated model gap MSE values match their logged selected-step values within 2×10⁻⁶.

## Full results

All model values are five-seed means. `r` is mean per-seed gap Pearson correlation (not a correlation computed from averaged predictions). Baseline values are deterministic for each image and have no across-seed SD.

### Gradient

| Network or baseline | Gap MSE | Pearson r | Gap prediction SD | Gradient MSE |
|---|---:|---:|---:|---:|
| N1 | 0.000020 | +0.999 | 0.0732 | 0.000003 |
| N2 | 0.007282 | -0.035 | 0.0414 | 0.003519 |
| N3 | 0.000121 | +0.991 | 0.0744 | 0.000017 |
| N4 | 0.006951 | -0.010 | 0.0392 | 0.003330 |
| N5 | 0.001223 | +0.928 | 0.0736 | 0.000119 |
| N6 | 0.007186 | -0.014 | 0.0418 | 0.003681 |
| N7 | 0.033802 | -0.029 | 0.1660 | 0.057229 |
| N8 | 0.008296 | +0.022 | 0.0557 | 0.006444 |
| constant_0.5 | 0.005354 | undefined | 0.0000 | 0.000126 |
| visible_mean | 0.005354 | undefined | 0.0000 | 0.000126 |
| boundary_mean | 0.005354 | undefined | 0.0000 | 0.000126 |
| harmonic | 0.000000 | +1.000 | 0.0732 | 0.000000 |

### Blurred Noise (scale 4)

| Network or baseline | Gap MSE | Pearson r | Gap prediction SD | Gradient MSE |
|---|---:|---:|---:|---:|
| N1 | 0.001071 | +0.961 | 0.0888 | 0.000073 |
| N2 | 0.058402 | +0.007 | 0.2196 | 0.099304 |
| N3 | 0.002235 | +0.930 | 0.0948 | 0.000148 |
| N4 | 0.046572 | +0.029 | 0.1898 | 0.074507 |
| N5 | 0.003515 | +0.868 | 0.0773 | 0.000209 |
| N6 | 0.046970 | +0.051 | 0.1943 | 0.076407 |
| N7 | 0.011050 | -0.008 | 0.0451 | 0.004392 |
| N8 | 0.021645 | +0.035 | 0.1066 | 0.023348 |
| constant_0.5 | 0.010777 | undefined | 0.0000 | 0.000330 |
| visible_mean | 0.008856 | undefined | 0.0000 | 0.000330 |
| boundary_mean | 0.007265 | undefined | 0.0000 | 0.000330 |
| harmonic | 0.001072 | +0.952 | 0.0645 | 0.000068 |

### Stripes (period 8, 30°)

| Network or baseline | Gap MSE | Pearson r | Gap prediction SD | Gradient MSE |
|---|---:|---:|---:|---:|
| N1 | 0.047837 | +0.805 | 0.3258 | 0.012414 |
| N2 | 0.126526 | +0.002 | 0.0397 | 0.040393 |
| N3 | 0.103635 | +0.589 | 0.3360 | 0.020971 |
| N4 | 0.127021 | -0.018 | 0.0386 | 0.040381 |
| N5 | 0.061705 | +0.736 | 0.3023 | 0.024287 |
| N6 | 0.126218 | +0.015 | 0.0396 | 0.040547 |
| N7 | 0.166538 | -0.023 | 0.1945 | 0.116669 |
| N8 | 0.125977 | +0.022 | 0.0416 | 0.040467 |
| constant_0.5 | 0.125000 | undefined | 0.0000 | 0.037335 |
| visible_mean | 0.125000 | undefined | 0.0000 | 0.037335 |
| boundary_mean | 0.125023 | undefined | 0.0000 | 0.037335 |
| harmonic | 0.119586 | +0.272 | 0.1333 | 0.030136 |

## Interpretation

### Gap MSE and spatial fidelity are not interchangeable

- **Gradient:** N1 (MSE ≈ 0.000020, r ≈ 0.999) and N3 (0.000121, r ≈ 0.991) recover the spatial trend well. N5 also retains strong spatial alignment (r ≈ 0.928), whereas N7 (shared 1×1 without upsampling) has poor MSE and near-zero correlation. Harmonic interpolation is essentially exact for a linear gradient.
- **Blurred noise:** N1 (MSE ≈ 0.001071, r ≈ 0.961), N3 (0.002235, r ≈ 0.930) and N5 (0.003515, r ≈ 0.868) show meaningful target alignment. Harmonic interpolation (MSE ≈ 0.001072, r ≈ 0.952) is virtually tied with N1 on MSE. N8 has lower MSE than N6 (0.021645 vs 0.046970), yet both correlations are near zero; N8's within-gap prediction SD is smaller (0.1066 vs 0.1943), compatible with a low-variance rather than texture-matching advantage. Constant 0.5 (MSE ≈ 0.010777), global visible mean (0.008856), and boundary-ring mean (0.007265) all beat N8 on gap MSE.
- **Stripes:** N1 (MSE ≈ 0.047837, r ≈ 0.805) and N5 (0.061705, r ≈ 0.736) outperform harmonic interpolation (MSE ≈ 0.119586, r ≈ 0.272), consistent with an architectural advantage for spatially periodic patterns. However, N1's across-seed variability is substantial and N5 can be more stable. Unshared networks have MSE around 0.126 and essentially zero correlation, close to the 0.5-fill MSE of 0.125.

### Why spatial correlation matters

Gap MSE mixes bias, prediction variance, and spatial alignment. A prediction near the target's overall gray level can obtain a modest MSE even without reproducing the target's pixel-by-pixel structure. Pearson correlation isolates linear alignment of the spatial fluctuations (but ignores absolute scale and offset), so it should be reported alongside MSE, not instead of it. Finite-difference gradient error offers another complementary, scale-sensitive measure; neither correlation nor gradient error alone establishes perceptual reconstruction quality.

### Scope and caveats

1. The five model seeds vary initialization/noise, but **the blurred-noise target image is fixed at seed 0**. These results do not establish generalization to other target realizations.
2. The same central gap and image content are used across seeds; model seed observations are not independent image samples.
3. Shared and unshared networks use different checkpoint-selection rules. This is necessary for the current experimental design but complicates attribution of cross-family performance differences.
4. Correlation is evaluated only within a 16×16 gap and may be sensitive to spatial smoothness and pixel dependence; no inferential confidence interval or p-value is claimed for these correlations.
5. The harmonic baseline uses true pixels surrounding the gap and is an appropriate deterministic interpolation comparator; unlike the network training protocol, it does not exclude the held-out visible subset. A strictly matched-information baseline would need to exclude those held-out pixels as well.
6. The target-aligned correlation and gap-gradient diagnostics are descriptive. They do not prove that a network has learned the generating rule or a human-like structural understanding.

## Recommended follow-up

Prioritize replication across several *image-generation seeds* for blurred noise, rather than only additional model seeds. Compare matched-information harmonic interpolation, evaluate the effect of gap size, and inspect Fourier/phase alignment for stripes. Retain per-seed plots to avoid concealing instability behind averages. These follow-ups require additional experiments or baseline re-evaluation; they were not performed here.

## Reproducibility and deliverables

The analysis was computed from the archived selected predictions and the repository's deterministic image-generation and gap-mask definitions. The detailed analysis was also exported separately as `seed_level_target_aligned.csv`, `summary_target_aligned.csv`, `full_target_aligned_analysis.xlsx`, `mse_vs_correlation.png`, and three per-image gap grids. Those auxiliary files are not included in this Markdown-only GitHub commit. No new training was performed.
