# Local boundary exploration and seed validation

Date: 16 November 2026. Contributor: Yudi.

## Scope

80 unique local CUDA training runs: 40 gap-width/blur-scale exploration runs; 16 additional N1/N3 model-seed runs; 16 independent-image runs; and eight N5 model-seed repeats. Archived seed-zero references are reused in validation tables and are not counted as new training runs.

This contribution uses the existing project model, training loop and image/mask definitions. The source snapshot matches the current repository's algorithm files; the repository's newer single-run CLI is separate from the independent batch runner used here.

## Findings

- At blur scales 4 and 8, N1 beats information-matched harmonic interpolation on all five tested image seeds, with model seed fixed at zero. This does not establish superiority at every blur scale or initialization.
- N3's stripe gap-16 advantage over N1 occurs only in original model seed zero; all four additional model seeds reverse that ordering. At gap 24, N1 has lower MSE in all five seeds.
- N3's blur-8 advantage over N1 occurs only in original image seed zero; all four additional target images reverse that ordering.
- N5's gap-32 advantage over gap 24 also occurs only in original model seed zero. Five-seed gap MSE is 0.081814 +/- 0.022224 for gap 24 and 0.097111 +/- 0.028129 for gap 32 (sample SD).
- N5's depth-eight ring has higher MSE and lower correlation than the outermost ring in every seed, at both gap widths. Interior correlation is not uniformly negative. Full ring-wise values are retained.
- N1's gap-24 MSE is lower than gap-16 MSE in three of five model seeds, so no universal monotonic gap-width relation is established.

## Protocol

64x64 grayscale, 16 channels, four levels, 2,000 optimization steps, Adam lr=0.01, and logs every 50 steps. Fixed Gaussian input noise is scaled by 0.1; no skips or input jitter. Shared networks use minimum recorded held-out MSE; unshared N2 uses the final step. Gap error never selects a checkpoint. Predictions remain in training mode. SSIM uses a fixed 3x3 window, unlike earlier 7x7 scores.

Model seeds change both initialization and input noise. Image seeds change the target while fixing model seed zero. These are separate validations, not a crossed image-seed x model-seed design. Harmonic hides both gap and held-out pixels and scores only the original gap.

## Validation and limitations

All 80 target/mask/config and metric records were audited. All 160 selected/final predictions were reconstructed from the saved weights in the full local record, with maximum absolute difference zero. The compact archive excludes model weights and intermediate snapshots, which remain in the full local record. It retains targets, masks, selected/final predictions, metrics, actual configurations, environment, source fingerprints and completion markers.

Follow-up conditions were chosen after the initial exploration; original seed zero is therefore not an independent validation observation. New-only summaries are provided. Means and sample SDs are descriptive, not confidence intervals. Gap pixels are not independent experimental repeats. Changing gap size changes both training visibility and held-out sampled positions. The target normalization follows the current sample-standard-deviation generator; no exact GP-optimal baseline is claimed.

These are local CUDA results. They are not pooled with, or presented as reproductions of, the team's GitHub-hosted CPU results. The source provenance and environment are recorded.

## Files and reproduction

- `tables/all_runs.csv`: 80 unique records, with all seeds preserved.
- `tables/validation_summary.csv`, `tables/paired_summary.csv`, and `tables/new_only_summary.csv`: N1/N3 validation and new-only summaries.
- `tables/n5_summary.csv`, `tables/n5_depth_summary.csv`: N5 seed and depth summaries.
- `images/`: curves, paired differences and gap reconstructions.
- `experiment-records.zip`: compact raw records, frozen plans, source snapshot and original execution/analysis scripts.
- `requirements-lock.txt`: recorded local dependencies.

To recompute the record audits and seed analyses from an empty output directory:

```bash
python experiments/local_validation/recompute_results.py --archive results/local-boundary-validation-2026-11-16/experiment-records.zip --out /path/to/new-analysis
```

The archive retains original local absolute paths as provenance; re-analysis uses relative archived source/record locations. No training or remote workflow is started by the recomputation command.
