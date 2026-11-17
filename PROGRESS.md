# Development Progress

Last updated: 2026-11-17

This file tracks the implementation and experiment runs. Pilot observations are kept separate from the formal results.

## Completed implementation and validation

- Repository scaffold, synthetic image generation, central-gap and held-out masks, model builder, training loop, metrics, logging, and tests are implemented.
- Eight ablation networks are defined by pooling/upsampling, filter size, and spatial weight sharing.
- The unshared variants use locally connected internal layers and a locally connected final output layer. Shared variants use spatially shared convolutional layers.
- Checkpoint selection follows Design v3:
  - Shared networks: lowest held-out MSE checkpoint.
  - Unshared networks: final training step, because held-out pixels receive no location-specific training signal and held-out MSE is not a valid stopping signal.
- Predictions are generated in training mode.
- Experiment workflows are manual-only; ordinary pushes do not launch expensive sweeps.
- The runner uses `--seed` for model initialization and fixed-noise input, and `--image-seed` for the synthetic target image. The held-out mask uses the configured mask seed.
- GitHub Actions tests and training smoke checks have passed.

## Pilot findings (preliminary)

The pilot tested networks 1–8 and exploratory networks 9–10 on gradient, blurred-noise, and stripe images at 8 channels, 600 steps, and one seed. The main leads were:

- Shared networks behaved broadly as expected; unshared networks did not reconstruct the gap reliably.
- Network 5 showed the strongest stripe continuation in the pilot, with useful correlation extending roughly 3–4 pixels into the gap from each edge.
- Network 1 performed well on smooth images, particularly blurred noise.
- Exploratory donor-filter networks 9 and 10 did not provide convincing evidence of useful pattern reconstruction without spatial weight sharing.
- The pilot exposed the invalid use of held-out early stopping for unshared networks; the evaluation protocol has since been corrected.

These observations are hypotheses to test with repeated runs, not final conclusions. See the pilot report and Design v3 for the full setup, tables, and caveats.

## Preliminary channel-width runs

Earlier Ablation 6 runs at 8 and 16 channels were diagnostic runs under an older code/evaluation state. Their held-out MSE should not be used to judge unshared-network gap reconstruction or as final evidence. Preserve their artifacts for traceability, but do not combine them with the corrected main sweeps.

## Main sweep and follow-up

- The 120-run main sweep (8 networks × 3 image families × 5 model seeds) completed at 16 channels and 2,000 steps. See `results/comprehensive-target-aligned-report.md`.
- The first independent blurred-noise image-seed test (image seed 1, model seed 0) completed successfully in GitHub Actions run 43706330133.
- A separate phase-two workflow covers image seeds 2 and 3 across networks 1, 3, 5, 6, and 8.
- Keep the gap-width sweep (8–32 px), Network 5 depth-profile/snapshot follow-ups, and networks 9–10 / hybrid / Bayesian extensions separate from the main sweep unless the team explicitly reprioritises them.

## Next steps

1. Complete image-seed generalization runs for seeds 2 and 3.
2. Compare gap MSE and spatial correlation across independent target images.
3. Check convergence before choosing which networks need longer training.
4. Add a harmonic baseline using only the pixels available to the networks.
5. Review the findings with the team before extending the experiment matrix.

## Reporting cautions

- Separate pilot, preliminary diagnostic, and formal sweep results.
- For unshared networks, report final-step gap reconstruction. Do not use held-out MSE to select their checkpoint.
- Keep all settings fixed across ablations unless a planned factor is being tested.
- Record workflow run links, commit SHA, configuration, seed, runtime, and artifact location for each experiment batch.
