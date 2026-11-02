# Development Progress

Last updated: 2026-11-02

This file records implementation status and the current experimental plan. Pilot results are preliminary and must not be presented as final study findings.

## Completed implementation and validation

- Repository scaffold, synthetic image generation, central-gap and held-out masks, model builder, training loop, metrics, logging, and tests are implemented.
- Eight ablation networks are defined by pooling/upsampling, filter size, and spatial weight sharing.
- The unshared variants use locally connected internal layers and a locally connected final output layer. Shared variants use spatially shared convolutional layers.
- Checkpoint selection follows Design v3:
  - Shared networks: lowest held-out MSE checkpoint.
  - Unshared networks: final training step, because held-out pixels receive no location-specific training signal and held-out MSE is not a valid stopping signal.
- Predictions are generated in training mode.
- Experiment workflows are manual-only; ordinary pushes do not launch expensive sweeps.
- The experiment runner supports a `--seed` override for model initialization and fixed-noise input. Image generation and held-out mask remain fixed by the configuration.
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

## Current decisions and pending confirmation

- Tom is conducting a formal width/time-complexity test, with 16 channels and 2,000 steps currently considered computationally feasible on GitHub Actions.
- Confirm the final channel width and step budget with the team after the timing test.
- The proposed main sweep is 8 networks × 3 image families × 5 seeds (120 runs), using one consistent protocol.
- Keep the gap-width sweep (8–32 px), Network 5 depth-profile/snapshot follow-ups, and networks 9–10 / hybrid / Bayesian extensions separate from the main sweep unless the team explicitly reprioritises them.

## Next steps

1. Complete and review the width/time-complexity test.
2. Agree and record the final channel width, training steps, seed list, and run allocation.
3. Verify one shared and one unshared run end-to-end under the final protocol, including checkpoint selection and saved predictions.
4. Launch the 120-run main sweep using manual GitHub Actions workflows.
5. Aggregate per-seed results; report mean and standard deviation across seeds, inspect outliers and convergence, and compare gap MSE and prediction quality.
6. Analyse the results together before committing to follow-up experiments.

## Reporting cautions

- Separate pilot, preliminary diagnostic, and formal sweep results.
- For unshared networks, report final-step gap reconstruction. Do not use held-out MSE to select their checkpoint.
- Keep all settings fixed across ablations unless a planned factor is being tested.
- Record workflow run links, commit SHA, configuration, seed, runtime, and artifact location for each experiment batch.
