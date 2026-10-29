# CPU Smoke Test: Ablation 6, 4 Channels

## Scope

This is a short end-to-end smoke run for the locally reproduced current model/training logic, not a formal experiment and not a full repository test-suite result.

- Image: 64 × 64 synthetic horizontal gradient
- Missing region: centered 16 × 16 square
- Held-out validation: 5% of visible pixels, seed 0
- Ablation: 6 (no pooling/upsampling, 3 × 3 locally connected internal filters)
- Output layer: shared 1 × 1 Conv2d + sigmoid
- Channels: 4
- Steps: 10
- Device: CPU
- Seed: 0
- Parameters: 7,602,341

## Result

- Best held-out MSE: 0.089566 at step 1
- Step-10 training MSE: 0.040223
- Step-10 held-out MSE: 0.098155
- Step-10 gap MSE: 0.023463
- Runtime: about 3.45 seconds

Training MSE fell over these 10 steps, while held-out MSE rose after the first step. This tiny run is only a pipeline check; it is not evidence of final reconstruction quality. The best checkpoint should be selected using held-out MSE, not gap MSE.

## Limitations

- Only 10 steps and one synthetic image were used.
- This run does not validate the full eight-ablation experiment.
- The source repository could not be cloned into the execution environment, so this was an independent local reproduction of the checked-in architecture and masked-training logic.
- Run the repository's pytest suite and official runner in a network-enabled project environment before treating this as a verified repository run.
