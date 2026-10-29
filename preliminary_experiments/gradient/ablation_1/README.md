# Preliminary Gradient Ablation Result

This directory contains metrics and prediction previews from an exploratory CPU run of Ablation 1 on the synthetic gradient image.

- Ablation: 1
- Training steps: 300
- Channels: 8
- Device: CPU
- Best checkpoint by held-out MSE: step 250
- Best held-out MSE: 0.0006337648374027821
- Status: preliminary trial result; not a formal experiment

These results are from a preliminary run and must not be treated as the final 32-channel, 3000-step experiment.

See [`metrics.csv`](metrics.csv) for per-step metrics. PNG previews are available below; the original `.npy` prediction arrays and `.pt` checkpoint were not uploaded to this repository.

## Prediction previews

| Step | Preview |
|---:|---|
| 10 | ![Prediction at step 10](images/prediction_step_00010.png) |
| 50 | ![Prediction at step 50](images/prediction_step_00050.png) |
| 100 | ![Prediction at step 100](images/prediction_step_00100.png) |
| 150 | ![Prediction at step 150](images/prediction_step_00150.png) |
| 200 | ![Prediction at step 200](images/prediction_step_00200.png) |
| 250 | ![Prediction at step 250](images/prediction_step_00250.png) |
| 300 | ![Prediction at step 300](images/prediction_step_00300.png) |

Best prediction: ![Best prediction](images/best_prediction.png)
