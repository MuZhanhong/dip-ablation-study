# November sweep: design

What was run and what each run saves. The claims and predictions are in PREDICTIONS.md;
how they are tested is in ANALYSIS_PLAN.md.

## Common settings

- Every network sees the same images and model seeds, so comparisons are paired.
- 5000 training steps for every run. The pilot needed about 5000; at 2000 the shared
  networks were far from converged.
- Image seed 0 is not used: it was unusually easy for the harmonic fill.
- Networks 1, 3, 5, 7 are the shared 2x2: upsampling (yes for 1 and 3, no for 5 and 7)
  x filter size (3x3 for 1 and 5, 1x1 for 3 and 7).

## Arms

| Arm | Workflow, job | Networks | Images | Model seeds | Runs | Answers |
| --- | --- | --- | --- | --- | --- | --- |
| Core, shared | sweep-core, shared | 1, 3, 5, 7 | blurred noise, l = 2, 3, 4, 6; image seeds 1-8 | 0, 1, 2 | 384 | Topics 2 and 3 |
| Core, unshared | sweep-core, unshared-null | 8 | blurred noise, l = 2, 3, 4; image seeds 1-8 | 0 | 24 | Topic 1 |
| Baselines | sweep-core, baselines | grey, harmonic | as core | none | 4 jobs | reference fills for every topic |
| Stripes | sweep-extras, stripes | 1, 3, 5, 7 | stripes, period 4, 8, 16 | 0, 1, 2 | 36 | Stripes arm (Topic 2 on single-frequency images) |
| Width | sweep-extras, width | 1, 3, 5 at 64 channels | l = 4; image seeds 1-4 | 0, 1, 2 | 36 | Wider-networks arm: are the 2x2 differences just from 16 channels being narrow? |
| Width, l = 2 | sweep-extras, width-l2 | 5 at 64 channels | l = 2; image seeds 1-4 | 0, 1, 2 | 12 | Wider-networks arm: does net 5's width follow l when it is wide? |
| Network 2 probe | sweep-extras, network-2 | 2 | l = 4; image seeds 1-4 | 0 | 4 | Topic 1: does useful information reach the gap without sharing? |
| Contrast | sweep-extras, contrast | 8 | l = 4, image seed 1, contrast 0.5, 1, 1.5, 2, 3 | 0, 1 | 10 | Topic 1: the unshared gap variance s_0^2 |

The l = 6 core runs were made in the test launch (run #2); l = 2, 3, 4 in runs #3-#5.

## What each run saves

All runs:
- metrics.csv: training, held-out and gap error every 50 steps.
- Predicted image every 250 steps, plus the final and selected predictions.
- fixed_noise.npy (the network's input), so saved weights can be rerun.

Shared networks (1, 3, 5, 7):

| File | When | Contents |
| --- | --- | --- |
| kernel_step_XXXXX.npy | Every 250 steps from step 0 | Plain tangent kernel along 8 lines |
| kernel_adam_step_XXXXX.npy | Every 250 steps from step 250 | Adam-weighted tangent kernel along the same lines. Adam has no memory at step 0, so there is none then |
| kernel_extras_step_XXXXX.npz | Same steps | For plain and Adam: Theta(q, q) for every pixel on each line (for correlation-style scaling), and the tie to one far visible pixel with its Theta(r, r) (for the floor) |
| Weights and Adam's memory | Every 500 steps | For full kernels and 2D maps, computed offline on a subset |

In the file names, XXXXX is the step number written with 5 digits, for example
kernel_step_00250.npy for step 250.

- The plain kernel shows whether the network itself has changed. The Adam-weighted kernel
  shows whether the kernel explains the step training actually takes.
- The kernel is saved at the same steps as the predictions, so changes in the ties can
  be lined up with changes in the fill.

Unshared networks (8, 2): no model files; final_features.npz (features entering the final
layers) for the linear probe.

Baselines: grey and harmonic fills of the same 8 images, filled from the training pixels
only, so the gap and held-out pixels are unknown to them exactly as to the networks
(Zhanong's suggestion). Their errors are compared with each network's per image.

## Where the kernel is measured

- 8 lines of 17 pixels (distances 0-16 px): 4 horizontal (row 32 from columns 8, 24, 40;
  row 16 from column 24) and the same 4 turned vertical.
- The line starting at the gap's first column runs across the gap.
- Far visible pixel (for the floor): the corner point 12 px in from the two edges
  farthest from the line's start, 34-47 px away. 12 px in so that net 5, which sees
  12 px each way, is not distorted by the border padding.
- Theta(q, q) varies 0.17x to 7x along a line (25x on stripes), which is why ties are
  scaled like a correlation (see ANALYSIS_PLAN.md).

## Offline computations (from saved weights)

| What | Runs | Cost |
| --- | --- | --- |
| Prior kernel K (100 untrained copies) | Nets 1, 3, 5, 7 at 16 and 64 channels | Minutes (done 17 Nov) |
| Full tangent kernel (all 4096 pixels) | Nets 1, 3, 5, 7; l = 2, 4, 6; image seeds 1-2; model seed 0; every saved step (24 runs x 11 steps) | About 6-7 h on a laptop |
| Full tangent kernel, wider networks | Nets 1, 3, 5 at 64 channels, steps 0 and 5000, 2 runs each | About 1 h |
| Stripes tie maps | All 36 stripes runs; plain and Adam at steps 0 (plain only), 2500, 5000; the gap's 256 rows plus far visible pixels | About 3 h, about 4 MB per step |

## Cost per run (laptop, 2 threads, 16 channels, 5000 steps)

| Network | Weights | Time |
| --- | --- | --- |
| 1, 3, 5, 7 | 0.01-0.03 M | 1-2 min |
| 8 | 17.9 M | 17 min |
| 2 | 30.0 M | 30 min |
| 6 | 118.6 M | 125 min, plus about 90 s per kernel measurement |

## Built-in widths of the prior kernel K (checked 17 Nov, before launch)

K is the covariance of the outputs of 100 untrained copies of a network. This is a
different object from the step-0 tangent kernel measured in the runs: K describes the
random starting output, the tangent kernel describes how training spreads a change.

Width where K's tie falls halfway from 1 to its floor (the floor, about 0.2 at every
distance, is each copy's whole output being a bit brighter or darker):

| Network | 16 channels | 64 channels | Matches l of about |
| --- | --- | --- | --- |
| 1 | 6.9 px | 6.9 px | 4.2 |
| 3 | 8.3 px | 8.3 px | 5.0 |
| 5 | 0.5 px | 0.5 px | no ties |
| 7 | 0.5 px | 0.5 px | no ties |

- The built-in scale is set by the architecture, not the channel width.
- l = 2, 3, 4 are all at or below nets 1 and 3's built-in scales, so l = 6 was added.
- Blurred input noise is a weak dial for net 5: input ties 6.7 px wide give output ties
  only 1.2 px wide (net 7, with no mixing at all, 1.9 px). So no input-blur runs.

## Why the contrast arm

Net 8's gap variance is larger (about 0.011) when the visible pixels sit close to 0.5,
and smaller (about 0.002) when they sit far from it. It switches between a typical
distance of 0.12 and 0.16 (10 Nov trials - likely cause, not tested in the sweep:
batch-norm statistics set by the visible pixels). Every core image sits at about 0.08,
on the larger side, so the arm moves one image across the switch:

| Contrast | 0.5 | 1 | 1.5 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| Mean distance from 0.5 | 0.040 | 0.079 | 0.119 | 0.158 | 0.226 |
| Pixels clipped at 0 or 1 | 0 | 0 | 0 | 0.8% | 11.3% |

Without it, net 8's error level (grey + s_0^2) is partly set by the images' contrast and
has to be reported as a confound. The arm uses only net 8, so it says nothing about the
shared networks.

## Linear probe (network 2, 4 runs; analysis parked)

Does useful information reach the gap in an unshared network with a route, and go unread
by the gap's untrained final layers, or does nothing arrive?
- (a) Fit a linear readout on visible pixels' features, test it on the gap (what a shared
  final layer would need).
- (b) Fit on half the gap pixels, test on the other half (is it there at all).
- Comparisons: net 8 (no route, the floor) and untrained net 2 (chance).
- (b) fails: nothing arrives. (b) works but (a) fails: it arrives in a form the gap's own
  layers cannot read.

## Not in the sweep

- Network 6 (unshared twin of net 5, about 2 h per run): parked.
- Harmonic on stripes: run experiments/run_baselines.py locally (seconds).
- Donor copies (networks 9, 10, copy all layers): need unshared weights, so run locally on
  2-3 trained network 2 runs.
- Steering net 5 (depth, filter size, other controls): after the draft.
- No-batch-norm control and input jitter: not included.
- Shi et al.'s Lipschitz cap: not used. Batch norm undoes it, and every network here uses
  batch norm.
