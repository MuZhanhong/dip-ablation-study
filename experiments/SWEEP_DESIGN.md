# November sweep: design

Every network sees the same images and model seeds, so comparisons are paired.
5000 steps throughout (the pilot needed about 5000; at 2000 the shared networks
were far from converged). Image seed 0 is not used: it was unusually easy for
harmonic fill.

## Arms and the threads they answer

| Arm | Workflow, job | Networks | Images | Model seeds | Runs | Threads |
| --- | --- | --- | --- | --- | --- | --- |
| Core, shared | sweep-core, shared | 1, 3, 5, 7 | blurred noise l = 2, 3, 4, 6; image seeds 1-8 | 0, 1, 2 | 384 | 2, 3, 4, 6, 7 |
| Core, unshared reference | sweep-core, unshared-null | 8 | l = 2, 3, 4; image seeds 1-8 | 0 | 24 | 1 (grey plus noise) |
| Stripes | sweep-extras, stripes | 1, 3, 5, 7 | stripes, period 4, 8, 16 | 0, 1, 2 | 36 | 2 |
| Width | sweep-extras, width | 1, 3, 5 at 64 channels | l = 4; image seeds 1-4 | 0, 1, 2 | 36 | 6 (spread), 4, routes |
| Width, l = 2 | sweep-extras, width-l2 | 5 at 64 channels | l = 2; image seeds 1-4 | 0, 1, 2 | 12 | 4 (does the width follow l in a wide network?) |
| Baselines | sweep-core, baselines | grey, harmonic | as core | none | 4 jobs | 7 |
| Network 2 probe | sweep-extras, network-2 | 2 | l = 4; image seeds 1-4 | 0 | 4 | 8 |
| Contrast | sweep-extras, contrast | 8 | l = 4, image seed 1, contrast 0.5, 1, 1.5, 2, 3 | 0, 1 | 10 | 9 (s_0^2) |

Threads: 1 unshared = grey plus noise; 2 sharing AND a route; 3 networks 1 and 3,
prior does most of the work; 4 network 5 learns its kernel; 5 network 5 against 6;
6 reliable edge, variable centre; 7 many images, exact expected errors;
8 does anything useful reach the gap in unshared networks; 9 contrast and the
unshared gap noise.

## What each run saves

- metrics.csv, predictions every 250 steps, selected/final prediction
- fixed_noise.npy (the input), so saved weights can be rerun
- shared networks: the tangent kernel along 8 lines every 250 steps, the same
  steps as the saved predictions (kernel_step_XXXXX.npy, about 3 s each), so
  changes in the ties can be lined up with changes in the fill; weights every
  500 steps (about 0.2 MB each), for full kernels and 2D maps computed offline
  on a subset of runs (about 90 s and 67 MB per full kernel, too much for every run)
- shared networks, from step 250 on: the Adam-weighted tangent kernel along the
  same lines (kernel_adam_step_XXXXX.npy). Same gradients, so almost no extra
  time. The plain kernel answers "has the network itself changed?"; the
  Adam-weighted one answers "does the kernel explain the actual fill?" (Adam
  has no memory at step 0, so there is no Adam kernel then)
- shared networks, at every kernel step: kernel_extras_step_XXXXX.npz (a few kB),
  plain and (after step 0) Adam: Theta(q, q) for every pixel q on each line, so a
  tie can be normalised like a correlation, Theta(p, q) / sqrt(Theta(p, p) Theta(q, q))
  (Theta(q, q) varies 0.17-7x along a line, 25x on stripes); and the tie from each
  line's start to one far visible reference pixel (the corner point 12 px in from the
  edges farthest from the start, 34-47 px away; 12 px in so that net 5, which sees
  12 px each way, is not distorted by the border padding there) with its Theta(r, r), to give the
  floor for curves that have not settled by 16 px. One extra gradient per line
  (about 5% more kernel time); the existing kernel files are unchanged
- grey and harmonic fills (job "baselines", seconds): same 8 images, filled
  from the training pixels only, so the gap and held-out pixels are unknown for
  them exactly as for the networks (Zhanong's suggestion). Their errors are
  compared per image and paired, like a network's
- unshared networks: no model files; final_features.npz (features entering the
  final layers) for the linear probe

## Cost (measured on a laptop, 2 threads, 16 channels, 5000 steps)

| Network | Weights | Time |
| --- | --- | --- |
| 1, 3, 5, 7 | 0.01-0.03 M | 1-2 min |
| 8 | 17.9 M | 17 min |
| 2 | 30.0 M | 30 min |
| 6 | 118.6 M | 125 min, plus about 90 s per kernel measurement |

GitHub runners may be slower: time one run of each network before launching an arm.

## Headline test for network 5 (idea 2): is its smoothness set by the image?

The core shared runs measure the tangent kernel of networks 1, 3, 5 and 7 every
250 steps, at l = 2, 3 and 4. No extra runs are needed.

The kernel is measured along 8 lines of 17 pixels: 4 horizontal (row 32 from
columns 8, 24, 40; row 16 from column 24) and the same 4 turned vertical. The
line starting on the gap's first column runs across the gap; look at it
separately as well. Vertical against horizontal checks direction: equal for
blurred noise, but on stripes a learned kernel should reach further along
the stripes than across them.

1. For each run, step and line, divide the 17 values by the distance-0 value.
2. Average over lines, images and seeds (96 curves per direction, network, step and l):
   the mean curve. Also record the root-mean-square, which shows ties that are
   present but random in sign (network 5 at step 0: mean about 0, RMS about 0.24).
3. Width = the distance where the mean curve falls to 0.5. The image's own
   correlation exp(-d^2 / 4 l^2) falls to 0.5 at d = 2 l sqrt(ln 2), about 1.67 l:
   3.3, 5.0 and 6.7 pixels at l = 2, 3 and 4.

Checks in the analysis:
- Adam: run the width test on both kernels. If only the Adam-weighted kernel
  tracks l, the optimiser is adapting to the image, not the network; if the
  plain kernel tracks l too, the network itself has learned.
- Batch norm floor: if curves do not fall to about 0 by distance 16, measure the
  width above that floor.
- The line that crosses the gap is reported separately as well as in the average.
- Cause, not just timing: on a subset, the fixed-kernel fill from the kernel at
  each saved step, compared with the actual fill at that step.
- Uncertainty: bootstrap over images for every width.

Predictions at step 5000:
- Shi et al. (no upsampling = a weaker version of the same bias): network 5
  narrower than network 1 at every l, and the same at every l.
- Ours (no upsampling = smoothness set by the image): network 5 widens during
  training, more for larger l, roughly tracking 1.67 l; network 1 stays near its
  built-in width.
- Both tracking l would mean even network 1 adapts in training.

## Idea 3: how much of the fill is built in, and how much is learned?

Three fills for each run, each scored by f_d in three bands (edge, middle, centre),
between harmonic (0) and the best possible fill (1):

| Fill | Kernel behind it | What it shows |
| --- | --- | --- |
| 1. Prior-kernel fill | K: covariance of outputs over 100 untrained copies | the built-in assumption alone, no training |
| 2. Frozen-kernel fill | tangent kernel at step 0, this network | training with no new ties |
| 3. Actual fill | tangent kernel changing through training (idea 2) | prior plus learning |

2 to 3 is what the network learned from this image. Step B repeats fill 2 with the
kernel frozen at steps 500, 1000, ..., 5000 to show when the useful ties appeared.

Offline, from saved weights:
- K for networks 1, 3, 5, 7 at 16 and 64 channels: minutes.
- Full tangent kernel (all 4096 pixels): nets 1, 3, 5, 7, l = 2, 4 and 6, image seeds
  1-2, model seed 0, every saved step (24 runs x 11 steps, about 6-7 h on a laptop).
  l = 6 is included because there the built-in scales of nets 1 and 3 are too narrow,
  the clearest test of "built in".
  Width arm: nets 1, 3, 5 at steps 0 and 5000 on 2 runs each (about 1 h).
- Stripes: every gap pixel's tie to every pixel of the image (the gap's 256 rows of
  the full kernel, plus a few far visible reference pixels), plain and Adam, at steps
  0 (plain only), 2500 and 5000, all 36 runs (about 3 h; about 4 MB per step). Each
  gap pixel gets a 64x64 map. Headline: its ties to visible pixels (the fill
  mechanism): do they run along the stripes? Ties to other gap pixels are context
  only. Picking rule, fixed before results: one far visible reference pixel, and one
  gap pixel each in ring 1 (edge), ring 4 (middle) and ring 8 (centre), each on a
  stripe's centre line.

Routes: networks 1, 3, 5, 7 form a 2x2 (upsampling x filter size). Prediction:
the upsampling route (3) carries a built-in smoothness scale (fill 1 explains it,
ties stay put, weaker on period-4 stripes); the 3x3 route (5) has no built-in ties
and learns them (fill 1 near grey, width grows with l). The width arm checks that
this contrast is not an effect of 16 channels being narrow.

## Built-in widths (checked before launch, 17 Nov)

Prior kernel K from 100 untrained copies, width where the tie falls halfway from
1 to the floor (the floor is each copy's whole output being a bit brighter or
darker, about 0.2 at every distance):

| Network | 16 channels | 64 channels | Matches l of about |
| --- | --- | --- | --- |
| 1 | 6.9 px | 6.9 px | 4.2 |
| 3 | 8.3 px | 8.3 px | 5.0 |
| 5 | 0.5 px | 0.5 px | no ties |
| 7 | 0.5 px | 0.5 px | no ties |

The built-in scale is set by the architecture, not the channel width. l = 2, 3, 4
are all at or below networks 1 and 3's scales, so l = 6 was added.

Blurred input noise is a weak dial for network 5: input ties 6.7 px wide give
output ties only 1.2 px wide (network 7, with no mixing at all, 1.9 px). Depth
wears correlation down. No input-blur runs.

## Network 5 and Shi et al. (offline, no new runs)

Question: is network 5's preference for coarse patterns built in (Shi et al.) or
picked up from the image (ours)?
- Shi's measure on our runs: from predictions every 250 steps, how much of each
  frequency band of the visible image has been fitted by each step, for networks
  1, 3, 5, 7. Reproduces their result in our setting.
- Frequency profile of the tangent kernel: Fourier transform of the kernel lines
  at each saved step (which frequencies training moves fastest). Shi: network 5
  already favours low frequencies at step 0, the same at every l. Ours: flat at
  step 0, low-pass appears in training, cutoff moves with l.
- Training time as a dial: does network 5's fill get rougher as training goes
  on (fill correlation width at each saved step)?
- Shi's Lipschitz cap does not apply: batch norm undoes it, and every network
  here uses batch norm.

Steering network 5 (depth, filter size, other controls): after the draft, once
these results are in.

## Not in the sweep

- Parked until after the draft: network 6 (unshared twin of 5, about 2 h per run).

## Contrast arm (s_0^2, the unshared gap noise)

Network 8's gap noise is loud (about 0.011) when visible pixels sit close to 0.5
and quiet (about 0.002) when they sit far from it, switching between a typical
distance of 0.12 and 0.16 (10 Nov trials; likely cause, not tested in the sweep:
batch-norm statistics set by the visible pixels). Every core image sits at about 0.08, all on the loud side,
so the arm moves one image across the switch:

| Contrast | 0.5 | 1 | 1.5 | 2 | 3 |
| --- | --- | --- | --- | --- | --- |
| Mean distance from 0.5 | 0.040 | 0.079 | 0.119 | 0.158 | 0.226 |
| Pixels clipped at 0 or 1 | 0 | 0 | 0 | 0.8% | 11.3% |

Without it, network 8's error level (grey + s_0^2) is partly set by our images'
contrast and has to be reported as a confound.

## Linear probe (network 2, 4 runs)

Does useful information reach the gap in an unshared network with a route, and
go unread by the gap's untrained final layers, or does nothing arrive?
- (a) fit a linear readout on visible pixels' features, test it on the gap
  (what a shared final layer would need);
- (b) fit on half the gap pixels, test on the other half (is it there at all).
Comparisons: network 8 (no route, the floor) and untrained network 2 (chance).
(b) fails: nothing arrives. (b) works but (a) fails: it arrives in a form the
gap's own layers cannot read.

- Exact expected errors for harmonic and best-possible fill: computed offline
  (the reference line; the per-image harmonic runs above are the paired comparison).
- Harmonic on stripes: run experiments/run_baselines.py locally (seconds).
- Donor copies (networks 9, 10, copy all layers): need unshared weights, so run
  locally on 2-3 trained network 2 runs.
- No-batch-norm control and input jitter: not included.
