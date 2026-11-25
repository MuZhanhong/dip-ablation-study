# November sweep: analysis plan

How the predictions in PREDICTIONS.md are checked.

Runs and saved files: see SWEEP_DESIGN.md.

## Which prediction is used

- Every run trains for 5000 steps.
- 5% of the visible pixels are held out from training; their error is recorded every
  50 steps.
- Main result: the final prediction (step 5000), for every network. The images have no
  noise, so there is nothing to overfit.- Also reported: the selected prediction (the step with the lowest held-out error).
- Unshared networks (nets 8 and 2) always use the final prediction. Their held-out
  pixels get no training signal, so selecting by held-out error means nothing for them.

## What is measured

### Saved during the runs

These are for the shared networks (1, 3, 5, 7) unless stated otherwise.

| What | When | Used for |
| --- | --- | --- |
| Plain kernel along 8 lines of 17 pixels (4 horizontal, 4 vertical) | Every 250 steps from step 0 (64-channel arm: every 500) | Widths, and when they change |
| Adam kernel along the same lines | Every 250 steps from step 250 | Widths of the step Adam actually takes |
| Theta(q, q) for every pixel q on each line, plain and Adam | Same steps | Correlation-style scaling |
| Tie to one far visible pixel per line (12 px in from two edges, 34-47 px away), and that pixel's own Theta(r, r) | Same steps | The floor each curve settles to |
| Predicted image | Every 250 steps, plus final and selected | Width inside the gap, f_d by ring, Shi's frequency measure |
| Training, held-out and gap error | Every 50 steps | Error curves, equal-error comparisons, stripes fitting speed |
| Weights and Adam's memory | Every 500 steps | Full kernels offline, fill (2) and step B |
| Features entering the final layers (nets 8 and 2 only) | End of training | Linear probe (parked) |

### Computed after the runs

| What | Details | Used for |
| --- | --- | --- |
| Prior kernel K | 100 untrained copies per network (done 17 Nov) | Fill (1) |
| Full tangent kernels (every pixel pair) | From saved weights and Adam's memory at steps 0, 500, ..., 5000, on 24 runs: nets 1, 3, 5, 7; l = 2, 4, 6; image seeds 1-2; model seed 0 | Fill (2), step B |
| Stripes tie maps | Each chosen gap pixel's tie to every pixel, at steps 0, 2500 and 5000 | Whether ties run along the stripes |
| Grey and harmonic fills | For the stripes and the contrast arm (the core images get theirs from the sweep's baselines job) | Grey: the Topic 1 measure. Harmonic: 0 on the f_d scale |
| Best possible fill on each image | The conditional mean of the gap given the training pixels, from the images' known covariance | Top of the f_d scale |
| Expected errors of harmonic and best fill over all possible images | Exact, from the covariance | A smooth reference line |

## Kernel widths

### Scaling the ties

Raw ties cannot be compared between pixels, because some pixels move more than others.

| Scaling | Formula | Role |
| --- | --- | --- |
| Correlation-style | Theta(p, q) / sqrt(Theta(p, p) Theta(q, q)) | Headline. Between -1 and 1, the same either way round |
| By q's own move | Theta(p, q) / Theta(q, q) | Reported |
| By p's own move | Theta(p, q) / Theta(p, p) | Reported. Theta(q, q) varies 0.17x to 7x along a line, so this mixes the tie with q's own size |

### Width and floor

- Width and floor are as defined in PREDICTIONS.md. The floor is the tie to the far
  visible pixel, measured in every run and at every step.
- Width = the distance where the tie falls halfway from 1 to the floor, with linear
  interpolation between pixels.
- A curve that never falls to halfway by 16 px gets width 16 px. It is flagged, and the
  number of such curves is reported.
### Averaging

- Average the 8 lines (one curve per run), then the 3 seeds (one curve per image). The
  width is read from that curve.
- The root-mean-square curve is reported too, because it shows ties that exist but have
  random signs.
- The line that crosses the gap is also reported on its own.
- Lines near an edge (4 of the 8 end within 8 px of one) are kept in the average and
  also reported separately.
- Horizontal vs vertical: a difference in every network means a bug. A difference only
  in net 5, late in training, may be real.

## Statistics

- Intervals: 95% percentile bootstrap over the 8 images (1000 resamples, rng seed 0),
  labelled approximate, since there are only 8 images.- Differences between networks: paired bootstrap, resampling the same images for both
  networks.- Six named headline tests. Everything else is descriptive.
  1. Topic 1: net 8's Var(D) is near 0.011, and Cov(D, e) is near 0.
  2. Topic 2: net 5's beta is above nets 1 and 3 (paired).
  3. Topic 2: net 5's a lies within -1 to +1 px.
  4. Topic 3: the learned part of the fill is larger for net 5.
  5. Stripes: net 5's ties run along the stripes at period 4.
  6. Shi: the band test, bins 2-3.

## Topic 2: does the width follow the image's scale?

### Main fit

For each network and image seed, fit width at step 5000 = a + beta x 1.67 l across
l = 2, 3, 4 and 6. Report beta and a as the mean over the 8 images, with intervals.

The images are paired across l (same white noise, blurred by different amounts), so
beta and a are compared between networks image by image.

1.67 l is the image's own width by the same halfway rule. Blurred noise has correlation
exp(-d^2 / (4 l^2)), which falls to 1/2 at d = 2 sqrt(ln 2) l = 1.67 l. A network that
fully copies the image's scale gives beta = 1 and a = 0.

### Conditions and checks

- Also fit on l = 2, 3, 4 only, and check whether widths are still growing at step 5000.
  Net 5 still widening at l = 6 would make a look positive.
- "a = 0" is claimed only if a's 95% interval lies within -1 to +1 px. A wider interval
  is inconclusive. This test is weak: a is an extrapolation to image width 0, outside
  the data (the smallest is 3.3 px).- Repeat the fit using each image's own measured width in place of 1.67 l. Measure the
  image's width the same way as the network's (same far pixel, halfway to the floor).- Report the floor at each l.- Report each image's residuals from its straight line. beta is described as an average
  slope over l = 2-6.- Median width over the saves at steps 4000-5000, as a check against one save landing on
  a spike in the Adam kernel.- Report the step-0 width at each l for every network. The untrained network has not
  seen the image, so this should not depend on l.- The l = 6 runs came from the test launch (run #2). Check their settings match runs
  #3-#5 before the analysis.
### Confound: does the width follow fit quality rather than scale?

Raising l also makes the image easier to fit.

- Equal training error: read the width at the first saved step where training error is
  at or below 10%, 1% and 0.1% of the image's variance, then refit beta. A level is used
  only if at least 90% of runs reach it.
- Within each l: the correlation across runs between final training error and width.- A check that changes difficulty without changing scale (for example contrast) would
  need new runs on nets 1, 3 and 5. The contrast arm used only net 8, so it cannot do
  this.
## Width inside the gap

What scale the network uses where it has no data. Only the gap can show this: anywhere
else, a network that fits the visible pixels simply copies the image's width.

- Measure: the correlation width of the output, using only pixels inside the 16x16 gap.
- Compared with the same measurement, in the same window, on the true image and on the
  best possible fill.

## Topic 3: fills

### Notation

- V: the training pixels (visible pixels minus held-out ones).
- y_V: their true values.
- f0: the network's output before training.

For a tangent kernel, the kernel fill is exactly where plain gradient descent would end
up if the kernel never changed.

### The fills

| Fill | Kernel | Gap values |
| --- | --- | --- |
| (1) Prior | K, first averaged by distance into a kernel that is the same everywhere (K from 100 copies has rank at most 99) | mean + K(gap, V) K(V, V)^-1 (y_V - mean) |
| (2) Frozen | Plain tangent kernel at step 0 | f0(gap) + Theta0(gap, V) Theta0(V, V)^-1 (y_V - f0(V)) |
| Step B | Plain or Adam kernel at step t = 500, ..., 5000 | As (2), with Theta_t |
| (3) Actual | - | The network's own prediction |

- Stabilising term before each inverse: 1e-6 times the mean diagonal of the kernel.
  Re-run at 1e-4 and 1e-8 to check the results do not change.
- All fills are scored by f_d in every ring.

## Topic 1 and the contrast arm

- Three numbers per image and ring: the mean of D (should be about 0, the 0.5 result),
  Var(D) = s_0^2, and Cov(D, e) (should be about 0). Also gap error minus grey error,
  which should equal Var(D) - 2 Cov(D, e); this checks the arithmetic.
- Raw error, not f_d: grey's f_d is strongly negative and changes with l, so it would
  hide the comparison.
- Grey at each contrast: grey's error grows like c^2 (c = contrast), so it is recomputed
  at each contrast, on the same image.
- Clipping: at contrast 3, about 11% of pixels clip at 0 or 1. Grey and e are computed
  against the same clipped image, so the difference is still s_0^2 - 2 Cov(D, e).

## Stripes

- Fitting speed: the number of steps until the training error falls below 10% of the
  stripes' own variance (0.0125). If every network gets there within 50 steps, the first
  250 steps are recomputed from the saved step-0 weights. Training is deterministic given
  the seed, the noise and the image.
- Tie maps: one far visible pixel, plus one gap pixel each in ring 1, ring 4 and ring 8,
  each on a stripe's centre line. The headline is the ties from gap pixels to visible
  pixels.
- "Ties run along the stripes": the tie along the stripe minus the tie across it, both
  at half a period. Correct ties also appear at the same phase one period across.
- Contrast: stripes have 3.5x the contrast of blurred noise (sd 0.35 vs 0.1), so they are
  compared across periods and networks, not with blurred noise.

## Shi et al. checks

- Band test: on a 64 px image, only frequency bins 1-3 (periods 64, 32 and 21 px) have
  usable power at both l = 2 and l = 6, because power falls like
  exp(-4 pi^2 l^2 f^2). The test uses bins 2 and 3. Bin 1 is left out of the test: it has
  only about 4 independent components, and it is probably fitted before the first saved
  image (step 250). It is still included in Shi's own measure.
- Shi's measure: the exact formula is checked against the paper before coding. Two
  versions: one against the training pixels, and one inside the gap only, where each
  band is scored against the best fill's score in that band.
- Reference for "built in": the plain kernel at step 0, because the Adam kernel depends
  on the image from its first step.

## Exploratory (chosen after looking, reported as such)
- Ties at 8 and 12 px.
- Plain vs Adam kernel.
- The upsampling x filter-size effect on beta.

## Lucier's room (local, after exams)

- Whole image, no gap. The training code currently fails with an empty gap, so it needs
  a no-gap mode first.
- Runs: nets 1, 3, 5, 7; l = 2 and 6; image seeds 1-2; model seeds 0-1; 10 rounds each.
- Steps per round (N): set for each network from a short trial, before any drift is
  looked at.

## Dropped or parked

- Fill (2a), the step-0 kernel with the first Adam step's reweighting: dropped.
- Linear probe on nets 8 and 2: parked.
