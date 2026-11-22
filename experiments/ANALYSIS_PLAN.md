# November sweep: analysis plan

How the predictions in PREDICTIONS.md are checked. Written 21-22 Nov 2026, before any
sweep results exist, and committed with the predictions. Later changes are added at the end, dated, before any
result they affect is looked at.

Runs and saved files: see SWEEP_DESIGN.md.

## When training stops, and which prediction we use

- Every run trains for 5000 steps.
- 5% of the visible pixels are held out from training, and their error is recorded
  every 50 steps.
- Two predictions are kept for every run: the final one (step 5000) and the selected
  one (the step with the lowest held-out error). Both are reported. Which one is the
  main result is decided after the results are in, and the report says which and why.
- Unshared networks (nets 8 and 2) always use the final prediction. Their held-out
  pixels get no training signal, so choosing by held-out error means nothing for them.

## What is measured

### Saved during the runs

These are for the shared networks (1, 3, 5, 7) unless stated otherwise.

| What | When | Used for |
| --- | --- | --- |
| Plain kernel along 8 lines of 17 pixels (4 horizontal, 4 vertical) | Every 250 steps from step 0 (64-channel arm: every 500) | Widths, and when they change |
| Adam kernel along the same lines | Every 250 steps from step 250 | Widths of the step Adam actually takes |
| Theta(q, q) for every pixel q on each line, plain and Adam | Same steps | Correlation-style scaling |
| Tie to one far visible pixel per line (12 px in from two edges, 34-47 px away), and that pixel's own Theta(r, r) | Same steps | The floor each curve settles to |
| Predicted image | Every 250 steps, plus final and selected | f_d by ring, width inside the gap, Shi's frequency measure |
| Training, held-out and gap error | Every 50 steps | Stopping, selection, error curves, stripes fitting speed |
| Weights and Adam's memory | Every 500 steps | Full kernels offline, fill (2) and step B |
| Features entering the final layers (nets 8 and 2 only) | End of training | Linear probe (parked) |

### Computed after the runs

These come from the saved files, or are computed from scratch.

| What | Details | Used for | Status |
| --- | --- | --- | --- |
| Prior kernel K | 100 untrained copies per network | Fill (1) | K done 17 Nov; fill (1) not done |
| Full tangent kernels (every pixel pair) | From saved weights and Adam's memory at steps 0, 500, ..., 5000, on 24 runs: nets 1, 3, 5, 7; l = 2, 4, 6; image seeds 1-2; model seed 0 | Fill (2), step B | After the runs |
| Stripes tie maps | Each chosen gap pixel's tie to every pixel, at steps 0, 2500 and 5000 | Whether ties run along the stripes | After the runs |
| Grey and harmonic fills | For the stripes and the contrast arm (the core images get theirs from the sweep's baselines job) | Grey: the Topic 1 measure. Harmonic: 0 on the f_d scale | After the runs |
| Best possible fill on each image | The conditional mean of the gap given the training pixels, from the images' known covariance | Top of the f_d scale | After the runs |
| Expected errors of harmonic and best fill, averaged over all possible images | Exact, from the covariance | A smooth reference line (f_d itself uses each image's own fills) | After the runs |

## Kernel widths

### Scaling the ties

Raw ties cannot be compared between pixels, because some pixels move more than others.
So each tie is scaled. All three scalings are reported:

| Scaling | Formula | What it means |
| --- | --- | --- |
| Correlation-style (planned main one) | Theta(p, q) / sqrt(Theta(p, p) Theta(q, q)) | Between -1 and 1, and the same either way round |
| By q's own move | Theta(p, q) / Theta(q, q) | How far p moves, as a fraction of how far q moves, when training pushes q |
| By p's own move | Theta(p, q) / Theta(p, p) | The original plan. Theta(q, q) varies from 0.17x to 7x along a line, so this one mixes the tie with q's own size |

Which one is the headline is decided after the runs, and the report says why.

Width and floor are as defined in PREDICTIONS.md; the floor is measured with the far
visible pixel in every run and step.

### Averaging and comparisons

- Average curve: average the 8 lines (one curve per run), then the 3 seeds (one per
  image), then the 8 images. The root-mean-square curve is reported too, because it
  shows ties that exist but have random signs.
- Gap line: the line that crosses the gap is also reported on its own.
- Horizontal vs vertical: a difference in every network means a bug; a difference only
  in net 5, late in training, may be real.
- Like with like: widths are compared at the same step and at the same training error.
- Uncertainty: bootstrap over the 8 images (resample them with replacement, 1000
  times), for every width and f_d.

## Topic 2: slope and intercept

For each network and each image seed, fit width at step 5000 = a + beta x 1.67 l,
across l = 2, 3, 4 and 6. The images are paired across l (same white noise, blurred by
different amounts), so beta and a are compared between networks image by image.

### Why 1.67 l

1.67 l is the image's own width by the same halfway rule: blurred noise has correlation
exp(-d^2 / (4 l^2)), which falls to 1/2 at d = 2 sqrt(ln 2) l = 1.67 l. So a network
that fully copies the image's scale gives beta = 1 and a = 0.

### Conditions

- Also fit on l = 2, 3, 4 only, and check that widths have stopped growing by step
  5000. Net 5 still widening at l = 6 would make a look positive.
- "a = 0" is claimed only if a's 95% interval lies within -1 to +1 px. A wider
  interval is inconclusive.
- Repeat the fit using each image's own measured width in place of 1.67 l.

## Width inside the gap

### What we want to know

What scale the network uses where it has no data. Only the gap can tell us this.
Everywhere else, any network that fits the visible pixels just copies the image's
width.

### Measure

How quickly the network's output changes across the gap: the correlation width of the
output, using only pixels inside the 16x16 gap.

### What it is compared with

The same measurement, made in the same window, on:
- the true image;
- the best possible fill.

## Topic 3: fills

### Notation

- V: the training pixels (visible pixels minus held-out ones).
- y_V: their true values.
- f0: the network's output before training.

For a tangent kernel, the kernel fill is exactly where plain gradient descent would end
up if the kernel never changed.

### The fills

| Fill | Kernel | Gap values | Status |
| --- | --- | --- | --- |
| (1) Prior | K, averaged by distance | mean + K(gap, V) K(V, V)^-1 (y_V - mean) | Not done |
| (2) Frozen | Plain tangent kernel at step 0 | f0(gap) + Theta0(gap, V) Theta0(V, V)^-1 (y_V - f0(V)) | After the runs (24-run subset) |
| Step B | Plain or Adam kernel at step t = 500, ..., 5000 | As (2), with Theta_t | After the runs |
| (3) Actual | - | The network's own prediction | With the runs |

A small stabilising term is added before each inverse; its size is decided before the
results are opened. All fills are scored by f_d in every ring.

## Topic 1 and the contrast arm

- Three numbers per image and ring: the mean of D (should be about 0, the 0.5
  result), Var(D) = s_0^2, and Cov(D, e) (should be about 0). Also gap error minus
  grey error, which should equal Var(D) - 2 Cov(D, e); this checks the arithmetic.
- Raw error, not f_d: grey's f_d is strongly negative and changes with l, so it would
  hide the comparison.
- Grey at each contrast: grey's error grows like c^2 (c = contrast), so it is
  recomputed at each contrast, on the same image.
- Clipping: at contrast 3, about 11% of pixels clip at 0 or 1. Grey and e are computed
  against the same clipped image, so that the difference is still
  s_0^2 - 2 Cov(D, e).

## Stripes

- Fitting speed: the number of steps until the training error falls below 10% of the
  stripes' own variance (0.0125). If every network gets there within 50 steps, the
  first 250 steps are recomputed from the saved step-0 weights. Training is
  deterministic given the seed, the noise and the image.
- Tie maps: the pixels are chosen now. One far visible pixel, plus one gap pixel each
  in ring 1, ring 4 and ring 8, each on a stripe's centre line. The headline is the
  ties from gap pixels to visible pixels.
- Contrast: stripes have 3.5x the contrast of blurred noise (sd 0.35 vs 0.1), so they
  are compared across periods and networks, not with blurred noise.

## Shi et al. checks

- Band test: on a 64 px image, only frequency bins 1-3 (periods 64, 32 and 21 px) have
  usable power at both l = 2 and l = 6, because power falls like
  exp(-4 pi^2 l^2 f^2). The test uses bins 2 and 3. Bin 1 is left out of this test: it
  has only about 4 independent components, and it is probably fitted before the first
  saved image (step 250), leaving no speed to compare. It is still included in Shi's
  own measure (reproduction).
- Shi's measure: their exact formula is checked against the paper before coding. There
  are two versions: one against the training pixels, and one inside the gap only,
  where each band is scored against the best fill's score in that band.
- Reference for "built in": the plain kernel at step 0, because the Adam kernel depends
  on the image from its first step.

## Lucier's room (local, after exams)

- Whole image, no gap. The training code currently fails with an empty gap, so it
  needs a no-gap mode first.
- Runs: nets 1, 3, 5, 7; l = 2 and 6; image seeds 1-2; model seeds 0-1; 10 rounds each.
- Steps per round (N): set for each network from a short trial, before any drift is
  looked at.

## Still to decide (after launch, before looking at the results they affect)

Each will be added at the end, dated, before any result it affects is looked at.

- Stripes: a number for "ties run along the stripes". Proposal: the tie along the
  stripe minus the tie across it, both at half a period. Correct ties also appear at
  the same phase one period across.
- Fill (1): K is first averaged by distance into a kernel that is the same everywhere,
  because K from 100 copies has rank at most 99.
- Fills (1) and (2): the size of the stabilising term.
- Statistics: the interval type (with only 8 images, t-based or labelled approximate);
  at most about 6 named headline tests; a rule for widths that never fall to halfway
  (capped at 16 px).
- Exact definitions: Shi's measure on masked images and inside the gap; the
  training-error levels for equal-error comparisons, chosen from the pilot's curves.
- Lines near an edge: 4 of the 8 lines end within 8 px of an edge; whether to report
  them separately.
- Possible extra fill (2a): the step-0 kernel with the first Adam step's reweighting,
  added only if there is time.
