# November sweep: analysis plan

How the predictions in PREDICTIONS.md are checked. Written 21 Nov 2026, before any
sweep results exist, and committed with the predictions. Later changes go under
"Changes after launch" at the end, dated, before any result is looked at.

Runs and saved files: see SWEEP_DESIGN.md.

## Stopping and which prediction is used

Every run trains 5000 steps; 5% of visible pixels are held out (error every 50
steps). Saved: the prediction every 250 steps, the final (step 5000) and the selected
(lowest held-out error) predictions. Both are reported for every run; which one is
the main result is decided after the results are in, and the report says so and gives
the reason. Unshared networks (8, 2) use the final prediction: their held-out pixels
get no training signal, so the held-out choice means nothing there.

## Measurements

Saved during the runs (shared networks 1, 3, 5, 7 unless stated):

| Measurement | When | Used for |
| --- | --- | --- |
| Plain kernel along 8 lines of 17 px (4 horizontal, 4 vertical) | Every 250 steps from 0 (width arm: every 500) | Widths, timing |
| Adam kernel, same lines | Every 250 steps from 250 | Widths of the step actually taken |
| Theta(q, q) for every q on each line, plain and Adam | Same steps | Correlation-style scaling |
| Tie to one far visible reference pixel per line (12 px in from two edges, 34-47 px away) and its Theta(r, r) | Same steps | The floor each curve settles to |
| Predicted image | Every 250 steps, final, selected | f_d by ring, gap-only output width, Shi's frequency measure |
| Training, held-out and gap error | Every 50 steps | Stopping, selection, error curves, stripes fitting speed |
| Weights and Adam's memory | Every 500 steps | Offline full kernels, fills (2) and step B |
| Features entering the final layers (network 2 only) | End | Linear probe (parked) |

Computed offline (after the runs, from saved files or from scratch):
- Prior kernel K (done 17 Nov) and fill (1) from it (not done yet).
- Full tangent kernels, every pixel pair, from saved weights (and Adam's memory) at
  steps 0, 500, ..., 5000, on a subset of 24 runs (nets 1, 3, 5, 7; l = 2, 4, 6; image
  seeds 1-2; model seed 0): for fill (2), step B and the floor check.
- Stripes tie maps (each gap pixel's tie to every pixel, steps 0, 2500, 5000).
- Grey and harmonic fills for the stripes and the contrast arm (core images get theirs
  from the sweep's baselines job).
- The best possible fill on each image (the conditional mean of the gap given the
  training pixels, from the images' known covariance): the top of the f_d scale.
- Exact expected errors of harmonic and best fill, averaged over all possible images:
  a smooth reference line (f_d itself uses each image's own fills).

## Kernel widths

- Ties are scaled three ways and all three are reported: correlation-style
  Theta(p, q) / sqrt(Theta(p, p) Theta(q, q)) (planned main one: between -1 and 1,
  symmetric); Theta(p, q) / Theta(q, q) (p's move as a fraction of q's own move when
  training pushes q); Theta(p, q) / Theta(p, p) (the original plan). Theta(q, q)
  varies 0.17-7x along a line, so the last one mixes the tie with q's own size. Which
  is the headline is decided after the runs, with the reason stated.
- Floor = the scaled tie to the far reference pixel (each network's own, every run and
  step; the floor is the part of a tie that does not depend on distance, e.g. a shared
  output bias moves every pixel equally). Width = distance where the mean curve falls
  halfway from 1 to the floor. Also reported: the tie at the end of each line (d = 16),
  which equals the floor only where the curve has settled. Check on the offline
  full-kernel subset: far-pixel floor against the average tie to every pixel more than
  24 px away; if they disagree, the full-kernel floor is used for that network.
- Mean curve: average the 8 lines point by point (one curve per run), then the seeds
  (one curve per image), then the images. The root-mean-square is reported alongside
  (shows ties that exist but have random signs).
- The line that crosses the gap is reported separately as well as in the average;
  horizontal and vertical lines are compared (blurred noise has no preferred
  direction: a difference in every network = a bug; only in net 5 late in training =
  possibly real, checked against each image's own chance direction).
- Widths are read at equal steps and at equal training error.
- Uncertainty: bootstrap over images (resample the 8 images with replacement, 1000
  times) for every width and f_d.

## Topic 2: slope and intercept

For each net and each image seed (images are paired across l: the same white noise,
blurred by different amounts), fit width at step 5000 = a + beta x 1.67 l over
l = 2, 3, 4, 6. beta and a are compared between nets image by image.

Why the intercept: with no built-in scale the width is proportional to l, so a = 0
even if beta < 1; a built-in width w0 with partial acquisition gives
a = (1 - beta) w0 > 0. Conditions:
- also fitted on l = 2, 3, 4 only, and widths checked to have stopped growing by step
  5000 (net 5 still widening at l = 6 would make a look positive);
- "a = 0" is claimed only if a's 95% interval lies within -1 to +1 px; a wider interval
  is inconclusive;
- the fit is repeated with each image's own measured width in place of 1.67 l.

## Gap-only output width

Output correlation width measured inside the 16x16 gap only (the whole image is 94%
visible pixels, so any network that fits them copies the image's width; the
whole-image width is a fit check only). Compared with the same width, measured the same
way on the same window, for the true image and for the best possible fill (not with
1.67 l: the best fill is smoother than the truth deep in the gap, and a 16 px window
reads low at large l).

## Topic 3: fills

Notation: V = the training pixels (visible minus held-out), y_V = their true values,
f0 = the network's output before training. For a tangent kernel, the kernel fill is
exactly where plain gradient descent would end up if the kernel never changed.

| Fill | Kernel | Gap values | Computed from | Status |
| --- | --- | --- | --- | --- |
| (1) Prior | K, averaged by distance | mean + K(gap, V) K(V, V)^-1 (y_V - mean) | Untrained copies only | Not done |
| (2) Frozen | Plain tangent kernel at step 0 | f0(gap) + Theta0(gap, V) Theta0(V, V)^-1 (y_V - f0(V)) | Step-0 weights and input noise (24-run subset) | After the runs |
| Step B | Plain or Adam kernel at step t = 500, ..., 5000 | As (2), with Theta_t | Weights (and Adam's memory) at step t | After the runs |
| (2a) Optional | Theta0 with the first Adam step's reweighting | As (2) | Step-0 weights, noise, image | If time |
| (3) Actual | Changing through training | The network's own prediction | Saved predictions | With the runs |

A small stabilising term is added before each inverse (size fixed before results are
opened). All fills are scored by f_d in every ring.

## Topic 1 and the contrast arm

Measure: net 8 (or net 2) gap error minus grey gap error on the same image, in every
ring (not f_d: grey's f_d is strongly negative and changes with l). The predicted
numbers come from 50-step trial runs on 10 Nov at the same contrast; "about" = within
a factor of 2. Contrast arm: grey's error grows like c^2, so it is recomputed at each
contrast (computed offline with change_contrast); at contrast 3 about 11% of pixels
clip and grey uses the same clipped image.

## Stripes

- Fitting speed: each stripe image is almost one frequency, so the training error
  curve is that frequency's fitting curve. Measure: steps until training error falls
  below 10% of the stripes' own variance (0.0125). If every network gets there within
  50 steps, the first 250 steps are recomputed offline from the saved step-0 weights
  (training is deterministic given the seed, noise and image).
- Division of labour: speed separates "built-in low-pass" from "no built-in low-pass";
  the tie maps and the gap fill separate "acquired" from "flat" (a flat kernel cannot
  continue the stripes into the gap).
- Tie maps: picking rule fixed now: one far visible reference pixel, and one gap pixel
  each in ring 1, ring 4 and ring 8, each on a stripe's centre line. Headline: ties
  from gap pixels to visible pixels.
- One image per period: uncertainty rests on the 3 model seeds; supporting evidence.
- Stripes have 3.5x the contrast of blurred noise (sd 0.35 vs 0.1): compared across
  periods and networks, not with blurred noise.

## Shi et al. checks

- Band test: on a 64 px image only radial frequency bins 1-3 (periods 64, 32, 21 px)
  have usable power at both l = 2 and l = 6 (power falls like exp(-4 pi^2 l^2 f^2)); the
  test uses bins 2 and 3. With a fixed kernel, the fraction of a band fitted by a given
  step does not depend on how strong the band is.
- Shi's frequency-band measure: their exact formula checked before coding; two
  versions: (i) against the training pixels, (iii) inside the gap only, each band
  scored against the best fill's score in that band. (ii) whole clean image dropped:
  the gap is 6% of pixels, so it nearly repeats (i).
- The plain step-0 kernel is the reference for "built in": Adam depends on the image
  from its first step.

## Lucier's room (local, after exams)

Whole image, no gap (needs a no-gap mode: the training code currently fails on an
empty gap). A fresh network is fitted for N steps, its output becomes the next target,
10 rounds. Nets 1, 3, 5, 7; l = 2 and 6; image seeds 1-2; model seeds 0-1. N is set per
network from a short trial, before any drift is looked at, so that round 1 reaches the
same, clearly imperfect training error. Measured each round: correlation width and
overall amplitude (standard deviation). No end point is predicted (repeated imperfect
refitting keeps smoothing: Mobahi et al. 2020), so rates are compared.

## Still to be fixed after launch, before any results are opened

Each will be added under "Changes after launch", dated, before any result is looked at:
- Stripes: a number for "ties run along the stripes" (proposal: tie along the stripe
  minus tie across it, both at half a period); correct ties also appear at the same
  phase one period across.
- Fills (1) and (2): K averaged by distance into a stationary kernel first (K from 100
  copies has rank at most 99); the size of the stabilising term.
- Statistics: lines and seeds averaged within each image before bootstrapping; interval
  type (with only 8 images, t-based or labelled approximate); at most about 6 named
  headline tests; a rule for widths that never fall to halfway (capped at 16 px).
- Exact definitions: Shi's measure on masked images and inside the gap; the training
  error levels for equal-error comparisons (chosen from the pilot's curves).
- Lines that end within 8 px of an edge (4 of 8): reported separately or not.

## Changes after launch

(none yet)
