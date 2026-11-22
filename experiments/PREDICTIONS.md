# November sweep: predictions

Written 20-21 Nov 2026, before any sweep results exist, and committed before launch.
Do not edit after launch: add any change under "Changes after launch", dated, with the
reason. How each prediction is checked: ANALYSIS_PLAN.md. Runs: SWEEP_DESIGN.md.

## Definitions

- Tie: Theta(p, q) = sum over weights w of g_p,w * g_q,w, with g_p,w = d(output at p)/dw:
  how strongly a training push on pixel q also moves pixel p. Plain kernel = this;
  Adam kernel = each weight's term divided by Adam's typical gradient size for that
  weight (the step actually taken). No Adam kernel at step 0.
- Prior kernel K: output covariance over 100 untrained copies (17 Nov: net 1 6.9 px,
  net 3 8.3 px, nets 5 and 7 no ties).
- Width: distance at which the scaled tie falls halfway from 1 to its far-away level
  (the floor, measured in every run).
- Image width: 1.67 l (3.3, 5.0, 6.7, 10 px at l = 2, 3, 4, 6).
- f_d in ring d (8 rings, edge to centre): 0 = as good as harmonic, 1 = best possible.

## Topic 1: sharing

Claim: without shared weights nothing trains the gap's own final layers, so the gap
comes out as grey (0.5) plus noise.

| Measure | Prediction |
| --- | --- |
| Net 8 gap error - grey gap error | About 0.011: positive in every ring, flat from edge to centre, the same at every image and l |
| Net 2 (4 runs), same measure | Positive on every image, about 0.04-0.05 |
| Contrast arm (net 8) | About 0.011 at contrast 0.5 and 1; 1.5 either; about 0.002 at 2 and 3 |

"About" = within a factor of 2 (numbers from 10 Nov trial runs).

Against: net 8 beating grey near the gap's edge, or its excess changing with l; net 2
clearly better than grey on most images. Contrast: no change with contrast (then no
confound), or a steady fall rather than a switch. Net 2 at grey plus noise does not
show that nothing reaches the gap (it could arrive unread). The contrast effect's
cause (batch-norm statistics) is a caveat, not tested.

## Topic 2: does network 5 acquire its scale?

Claim: net 5 (3x3, no upsampling) has no built-in scale and takes the image's during
training; nets 1 and 3 have built-in scales. Kernels are known to adapt to their target
in many networks, so the claim is about degree. Shi et al. call architecture "not
critical" (p. 7) and attribute the bias to the network; "built in" below is our
reading of that, which they do not state or test.

Fit width at step 5000 = a + beta x 1.67 l across l = 2, 3, 4, 6 (beta = how much the
width follows the image; a = a fixed built-in part).

| Net | Plain kernel, step 0 | Step 5000 (plain and Adam) | beta | a |
| --- | --- | --- | --- | --- |
| 1 | Local ties | Near its own step-0 width at every l | Near 0 | Near its own width |
| 3 | Local ties | Near its own step-0 width at every l | Near 0 | Near its own width |
| 5 | No consistent ties (random signs) | Grows with l towards 1.67 l; Adam at least as close as plain | Clearly above 0 and above nets 1, 3 | Within 1 px of 0 |
| 7 | No ties above the floor | No ties above the floor | - | - |

Also: net 5 widens gradually over training; horizontal = vertical widths. Gap-only
output width: nets 1 and 3 near their own width and stable; net 5 starts rough and
moves towards the best fill's gap width, not getting rougher late in training; net 7
rough throughout.

| Plain follows l | Adam follows l | Meaning |
| --- | --- | --- |
| Yes | Yes | Scale stored in the weights |
| No | Yes | Scale lives in Adam's memory |
| No | No | Net 5 does not acquire the scale |

Against: beta(net 5) near 0; beta(net 5) no larger than nets 1 and 3; net 5's a
clearly above 1 px (a small built-in scale); net 5's gap width falling late in training.

## Topic 3: built in vs learned

Claim: for nets 1 and 3 the built-in prior does most of the work; for net 5 most of
the fill is learned from the image. Fills: (1) from K, the architecture alone;
(2) from the frozen step-0 tangent kernel, training with ties that never change;
(3) the actual run. (2) to (3) = learned.

| Net | (1) Prior | (2) Frozen | (3) Actual |
| --- | --- | --- | --- |
| 1 | Decent near the edge | Close to (3) | Best near l = 4, worse at 2 and 6 |
| 3 | Same, slightly smoother | Close to (3) | Similar to net 1 |
| 5 | Poor (about a constant fill) | Poor | Clearly better, at every l |
| 7 | Poor | Poor | Poor |

Step B (kernel frozen at later steps): net 5's frozen fill moves towards (3) as the
step grows, the Adam version faster; nets 1 and 3 close to (3) from step 0.

## Extra arms

Stripes (periods 4, 8, 16):

| Net | Period 16 | Period 4 | Fitting speed, period 4 vs 16 | Ties |
| --- | --- | --- | --- | --- |
| 1, 3 | Good | Poor | Much slower | Round, near their own width |
| 5 | Good | Better than nets 1, 3 | About the same | Stretch along the stripes |
| 7 | Poor | Poor | - | None |

Against: net 5 also poor and slowed on period 4 (a built-in low-frequency bias).

Width (64 channels): nets 1 and 3 unchanged, seeds agree more. Net 5: plain kernel
changes less than at 16 channels; Adam kernel and fill closer to 1.67 l (Tachella et
al.); the l = 2 runs follow l too. Against: net 5's acquisition disappearing at 64.

Shi et al. checks:
- Same frequency bands (periods 32 and 21 px) at l = 2 vs 6: built in = same speed;
  acquired = faster at l = 2 for net 5 (supporting evidence only).
- Kernel frequency profile of net 5: flat at step 0, low-pass appearing in training,
  cutoff moving with l (our reading of Shi: low-pass at step 0, cutoff fixed).
- Shi's measure: nets 1, 3, 5 fit low bands first at every l (reproduction). Inside
  the gap, middle bands at l = 2: net 5 better than nets 1 and 3. Against: the reverse.

## Lucier's room (after exams)

Repeated refitting of fresh networks to their own output. Nets 1 and 3 widen the image
fast from l = 2; net 5 clearly more slowly; at l = 6 all slow. Against: net 5 widening
as fast as nets 1 and 3.

## Changes after launch

(none yet)
