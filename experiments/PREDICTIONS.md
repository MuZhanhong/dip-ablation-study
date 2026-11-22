# November sweep: predictions

Written 20-22 Nov 2026, before any sweep results exist, and committed before launch.
Do not edit after launch: any change is added at the end, dated, with the reason. How each prediction is checked: ANALYSIS_PLAN.md. Runs: SWEEP_DESIGN.md.

## Definitions

- Tie between pixels p and q: how much a training step that pushes the output at q
  also moves the output at p. As a formula, Theta(p, q) = sum over every weight w of
  (d out_p / dw) x (d out_q / dw). This is the tangent kernel.
- Plain kernel: exactly that sum.
- Adam kernel: the same sum, but each weight's term is divided by the amount Adam
  scales that weight's step down by, so it describes the step Adam actually takes.
  Adam has no memory before its first step, so at step 0 there is only the plain kernel.
- Prior kernel K: how strongly two pixels' outputs vary together across 100 untrained
  copies of the network, each with different random weights. Measured 17 Nov: net 1's
  ties fall to half by 6.9 px and net 3's by 8.3 px; nets 5 and 7 have no ties.
- Width of a tie curve: the distance where the tie has dropped halfway from 1
  (distance 0) down to its far-away level.
- Floor: that far-away level, i.e. the part of a tie that does not depend on distance.
  It is measured in every run using one distant pixel.
- Image width: blurred noise with blur scale l has width 1.67 l, which is 3.3, 5.0,
  6.7 and 10 px at l = 2, 3, 4 and 6.
- f_d: how good the fill is in ring d of the gap (ring 1 is the gap's edge, ring 8 is
  its centre). 0 means as good as the harmonic fill; 1 means as good as the best
  possible fill.

## Topic 1: what happens without weight sharing

### Claim

Without shared weights, our runs should give

gap error = grey error + s_0^2,

where s_0^2 is the variance of the gap output.

Why: in an unshared network (nets 8 and 2), every pixel has its own weights in the
final layers. The gap's own weights get no training signal, because no gap pixel is in
the loss, so they stay at their random starting values. In our 10 Nov trial runs this
gave a gap output centred on grey (0.5), with noise unrelated to the true image. The
size of that noise, s_0^2, is set through batch norm, whose statistics the gap shares
with the visible pixels.

Derivation. Over the gap pixels, let D = gap output - 0.5 (the noise) and
e = truth - 0.5 (grey's error at each pixel). D has mean 0 (the 0.5 result), so

gap error = mean of (D - e)^2 = grey error + Var(D) - 2 Cov(D, e)
          = grey error + s_0^2 - 2 Cov(D, e),

and Cov(D, e) = 0 when the noise is unrelated to the true image.

### Measure

Gap error minus grey error, on the same image, in each of the 8 rings. Under the claim,
this equals s_0^2.

### Predictions

| What | Prediction |
| --- | --- |
| Net 8 | s_0^2 about 0.011. Same in every ring (no better at the gap's edge), on every image and at every l |
| Net 2 (4 runs) | s_0^2 about 0.04-0.05, on every image |
| Net 8 at different contrasts | s_0^2 about 0.011 at contrast 0.5 and 1; about 0.002 at contrast 2 and 3; contrast 1.5 could go either way |

"About" means within a factor of 2. The numbers come from short (50-step) trial runs
on 10 Nov.

### Why the contrast arm

s_0^2 depends on how far the visible pixels sit from 0.5. We predict a switch between
contrast 1.5 and 2, not a steady fall. The cause (batch-norm statistics) is not tested
and goes in the report as a caveat.

### What would show the claim is wrong

- Gap error minus grey error smaller near the gap's edge than in the middle (net 8
  using the visible pixels), or changing with l.
- Net 2 clearly better than grey on most images (gap error minus grey error below 0).
- s_0^2 not changing with contrast at all, or falling steadily rather than switching.

Note: net 2 matching grey plus noise does not prove that nothing reaches the gap. A
signal could arrive and go unused.

## Topic 2: does network 5 learn its scale from the image?

### Claim

Nets 1 and 3 (with upsampling) have a scale built in: their ties already have a set
width before training. Net 5 (3x3 filters, no upsampling) has no built-in scale, and it
takes the image's scale during training. Kernels are already known to adapt to what
they are fitting in many networks, so the claim is about how much: net 5 adapts much
more than nets 1 and 3.

Shi et al. attribute the low-frequency preference to a latent ability of the network,
found across architectures, and conclude that the architecture is not critical
(p. 7). We test this directly in the Shi et al. checks under Extra arms.

### Measure

The width of the tie curves at step 5000, at each l. Fit width = a + beta x 1.67 l
across l = 2, 3, 4 and 6.
- beta: how much the width follows the image's width (1 = fully, 0 = not at all).
- a: the fixed, built-in part of the width.

### Predictions

| Net | Step 0 (plain kernel) | Step 5000 (plain and Adam) | beta | a |
| --- | --- | --- | --- | --- |
| 1 | Local ties | Stays near its own step-0 width at every l | Near 0 | Near its own width |
| 3 | Local ties | Stays near its own step-0 width at every l | Near 0 | Near its own width |
| 5 | No consistent ties (random signs) | Grows with l towards 1.67 l; Adam at least as close as plain | Clearly above 0, and above nets 1 and 3 | Within 1 px of 0 |
| 7 | No ties above the floor | No ties above the floor | - | - |

Also:
- Net 5's width grows gradually over training.
- Horizontal and vertical widths are the same.
- The output's own width inside the gap:
  - nets 1 and 3: near their own width and stable;
  - net 5: starts rough and moves towards the width of the best possible fill,
    without getting rougher late in training;
  - net 7: rough throughout.

### Where the scale is stored

| Plain follows l | Adam follows l | Meaning |
| --- | --- | --- |
| Yes | Yes | The scale is stored in the weights |
| No | Yes | The scale lives in Adam's memory |
| No | No | Net 5 does not acquire the scale |

### Other evidence for this claim (details under Extra arms)

What we look for each time: net 5 behaving differently from nets 1 and 3, in a way a
learned scale explains.

- Stripes. Our claim: net 5 has no built-in preference for coarse patterns, so it fits
  fine stripes (period 4) about as fast as coarse ones (period 16). Nets 1 and 3 do
  have that preference, so they are much slower on fine stripes. Speed alone cannot
  separate "learned a scale" from "no scale at all", because a flat kernel is also
  fast. So learning is checked separately: net 5's ties stretch along the stripes,
  and its fill carries them on into the gap, which a flat kernel cannot do.
- Same frequency bands at l = 2 and 6. Our claim: net 5 fits these bands faster at
  l = 2 than at l = 6, by more than nets 1 and 3 do. A network with a fixed bias would
  fit them at the same speed at both l. Every network may show some difference,
  because Adam's step sizes depend on the image and the gap mixes bands. So only net
  5's difference being larger than nets 1 and 3's counts, and even then it is
  supporting evidence only.

### What would show the claim is wrong

- Net 5's beta near 0, or no larger than for nets 1 and 3.
- Net 5's a clearly above 1 px, which would mean a small built-in scale.
- Net 5's width inside the gap falling late in training.

## Topic 3: how much of the fill is built in, and how much is learned?

### Claim

For nets 1 and 3, the built-in prior does most of the work. For net 5, most of the fill
is learned from the image during training. So the learned part is larger for net 5
than for nets 1 and 3.

### Measure

Three fills of the gap for each run, each scored by f_d in every ring:
1. Prior fill: from K, so from the architecture alone, with no training.
2. Frozen fill: from the tangent kernel at step 0, i.e. where plain gradient descent
   would end up if the ties never changed.
3. Actual fill: what the network really produces.

The learned part is (3) minus (2). Part of it could come from Adam rather than from
the ties changing; step B with the Adam kernel checks this.

### Predictions

| Net | (1) Prior | (2) Frozen | (3) Actual |
| --- | --- | --- | --- |
| 1 | Decent near the gap's edge | Close to (3) | Best near l = 4, worse at 2 and 6 |
| 3 | Same as net 1, slightly smoother | Close to (3) | Similar to net 1 |
| 5 | Poor (about a constant fill) | Poor | Clearly better than (1) and (2), at every l |
| 7 | Poor | Poor | Poor |

### Kernel frozen at later steps (step B)

The same as (2), but with the kernel frozen at step 500, 1000, ..., 5000 instead of
step 0. For net 5, the frozen fill moves towards (3) as the step grows, and faster with
the Adam kernel. For nets 1 and 3 it is close to (3) from step 0.

### What would show the claim is wrong

- Net 5's learned part no larger than nets 1 and 3's.
- Net 5's frozen fill (2) about as good as its actual fill (3), meaning nothing was
  learned.
- Nets 1 and 3's frozen fill clearly worse than their actual fill, meaning they learn
  as much as net 5.

## Extra arm: stripes

### Claim

The same as Topic 2, tested on images that are each almost a single frequency. Net 5
has no built-in preference for coarse patterns, and it learns the stripes' scale. Nets
1 and 3 prefer coarse patterns.

### Measure

Three stripe images at 30 degrees, with periods 4, 8 and 16 px, all with the same
contrast.
- Fitting speed: steps until the training error falls below 10% of the stripes'
  variance.
- Gap fill: how well the stripes are carried on into the gap.
- Tie maps: the ties from gap pixels to visible pixels.

### Predictions

| Net | Gap fill, period 16 | Gap fill, period 4 | Fitting speed, period 4 vs 16 | Ties |
| --- | --- | --- | --- | --- |
| 1, 3 | Good | Poor | Much slower on period 4 | Round, near their own width |
| 5 | Good | Better than nets 1 and 3 | About the same | Stretch along the stripes |
| 7 | Poor | Poor | - | None |

### What would show the claim is wrong

- Net 5 also much slower on period 4 than on period 16, meaning a built-in
  low-frequency bias.
- Net 5's ties round rather than along the stripes, or its fill not carrying the
  stripes into the gap, meaning no scale is learned.

There is one image per period, so the uncertainty rests on the 3 model seeds. This is
supporting evidence only.

## Extra arm: wider networks (64 channels)

### Claim

Net 5 still learns the image's scale at 64 channels. In a wider network the plain
kernel changes less during training, so the learning shows up mainly in the Adam
kernel and in the fill.

### Measure

The same widths and fills as Topics 2 and 3, for the 64-channel runs listed in
SWEEP_DESIGN.md, compared with the same runs at 16 channels.

### Predictions

- Nets 1 and 3: widths unchanged from 16 channels; the seeds agree more closely.
- Net 5: the plain kernel changes less during training than at 16 channels. The Adam
  kernel and the fill get closer to 1.67 l than at 16 channels, including in the
  l = 2 runs. (Tachella et al.: trained with Adam, wide networks still adapt to the
  image instead of behaving like their fixed-kernel limit.)

### What would show the claim is wrong

- Net 5's Adam kernel and fill no longer following l at 64 channels, meaning the
  learning disappears in wider networks.

## Extra arm: checks against Shi et al.

### Claim

Shi et al. observe that DIP fits low frequencies first. While the observation is
correct in most settings, we disagree with the explanation. They attribute it to "a
latent ability to learn low frequencies faster than learning high frequencies"
(p. 7), found across architectures, and conclude that the architecture is not
critical. Our claim is that net 5's preference is learned from the image, while for
nets 1 and 3 the preference is mostly derived from their architecture, mostly from
the upsampling.

### Measures and predictions

1. Their own measure (reproduction): nets 1, 3 and 5 all fit low bands before high
   ones, at every l.
2. Inside the gap, middle bands at l = 2: net 5 fills these better than nets 1 and 3.
3. Kernel frequency profile:
   - net 5: flat at step 0, with a low-pass appearing during training and a cutoff
     that moves with l;
   - nets 1 and 3: already low-pass at step 0, with a cutoff that stays roughly fixed.
4. Same frequency bands at l = 2 and 6 (periods 32 and 21 px): net 5 fits them faster
   at l = 2 than at l = 6, by more than nets 1 and 3 do. Supporting evidence only, as
   explained in Topic 2.

### What would show the claim is wrong

- Inside the gap at l = 2, nets 1 and 3 filling the middle bands better than net 5.
- Net 5's kernel already low-pass at step 0, or its cutoff not moving with l.
- Net 5's l = 2 vs 6 difference no larger than nets 1 and 3's.

## Lucier's room (run after exams)

### Claim

A network's own bias shows up when it is refitted again and again to its own output:
a fresh network is fitted to the image, its output becomes the next target, and so on.
A built-in scale pulls the image towards that scale every round. Net 5, with no
built-in scale, follows whatever image it is given, so it changes the image much less.

### Measure

The correlation width and the amplitude (standard deviation) of the image after each
of 10 rounds, at l = 2 and 6. Each network's number of steps per round is set so that
round 1 reaches the same, clearly imperfect training error. We compare how fast the
width grows. No end point is predicted, because repeated imperfect refitting keeps
smoothing (Mobahi et al. 2020).

### Predictions

- l = 2 (image finer than nets 1 and 3's own scales): nets 1 and 3 widen the image
  fast; net 5 clearly more slowly.
- l = 6 (image already coarser than their own scales): all networks widen it slowly.

### What would show the claim is wrong

- Net 5 widening the l = 2 image as fast as nets 1 and 3.
