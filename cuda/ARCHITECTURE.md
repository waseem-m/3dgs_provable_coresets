# CUDA sensitivity architecture

## Boundaries

The Python package remains responsible for PLY and camera I/O, command-line
handling, camera grouping, scene aggregation, coreset sampling, and GraphDECO
training/evaluation delegation. `diff_gaussian_sensitivity` is a forward-only
CUDA extension. It is not used for training and has no backward operation.

The extension is derived from GraphDECO rasterizer commit
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`. The stock preprocessing, tile
duplication, radix sorting, and rendering path remains intact.

## Renderer stages

The implementation uses explicit stage names:

- `geometry_state.rgb` is the feature buffer produced by pinned
  `computeColorFromSH`.
- `C^G = sum_g(a_g)` is the Gaussian-only pixel accumulation before background.
- `C^B = C^G + T_final * background` is GraphDECO `renderCUDA/out_color`.
- `C^F = clamp(C^B, 0, 1)` is the normal wrapper output when trained exposure is
  disabled.

Sensitivity uses the individual `a_g` values only. It never uses `C^B`, `C^F`,
residual background, or per-Gaussian saturation/headroom attribution.

## Sparse sensitivity passes

For each camera, GraphDECO preprocessing, tile duplication, radix sorting, and
forward compositing run unchanged. The extension retains the geometry, sorted
tile lists, final transmittance, and contributor counts.

A denominator replay then launches one 16×16 block per tile. Every pixel thread
traverses only `ranges[tile_id]` in `point_list` and reproduces GraphDECO's
power test, alpha cap and cutoff, transmittance, and early termination. L1
accumulates each accepted channel contribution. L2 squares each individual
channel contribution before any aggregation.

A second sparse replay uses the same acceptance helper and tile range to form
per-Gaussian numerators and ratios. Channel, pixel, tile, and image maxima plus
scene partials are returned to Python.

For the no-color variant, the sensitivity term is the scalar compositing
weight `alpha * T_before`, replicated over the three channel slots. The
renderer-stage diagnostic `C^G` is never redefined: it remains the actual RGB
sum of `feature * alpha * T_before`.

For a query `q`, let `term(a)=a` for L1 and `term(a)=a²` for L2. The replay
forms

```text
r[g,q] = sum_{(i,p,c) in q} term(a[g,i,p,c])
         / sum_h sum_{(i,p,c) in q} term(a[h,i,p,c]).
```

Channel queries contain one `(i,p,c)`, pixel queries contain the three channels
of one pixel, tile queries contain all channels in one non-overlapping 16×16
tile, and image queries contain one full camera image. Their sensitivities are
`max_q r[g,q]`. The scene is one query spanning every camera, so scene
sensitivity is its single ratio with no maximum. Empty queries return zero.
For standard L2 the square is applied before every channel or spatial sum.

The optional RGB `l2-agg` variant instead uses `(sum_c a[g,i,p,c])²`
independently for each Gaussian/pixel, then aggregates over pixels. Channel
output stays ordinary channel-square L2; its filename and per-output metadata
identify it as such. `l2-channel` is an exact alias of standard `l2`.
No-color replicates one weight across channels, so the extra constant factors
in `l2-agg` cancel: it is mathematically redundant with no-color `l2`.

Mathematically, every denominator sums over all Gaussians. GraphDECO's tile list
is the exact sparse representation of the only Gaussians that can contribute to
that tile, so off-list Gaussians are implicit zeros and are never inspected.
There is no full-model scan per pixel or tile and no Gaussian×pixel tensor.

## Backend semantics

`gs-coresets sens` defaults to `--backend cuda`. This backend uses GraphDECO
projection and compositing semantics. `--backend pytorch` uses the regular
vectorized PyTorch renderer and is never selected automatically. The historical
PyTorch implementation remains unchanged and is not a CUDA correctness oracle.

Corrected CUDA supports maximum reduction for channel, pixel, tile, and image
queries plus a single global scene ratio. Per-batch and mean sensitivity are not
part of the corrected CUDA method.
