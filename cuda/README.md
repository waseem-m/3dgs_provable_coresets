# 3DGS Coresets: CUDA and PyTorch backends

This standalone package installs `gs-coresets` and `gs_coresets`.
Sensitivity defaults to CUDA. Select `--backend pytorch` explicitly to use the
PyTorch tensor-based renderer and sensitivity implementation. Missing CUDA support is an error, never an
automatic fallback.

## Source installation

From this directory, initialize the pinned dependency and its required nested
modules (the optional viewer is not needed):

```bash
git submodule update --init -- external/gaussian_splatting
git -C external/gaussian_splatting submodule update --init --recursive -- \
  submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
conda env create -f environment.yml
conda activate gs-coresets
python -m pip install --no-build-isolation --no-deps \
  ./external/gaussian_splatting/submodules/diff-gaussian-rasterization \
  ./external/gaussian_splatting/submodules/simple-knn \
  ./external/gaussian_splatting/submodules/fused-ssim \
  ./extensions/diff_gaussian_sensitivity
python -m pip install --no-build-isolation --no-deps -e .
python -m gs_coresets.cli --help
```

Use a CUDA-capable build machine and a CUDA toolkit compatible with the installed
PyTorch build. Rebuild the extension when changing PyTorch/CUDA versions.
The environment recipe declares Python 3.10, PyTorch 2.8 and CUDA 12.8; this
recipe is not a claim of completed GPU validation for that combination.

The release passed an A100 packaging smoke test with Python 3.10, PyTorch 2.0,
and CUDA 12.1, covering independent installation, co-installation, sensitivity
execution, and selection-output safety. This does not validate the separate
PyTorch 2.8/CUDA 12.8 environment recipe. Python 3.10 or later is required.

To co-install the PyTorch-only package into an already compatible environment,
initialize its separate GraphDECO submodule as described in its README, then
run `python -m pip install --no-build-isolation --no-deps -e ../pytorch`.
Its command and import namespace are distinct.

## Sensitivity definitions

For an accepted GraphDECO contribution, `a_c = feature_c * alpha * T_before`.
The feature is exactly GraphDECO's post-SH feature, including its own lower
clamp, with no additional color clamp. Background and final-image clamping do
not enter sensitivity.

| Norm | Channel quantity | Pixel quantity before spatial aggregation |
|---|---|---|
| `l1` | `a_c` | `sum_c a_c` |
| `l2`, alias `l2-channel` | `a_c²` | `sum_c a_c²` |
| `l2-agg` | ordinary `a_c²` | `(sum_c a_c)²` |

Channel, pixel, non-overlapping 16×16 tile, and image sensitivities maximize
each Gaussian's fraction over queries. Scene sensitivity is one ratio over all
cameras. For both L2 definitions, spatial aggregation follows the indicated
per-pixel operation; it never squares a whole tile/image sum.

The no-color variant uses `alpha * T_before` rather than SH/RGB features.
Repeated channel factors cancel from its ratios. No-color `l2-agg` therefore
duplicates no-color `l2`; no-color channel duplicates pixel at a fixed norm.

Mathematically denominators sum all Gaussian contributions. Computationally,
GraphDECO's sorted tile lists are the exact sparse support: off-list Gaussians
contribute zero and are not scanned.

CUDA supports five granularities and max reduction only; camera chunking is
execution scheduling, not a per-batch sensitivity query. The PyTorch backend
supports L1/L2, mean/max reductions, and per-batch queries in addition to the
other five granularities. Its renderer and aggregation rules differ from the
GraphDECO-native CUDA backend; switching backends can change sensitivity values.

## Outputs and selection

Existing CUDA L2 files remain `per_<granularity>_l2_max.pt`.
L2-agg writes `per_<pixel|tile|image|scene>_l2_agg_max.pt`; its optional channel
output is ordinary `per_channel_l2_max.pt`. No-color adds `_nocolor` before
`.pt`. Per-output metadata identifies ordinary channel L2 even within an
L2-agg run.

Use `coreset --sens-norm l2-agg` only for pixel or coarser granularities; use
`--sens-norm l2` for its channel tensor. Use `--sens-nocolor` when selecting
a no-color file. The resolver never substitutes a missing RGB/no-color file.

See [REPRODUCING.md](REPRODUCING.md) for direct commands and selection safety,
and [ARCHITECTURE.md](ARCHITECTURE.md) for renderer-stage and replay details.

## Licensing

Original `gs_coresets/` Python code is MIT-licensed. The derivative in
`extensions/diff_gaussian_sensitivity/` and pinned GraphDECO dependencies
retain their own licenses. This combined distribution is not wholly MIT;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
