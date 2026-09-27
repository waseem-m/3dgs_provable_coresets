# 3DGS Coresets: standalone PyTorch implementation

This package installs `gs-coresets-pytorch` and `gs_coresets_pytorch`.
It does not import the sibling CUDA package or require its sensitivity
extension. The existing PyTorch renderer and sensitivity mathematics are
preserved, including their historical behavior.

“PyTorch” describes the sensitivity implementation, not a CPU-only application.
Sensitivity workloads normally need an NVIDIA GPU. Stock GraphDECO rendering,
training and evaluation delegation also requires its native dependencies.

## Source installation

From this directory:

```bash
git submodule update --init -- external/gaussian_splatting
git -C external/gaussian_splatting submodule update --init --recursive -- \
  submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
conda env create -f environment.yml
conda activate gs-coresets-pytorch
python -m pip install --no-build-isolation --no-deps \
  ./external/gaussian_splatting/submodules/diff-gaussian-rasterization \
  ./external/gaussian_splatting/submodules/simple-knn \
  ./external/gaussian_splatting/submodules/fused-ssim
python -m pip install --no-build-isolation --no-deps -e .
python -m gs_coresets_pytorch.cli --help
```

The environment recipe declares Python 3.10, PyTorch 2.8 and CUDA 12.8.
It is an installation recipe, not a statement that this reorganized package
has completed GPU validation under that environment. Native dependencies must
be built for the active PyTorch/CUDA combination.

The CUDA package can coexist in the same compatible environment, using a
separate command and namespace. No files from its directory are required for
this package's installation or operation.

## Interface and compatibility

Use `python -m gs_coresets_pytorch.cli` or `gs-coresets-pytorch`.
Available commands include `sens_cams`, `sens`, `coreset`, `all_coresets`,
`classify`, `finetune`, and GraphDECO `train`, `render`, `metrics`,
`full_eval` delegation.

Sensitivity accepts historical `l1`/`l2`, RGB/no-color, max/mean, and all six
historical granularities. There is no CUDA backend selector or L2-agg option
in this package. This renderer and the CUDA package's default GraphDECO-native
renderer need not produce identical images or sensitivities.

Selection improvements are shared with the combined package: stable Top-K
ties, optional exact raw PLY rows, and selection-provenance manifests. These
change selection/I/O behavior, not renderer or sensitivity equations.
Stable cutoff ties may select different Gaussians than older native
`torch.topk` calls.

See [REPRODUCING.md](REPRODUCING.md) for a single-scene workflow.

## Licensing

Original Python code is MIT-licensed. GraphDECO and nested dependencies are
separate works under their own terms; the MIT license does not override them.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
