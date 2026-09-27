# 3DGS Coresets: standalone PyTorch implementation

This package installs `gs-coresets-pytorch` and `gs_coresets_pytorch`.
It does not import the sibling CUDA package or require its sensitivity
extension. Rasterization and sensitivity computation use PyTorch tensor
operations and support batched multi-camera processing.

“PyTorch” describes the sensitivity implementation, not a CPU-only application.
Sensitivity workloads normally need an NVIDIA GPU. The `train`, `render`, and
`metrics` commands run stock GraphDECO tools and require their native dependencies.

## Source installation

From the repository root:

```bash
cd pytorch
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

Python 3.10 or later is required. Build native dependencies for the active
PyTorch/CUDA combination.

| Environment | Verification status |
|---|---|
| Python 3.10, PyTorch 2.0, CUDA 12.1, NVIDIA A100 | Installation, co-installation, sensitivity execution, and selection-output safety tested |
| Python 3.10, PyTorch 2.8, CUDA 12.8 (`environment.yml`) | Provided installation recipe; not GPU-tested for this release |

The CUDA package can coexist in the same compatible environment, using a
separate command and namespace. No files from its directory are required for
this package's installation or operation.

## Interface and compatibility

Use `python -m gs_coresets_pytorch.cli` or `gs-coresets-pytorch`.
Available commands include `sens_cams`, `sens`, `coreset`, `all_coresets`,
`classify`, and `finetune`. The `train`, `render`, `metrics`, and `full_eval`
commands run the corresponding GraphDECO tools.

Sensitivity supports `l1`/`l2`, RGB/no-color, max/mean reductions, and six
granularities: channel, pixel, tile, image, batch, and scene.
There is no CUDA backend selector or L2-agg option in this package. Its
sensitivity renderer and aggregation rules differ from the GraphDECO-native
CUDA sensitivity backend, so their sensitivity outputs are not interchangeable.
The `render` command uses stock GraphDECO, not this PyTorch sensitivity renderer.

Top-K selects the highest-scoring Gaussians, breaking equal-score ties by
ascending original Gaussian index. Enable `--preserve-raw-parameters` to copy
complete original PLY vertex rows and save a JSON record of the selected
indices and file hashes. This stable tie rule can differ from native
`torch.topk`, which does not specify membership among tied cutoff scores.

See [REPRODUCING.md](REPRODUCING.md) for a single-scene workflow.

## Licensing

Original Python code is MIT-licensed. GraphDECO and nested dependencies are
separate works under their own terms; the MIT license does not override them.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
