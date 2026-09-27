# 3DGS Coresets

Code for **Provable Pruning for Efficient 3D Gaussian Splatting via Coresets**.

**Waseem Mousa and Alaa Maalouf**

Department of Computer Science, University of Haifa

## Choose an implementation

- Choose **CUDA** to compute sensitivities using stock GraphDECO rasterization.
- Choose **PyTorch** to use or adapt a tensor-based renderer with batched
  multi-camera processing, without building the CUDA sensitivity extension.
  This implementation also normally runs on an NVIDIA GPU.

| Folder | Command | Python module | Sensitivity backends |
|---|---|---|---|
| [cuda/](cuda/README.md) | `gs-coresets` | `gs_coresets` | CUDA (default), explicit PyTorch |
| [pytorch/](pytorch/README.md) | `gs-coresets-pytorch` | `gs_coresets_pytorch` | PyTorch only |

Each folder is independently installable and includes its own pinned GraphDECO
dependency. Neither package imports the other. They may be installed together
in a compatible environment; their Python packages and command names differ.
Keep the source checkout in place when using the documented editable installs.

The **CUDA implementation** computes sensitivities directly within stock
GraphDECO's rasterization pipeline. The **PyTorch implementation** uses a
renderer and sensitivity computation written with PyTorch tensor operations,
including batched multi-camera processing.

The renderers used during sensitivity computation differ in rasterization and
aggregation, so switching sensitivity backends can change the results.
The separate `render` command in both packages runs stock GraphDECO rendering;
it does not select between the two sensitivity renderers.
See the implementation-specific documentation for supported sensitivity modes;
keep outputs separate and record which backend produced them.

## Installation and usage

Clone the source and follow the instructions in the chosen folder:

```bash
git clone https://github.com/waseem-m/3dgs_provable_coresets.git
cd 3dgs_provable_coresets
```

- [CUDA installation](cuda/README.md) and [single-scene workflow](cuda/REPRODUCING.md).
- [PyTorch installation](pytorch/README.md) and [single-scene workflow](pytorch/REPRODUCING.md).

There is no root-level Python distribution. Install from `cuda/` or
`pytorch/`. CUDA extensions are built from source for the active PyTorch/CUDA
environment; prebuilt wheels are not supplied.

Both CLIs support sensitivity calculation, coreset construction, and camera
extraction, and run GraphDECO's training, rendering, and evaluation tools.
Top-K breaks equal ranking-score ties by ascending original Gaussian index. Optional
raw PLY preservation copies complete selected vertex records and records their
identities and hashes in a selection-provenance JSON file. See each workflow.

## Dependencies and licensing

Both implementations pin GraphDECO Gaussian Splatting at
`54c035f7834b564019656c3e3fcc3646292f727d`, with rasterizer
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`.

This is a **mixed-license distribution**. Original Python code is MIT-licensed;
the GraphDECO-derived CUDA extension and GraphDECO dependencies retain their
research/evaluation, non-commercial terms. The root MIT notice does not
override those terms. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and the component license files before use.

## Citation

```bibtex
@misc{mousa2026gscoresets,
  title  = {Provable Pruning for Efficient 3D Gaussian Splatting via Coresets},
  author = {Mousa, Waseem and Maalouf, Alaa},
  year   = {2026},
  note   = {Submitted manuscript}
}
```
