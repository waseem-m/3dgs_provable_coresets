# 3DGS Coresets

Code for **Provable Pruning for Efficient 3D Gaussian Splatting via Coresets**.

**Waseem Mousa and Alaa Maalouf**

Department of Computer Science, University of Haifa

## Choose an implementation

| Folder | Command | Python module | Sensitivity backends |
|---|---|---|---|
| [cuda/](cuda/README.md) | `gs-coresets` | `gs_coresets` | CUDA (default), explicit PyTorch |
| [pytorch/](pytorch/README.md) | `gs-coresets-pytorch` | `gs_coresets_pytorch` | PyTorch only |

Each folder is independently installable and includes its own pinned GraphDECO
dependency. Neither package imports the other. They may be installed together
in a compatible environment; their Python packages and command names differ.
Keep the source checkout in place when using the documented editable installs.

The CUDA backend implements Gaussian-contribution sensitivities directly over
stock GraphDECO rasterization. The PyTorch implementation retains the historical
renderer and sensitivity definitions. They are not interchangeable numerical
implementations; use separate output directories and record the backend.

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

Both CLIs support sensitivity calculation, coreset construction, camera
extraction, and GraphDECO training/rendering/evaluation delegation. Top-K now
breaks equal ranking-score ties by ascending original Gaussian index. Optional
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
