# Provable Pruning for Efficient 3D Gaussian Splatting via Coresets

Official research code for constructing sensitivity-based coresets of 3D
Gaussian Splatting models, pruning them to a requested size, and evaluating the
result before and after fine-tuning.

Authors: **Waseem Mousa**, **Alaa Maalouf**, Department of Computer Science,
University of Haifa.

This is the public code release accompanying the submitted paper. Publication
metadata will be updated after the review process.

## Features

- Per-channel, per-pixel, per-tile, per-image, per-batch, and per-scene
  sensitivity estimation.
- Sensitivity-aware multinomial sampling and deterministic top-k selection.
- Uniform-sampling baselines with reproducible seeds.
- Coreset construction by target size or pruning ratio.
- GraphDECO-compatible training, fine-tuning, rendering, and metric commands.

## Installation

The validated configuration is Linux, Python 3.10, PyTorch 2.8, and CUDA 12.8.
An NVIDIA GPU is required for training, sensitivity computation, rendering, and
fine-tuning.

```bash
git clone --recursive https://github.com/waseem-m/3dgs_provable_coresets.git
cd 3dgs_provable_coresets
conda env create -f environment.yml
conda activate gs-coresets
```

If the repository was cloned without `--recursive`:

```bash
git submodule update --init --recursive
conda env update -f environment.yml
```

Confirm the command-line interface:

```bash
gs-coresets --help
gs-coresets coreset --help
```

## Quick start from a pretrained model

The following example uses a GraphDECO model saved at iteration 30,000.

```bash
DATASET=/path/to/scene
MODEL=/path/to/model
PLY="$MODEL/point_cloud/iteration_30000/point_cloud.ply"

gs-coresets sens_cams \
  --model "$PLY" \
  --model-dir "$MODEL" \
  --source "$DATASET" \
  --out outputs/cameras \
  --extract-data-device cpu \
  --extract-resolution -1

gs-coresets sens \
  --model "$PLY" \
  --cams outputs/cameras/extract_cameras_train.json \
  --out outputs/sensitivities \
  --sensitivity-norm l1 \
  --sensitivity-reduce max

gs-coresets coreset \
  --model "$PLY" \
  --sens outputs/sensitivities \
  --sens-granularity per_image \
  --sens-norm l1 \
  --sens-reduce max \
  --prune-ratio 0.90 \
  --sampling-mode topk \
  --out outputs/per_image_l1_max_0.90prune.ply
```

For a uniform baseline, replace the sensitivity arguments with `--uniform` and
set `--seed` explicitly.

## Repository layout

```text
gs_coresets/                 Python package and CLI implementation
external/gaussian_splatting/ pinned public GraphDECO submodule
```

## Verifying the installation

Confirm the installed CLI and inspect the method-specific command:

```bash
gs-coresets --help
gs-coresets sens --help
gs-coresets coreset --help
```

For a direct single-scene workflow from a pretrained model through fine-tuning
and evaluation, see [REPRODUCING.md](REPRODUCING.md).

## Citation

```bibtex
@misc{mousa2026gscoresets,
  title  = {Provable Pruning for Efficient 3D Gaussian Splatting via Coresets},
  author = {Mousa, Waseem and Maalouf, Alaa},
  year   = {2026},
  note   = {Submitted manuscript}
}
```

## License

The code authored for this repository is available under the MIT License. The
pinned Gaussian Splatting submodule and its nested dependencies retain their
own licenses, including research-only restrictions. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
