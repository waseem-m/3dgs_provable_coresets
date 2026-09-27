# Changelog

## 0.3.0

- Separate, independently installable `cuda/` and `pytorch/` packages.
- Combined `gs-coresets` package retains CUDA and explicit PyTorch backends;
  standalone `gs-coresets-pytorch` has its own Python namespace.
- GraphDECO-native CUDA L1, channel-square L2 and RGB-aggregate L2-agg, with
  explicit no-color variants and per-output semantics metadata.
- Stable Top-K ranking with original-index tie-breaking. This deliberately
  replaces native Top-K's unspecified cutoff-tie membership.
- Optional lossless source PLY vertex-row selection and provenance manifests.
- New coreset CLI outputs are staged and published without overwriting existing
  files; raw-selection manifests are published last.
- Preserved historical PyTorch rendering and sensitivity mathematics.
- Independently pinned GraphDECO dependencies and scoped licensing notices.

## 0.1.0

Initial method and CLI release, with an independently rewritten PyTorch
renderer and acknowledgment of torch-splatting's influence on early work.
