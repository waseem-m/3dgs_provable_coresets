# Contributing

Keep changes focused and document public CLI behavior. Install from the
implementation directory you intend to change. Keep the two distributions
independently usable and preserve upstream licenses and dependency pins.

For lightweight source and interface checks:

```bash
python -m compileall -q cuda/gs_coresets pytorch/gs_coresets_pytorch
gs-coresets --help
gs-coresets-pytorch --help
```

Validate numerical changes privately against the relevant backend's definition
using new output directories. Do not assume the PyTorch and CUDA sensitivity
definitions are interchangeable. Do not include datasets, model artifacts, or
machine-specific execution infrastructure in public changes.
