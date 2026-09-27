# Third-party software

The original Python package in `gs_coresets/` is released under MIT. Early
experiments were informed by
[`hbb1/torch-splatting`](https://github.com/hbb1/torch-splatting), an
MIT-licensed pure-PyTorch 3DGS implementation by Binbin Huang. The released
renderer was subsequently independently rewritten and does not intentionally
incorporate source code from that project.

The repository pins
[`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting)
as a Git submodule. Gaussian Splatting and its nested dependencies are separate
works governed by their own license files. In particular, Gaussian Splatting's
license limits use to research and evaluation and prohibits commercial use
without prior consent from its licensors. The MIT license at this repository's
root does not override third-party terms.

`extensions/diff_gaussian_sensitivity/` is an in-tree derivative of
`diff-gaussian-rasterization` commit
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`. Its source retains GraphDECO
copyright headers and is distributed under the complete license copied into
that directory. Consequently, the combined CUDA distribution is limited to
non-commercial research and evaluation even though independently authored
Python portions remain under MIT.
