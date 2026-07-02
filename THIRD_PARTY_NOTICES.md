# Third-party software

The Python source in this repository is released under the MIT License. Early
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
