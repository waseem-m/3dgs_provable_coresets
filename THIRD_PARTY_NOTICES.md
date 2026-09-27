# Third-party software and license boundaries

Original Python code in `cuda/gs_coresets/` and
`pytorch/gs_coresets_pytorch/` is released under the MIT License.
Early experiments were informed by
[`hbb1/torch-splatting`](https://github.com/hbb1/torch-splatting), an
MIT-licensed pure-PyTorch 3DGS implementation by Binbin Huang. The released
renderer was subsequently independently rewritten and does not intentionally
incorporate source code from that project.

Each implementation pins
[`graphdeco-inria/gaussian-splatting`](https://github.com/graphdeco-inria/gaussian-splatting)
at `54c035f7834b564019656c3e3fcc3646292f727d`. Its rasterizer is pinned at
`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`. These submodules and their nested
dependencies retain their own license files. In particular, GraphDECO limits
use to research and evaluation and prohibits commercial use without prior
consent from its licensors.

`cuda/extensions/diff_gaussian_sensitivity/` is an in-tree derivative of that
rasterizer. Its original copyright headers and complete
[license](cuda/extensions/diff_gaussian_sensitivity/LICENSE.md) are retained.
Its vendored dependencies also retain their notices, including GLM's
`copying.txt`.

The complete repository is a mixed-license distribution. The root MIT license
covers original work, not the GraphDECO derivative, submodules, or vendored
third-party code. The combined CUDA distribution is limited to non-commercial
research and evaluation under the applicable GraphDECO terms.
