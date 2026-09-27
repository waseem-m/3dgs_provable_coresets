from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent


setup(
    name="diff_gaussian_sensitivity",
    version="0.1.0",
    packages=["diff_gaussian_sensitivity"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_sensitivity._C",
            sources=[
                str(ROOT / "cuda_rasterizer" / "rasterizer_impl.cu"),
                str(ROOT / "cuda_rasterizer" / "forward.cu"),
                str(ROOT / "cuda_rasterizer" / "backward.cu"),
                str(ROOT / "sensitivity.cu"),
                str(ROOT / "sensitivity_points.cu"),
                str(ROOT / "ext.cpp"),
            ],
            include_dirs=[str(ROOT / "third_party" / "glm")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
