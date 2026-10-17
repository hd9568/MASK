from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def include_dirs():
    return [
        str((ROOT / "cutlass" / "include").resolve()),
        str((ROOT / "cutlass" / "examples" / "common").resolve()),
        str((ROOT / "cutlass" / "tools" / "util" / "include").resolve()),
        str((ROOT / "src").resolve()),
        str(ROOT),
    ]


setup(
    name="SRBCRSSpMM",
    ext_modules=[
        CUDAExtension(
            name="SRBCRSSpMM",
            sources=[
                "SRBCRSSpMM.cpp",
                "SRBCRSSpMM_kernel.cu",
            ],
            include_dirs=include_dirs(),
            define_macros=[("f32", None)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                    "-ftemplate-backtrace-limit=0",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
