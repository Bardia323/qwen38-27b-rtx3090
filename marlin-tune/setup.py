"""Standalone build of vLLM 0.27.1's GPTQ-Marlin kernel, tunable for sm86.

Build with build.sh (which sets CUDA_HOME to a CUDA 13 toolchain).  Nothing is
installed into the vLLM venv: the extension is built in-place here and imported
by adding this directory to sys.path.
"""

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc")
MARLIN = os.path.join(SRC, "libtorch_stable", "quantization", "marlin")

NAMESPACE = "marlin_tune"

sources = [
    os.path.join(HERE, "binding.cpp"),
    os.path.join(MARLIN, "marlin.cu"),
    os.path.join(HERE, "gemv_4bit.cu"),
] + sorted(glob.glob(os.path.join(MARLIN, "mt_kernel_*.cu")))

common_defines = [
    "-DMARLIN_NAMESPACE_NAME=%s" % NAMESPACE,
    # torch_utils.h needs aoti_torch_get_current_cuda_stream, which shim.h
    # only declares under USE_CUDA (vLLM's CMake defines it too).
    "-DUSE_CUDA",
]

cxx_flags = ["-O3", "-std=c++17", "-fdiagnostics-color=always"] + common_defines

nvcc_flags = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_86,code=sm_86",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    # NB: vLLM does *not* build marlin with --use_fast_math (only the Kimi-K3
    # kernels get it), so we do not either.
    # CUDA 13 defaults to -static-global-template-stub=true, which gives explicit
    # __global__ template instantiations internal linkage; marlin.cu takes their
    # address from another TU.  vLLM's own CMakeLists sets this too.
    "-static-global-template-stub=false",
    "-Xcompiler",
    "-fPIC",
] + common_defines

setup(
    name="marlin_tune",
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name="marlin_tune_ext",
            sources=sources,
            include_dirs=[SRC, MARLIN],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
