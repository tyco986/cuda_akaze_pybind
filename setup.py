from skbuild import setup
import os
import pybind11

cmake_args = [f"-Dpybind11_DIR={pybind11.get_cmake_dir()}"]
# Allow override: CUDA_ARCH=native or CUDA_ARCH=75;80;90
if os.environ.get("CUDA_ARCH"):
    cmake_args.append(f"-DCUDA_ARCH={os.environ['CUDA_ARCH']}")

setup(
    name="cuda-akaze",
    version="0.1.0",
    description="CUDA-AKAZE: keypoint detection, descriptor computation, and matching",
    author="",
    license="",
    python_requires=">=3.7",
    packages=["cuda_akaze"],
    package_dir={"cuda_akaze": "cuda_akaze"},
    cmake_install_dir="cuda_akaze",
    cmake_args=cmake_args,
)
