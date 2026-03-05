from skbuild import setup
import pybind11

setup(
    cmake_args=[
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ],
    name="cuda-akaze",
    version="0.1.0",
    description="CUDA-accelerated AKAZE feature detection and matching",
    author="",
    license="MIT",
    packages=["cuda_akaze"],
    python_requires=">=3.8",
    install_requires=["numpy"],
    cmake_install_dir="cuda_akaze",
)
