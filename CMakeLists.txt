cmake_minimum_required(VERSION 3.18)

project(CudaAkaze LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Adaptive CUDA architecture ---
# User can override: -DCUDA_ARCH="75;80;90" or -DCUDA_ARCH=native
set(CUDA_ARCH "" CACHE STRING "CUDA architectures (e.g. 75;80;90, native, or empty for auto)")

if(CUDA_ARCH STREQUAL "")
  # Auto-select based on CUDA version (min supported arch)
  if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    set(CUDA_ARCH "75;80;90")
  elseif(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
    set(CUDA_ARCH "60;70;75;80")
  elseif(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
    set(CUDA_ARCH "35;50;60;70")
  else()
    set(CUDA_ARCH "75;80;90")
  endif()
  message(STATUS "CUDA architectures (auto): ${CUDA_ARCH}")
else()
  message(STATUS "CUDA architectures (user): ${CUDA_ARCH}")
endif()

# Compiler flags (no arch here - use CUDA_ARCHITECTURES)
if (WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options /O2")
endif()
if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2")
  if (NOT APPLE)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --compiler-options -O2,-D_FORCE_INLINES,-DVERBOSE_NOT")
  endif()
endif()

# Core AKAZE library (CUDA + C++)
add_library(akaze_core STATIC
  akaze.cpp
  akazed.cu
  fed.cpp
  akaze.h
  akazed.h
  akaze_structures.h
  cuda_utils.h
  fed.h
)
target_include_directories(akaze_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(akaze_core PUBLIC CUDA::cudart)
set_target_properties(akaze_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES "${CUDA_ARCH}"
)

# Python module (built as _cuda_akaze for cuda_akaze package)
pybind11_add_module(_cuda_akaze akaze_bindings.cpp)
target_link_libraries(_cuda_akaze PRIVATE akaze_core CUDA::cudart)

# Install to cuda_akaze package
install(TARGETS _cuda_akaze
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)
