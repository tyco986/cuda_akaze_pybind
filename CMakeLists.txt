cmake_minimum_required(VERSION 3.18)

project(CudaAkaze LANGUAGES CXX CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(pybind11 CONFIG QUIET)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") # -arch=sm_35;
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT;-Xcompiler=-fPIC")
  endif()
endif()

set(cuda_sources
  akazed.cu
  akazed.h
  akaze.cpp
  akaze.h
  akaze_structures.h
  cuda_utils.h
  fed.cpp
  fed.h
)  

set(sources
  main.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

#SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuakaze ${cuda_sources} ${sources} OPTIONS -arch=sm_75)

set_target_properties(cuakaze PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cuakaze ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Reproducibility test executable (single run, outputs to file)
set(repro_sources repro_test.cpp)
cuda_add_executable(repro_test ${cuda_sources} ${repro_sources} OPTIONS -arch=sm_75)
set_target_properties(repro_test PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
target_link_libraries(repro_test ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Standalone gHammingMatch test - loads saved inputs, runs kernel only
cuda_add_executable(test_gHammingMatch test_gHammingMatch.cu cuda_utils.h akaze_structures.h OPTIONS -arch=sm_75)
set_target_properties(test_gHammingMatch PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
target_include_directories(test_gHammingMatch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_gHammingMatch ${CUDA_cudadevrt_LIBRARY})

# Debug build option: -DCMAKE_BUILD_TYPE=Debug for GDB

# Python module (pybind11 + scikit-build)
if(pybind11_FOUND AND CUDA_FOUND)
  cuda_add_library(akaze_core STATIC akazed.cu akaze.cpp fed.cpp OPTIONS -arch=sm_75)
  set_target_properties(akaze_core PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
  target_include_directories(akaze_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  pybind11_add_module(_cuda_akaze akaze_bindings.cpp)
  target_link_libraries(_cuda_akaze PRIVATE akaze_core)
  target_include_directories(_cuda_akaze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS _cuda_akaze LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_akaze/__init__.py DESTINATION .)
endif()