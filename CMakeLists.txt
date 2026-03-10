cmake_minimum_required(VERSION 3.18)

project(CudaAkaze LANGUAGES CXX CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
find_package(pybind11 CONFIG QUIET)
if(NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif()

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

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(cuda_sources
  ${SRC_DIR}/akazed.cu
  ${SRC_DIR}/akazed.h
  ${SRC_DIR}/akaze.cpp
  ${SRC_DIR}/akaze.h
  ${SRC_DIR}/akaze_structures.h
  ${SRC_DIR}/cuda_utils.h
  ${SRC_DIR}/fed.cpp
  ${SRC_DIR}/fed.h
)

set(sources
  ${SRC_DIR}/main.cpp
)

include_directories(${SRC_DIR})

cuda_add_executable(cuakaze ${cuda_sources} ${sources} OPTIONS -arch=sm_75)

set_target_properties(cuakaze PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cuakaze ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS})

# Python module (pybind11 + scikit-build)
if(pybind11_FOUND AND CUDA_FOUND)
  cuda_add_library(akaze_core STATIC ${SRC_DIR}/akazed.cu ${SRC_DIR}/akaze.cpp ${SRC_DIR}/fed.cpp OPTIONS -arch=sm_75)
  set_target_properties(akaze_core PROPERTIES COMPILE_FLAGS "${EXTRA_CXX_FLAGS}")
  target_include_directories(akaze_core PUBLIC ${SRC_DIR})

  pybind11_add_module(_cuda_akaze ${SRC_DIR}/akaze_bindings.cpp)
  target_link_libraries(_cuda_akaze PRIVATE akaze_core)
  target_include_directories(_cuda_akaze PRIVATE ${SRC_DIR})
  install(TARGETS _cuda_akaze LIBRARY DESTINATION . RUNTIME DESTINATION . ARCHIVE DESTINATION .)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_akaze/__init__.py DESTINATION .)
endif()