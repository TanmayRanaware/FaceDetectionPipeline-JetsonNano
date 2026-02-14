cmake_minimum_required(VERSION 3.18)
project(jetson_face_detect LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include dirs
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# TensorRT paths (adjust if needed)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include/aarch64-linux-gnu /usr/include
)
find_library(TENSORRT_LIB nvinfer
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Install TensorRT dev packages and verify include/lib paths.")
endif()

# CUDA runtime
find_library(CUDA_RT cudart
  PATHS /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu
)
if(NOT CUDA_RT)
  message(FATAL_ERROR "CUDA runtime (cudart) not found.")
endif()

add_executable(jetson-face-detect
  src/main.cpp
  src/camera/V4L2Camera.cpp
  src/gpu/CudaUtils.cpp
  src/gpu/CudaPreprocessor.cu
  src/gpu/CudaOverlay.cu
  src/inference/TrtFaceDetector.cpp
  src/encoder/V4L2Encoder.cpp
  src/io/FileWriter.cpp
  src/pipeline/VideoPipeline.cpp
)

target_include_directories(jetson-face-detect PRIVATE ${TENSORRT_INCLUDE_DIR})

target_link_libraries(jetson-face-detect
  pthread
  ${TENSORRT_LIB}
  ${TENSORRT_PLUGIN_LIB}
  ${CUDA_RT}
)

# Optional: add -O3 and other flags
target_compile_options(jetson-face-detect PRIVATE -O3 -march=native)
target_compile_options(jetson-face-detect PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
