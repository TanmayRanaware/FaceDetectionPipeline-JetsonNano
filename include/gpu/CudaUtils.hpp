#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

inline void cudaCheck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) throw std::runtime_error(msg);
}
