#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "common/Types.hpp"

class CudaOverlay {
public:
  void init(int w, int h);  // allocate device buffer for NV12 copy
  ~CudaOverlay();
  // nv12 is host pointer (MMAP); copies to device, draws, copies back
  void drawBoxesNV12(void* nv12, int w, int h, const std::vector<FaceBox>& faces, cudaStream_t stream);

private:
  void* d_nv12_ = nullptr;
  size_t buf_bytes_ = 0;
};
