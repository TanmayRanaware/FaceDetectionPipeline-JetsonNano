#pragma once
#include <cuda_runtime.h>

class CudaPreprocessor {
public:
  void init(int in_w, int in_h, int out_w, int out_h);
  // NV12 (Y + UV) - nv12 can be host pointer (MMAP); copies to device then preprocesses.
  void preprocessNV12ToCHW(const void* nv12, int in_w, int in_h,
                           float* d_out_chw, int out_w, int out_h,
                           cudaStream_t stream);

  ~CudaPreprocessor();

private:
  int in_w_ = 0, in_h_ = 0;
  int out_w_ = 0, out_h_ = 0;
  void* d_nv12_ = nullptr;  // device buffer for host->device copy
};
