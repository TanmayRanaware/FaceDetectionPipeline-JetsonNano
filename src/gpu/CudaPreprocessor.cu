#include "gpu/CudaPreprocessor.hpp"
#include "gpu/CudaUtils.hpp"
#include <cuda_runtime.h>

// NOTE: This assumes nv12 is host memory (MMAP). We copy host->device then run kernel.
// For best perf on Jetson, use DMABUF/EGL interop for zero-copy later.

__device__ __forceinline__ float clamp01(float x) {
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

__global__ void nv12_to_chw_kernel(const uint8_t* nv12, int in_w, int in_h,
                                  float* out, int out_w, int out_h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;

  int src_x = (x * in_w) / out_w;
  int src_y = (y * in_h) / out_h;

  const uint8_t* y_plane = nv12;
  const uint8_t* uv_plane = nv12 + in_w * in_h;

  int Y = y_plane[src_y * in_w + src_x];

  int uv_x = (src_x / 2) * 2;
  int uv_y = (src_y / 2);
  int uv_idx = uv_y * in_w + uv_x;
  int U = uv_plane[uv_idx + 0];
  int V = uv_plane[uv_idx + 1];

  // YUV -> RGB (BT.601 approx)
  float fY = (float)Y;
  float fU = (float)U - 128.f;
  float fV = (float)V - 128.f;

  float R = fY + 1.402f * fV;
  float G = fY - 0.344136f * fU - 0.714136f * fV;
  float B = fY + 1.772f * fU;

  R = clamp01(R / 255.f);
  G = clamp01(G / 255.f);
  B = clamp01(B / 255.f);

  int idx = y * out_w + x;
  out[idx] = R;
  out[out_w * out_h + idx] = G;
  out[2 * out_w * out_h + idx] = B;
}

CudaPreprocessor::~CudaPreprocessor() {
  if (d_nv12_) cudaFree(d_nv12_);
  d_nv12_ = nullptr;
}

void CudaPreprocessor::init(int in_w, int in_h, int out_w, int out_h) {
  in_w_ = in_w; in_h_ = in_h;
  out_w_ = out_w; out_h_ = out_h;

  if (d_nv12_) {
    cudaFree(d_nv12_);
    d_nv12_ = nullptr;
  }
  size_t nv12_bytes = (size_t)in_w * in_h * 3 / 2;
  cudaCheck(cudaMalloc(&d_nv12_, nv12_bytes), "cudaMalloc d_nv12 failed");
}

void CudaPreprocessor::preprocessNV12ToCHW(const void* nv12, int in_w, int in_h,
                                           float* d_out_chw, int out_w, int out_h,
                                           cudaStream_t stream) {
  size_t nv12_bytes = (size_t)in_w * in_h * 3 / 2;
  cudaCheck(cudaMemcpyAsync(d_nv12_, nv12, nv12_bytes, cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsync nv12 failed");

  dim3 block(16, 16);
  dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y);
  nv12_to_chw_kernel<<<grid, block, 0, stream>>>(
      (const uint8_t*)d_nv12_, in_w, in_h, d_out_chw, out_w, out_h);
}
