#include "gpu/CudaOverlay.hpp"
#include "gpu/CudaUtils.hpp"
#include <cuda_runtime.h>

// Draw rectangle on NV12 by modifying Y plane only (bright lines).
__global__ void draw_rect_yplane(uint8_t* yplane, int w, int h,
                                 int x1, int y1, int x2, int y2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;

  bool on_border = (y == y1 || y == y2) && (x >= x1 && x <= x2);
  on_border = on_border || ((x == x1 || x == x2) && (y >= y1 && y <= y2));

  if (on_border) {
    yplane[y * w + x] = 235; // bright
  }
}

void CudaOverlay::init(int w, int h) {
  if (d_nv12_) cudaFree(d_nv12_);
  buf_bytes_ = (size_t)w * h * 3 / 2;
  cudaCheck(cudaMalloc(&d_nv12_, buf_bytes_), "cudaMalloc overlay d_nv12 failed");
}

CudaOverlay::~CudaOverlay() {
  if (d_nv12_) cudaFree(d_nv12_);
  d_nv12_ = nullptr;
}

void CudaOverlay::drawBoxesNV12(void* nv12, int w, int h, const std::vector<FaceBox>& faces, cudaStream_t stream) {
  size_t nv12_bytes = (size_t)w * h * 3 / 2;
  if (nv12_bytes > buf_bytes_ || !d_nv12_) init(w, h);

  cudaCheck(cudaMemcpyAsync(d_nv12_, nv12, nv12_bytes, cudaMemcpyHostToDevice, stream),
            "overlay cudaMemcpy H2D failed");

  uint8_t* yplane = (uint8_t*)d_nv12_;
  dim3 block(16, 16);
  dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

  for (const auto& b : faces) {
    int x1 = (int)b.x1, y1 = (int)b.y1, x2 = (int)b.x2, y2 = (int)b.y2;
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 >= w) x2 = w - 1; if (y2 >= h) y2 = h - 1;
    draw_rect_yplane<<<grid, block, 0, stream>>>(yplane, w, h, x1, y1, x2, y2);
  }

  cudaCheck(cudaMemcpyAsync(nv12, d_nv12_, nv12_bytes, cudaMemcpyDeviceToHost, stream),
            "overlay cudaMemcpy D2H failed");
}
