#pragma once
#include <cstdint>
#include <vector>

struct FaceBox {
  float x1, y1, x2, y2;
  float score;
};

struct FrameRef {
  int cam_index = -1;         // V4L2 camera buffer index
  void* cam_ptr = nullptr;    // MMAP pointer (NV12)
  size_t cam_bytes = 0;
  int dmabuf_fd = -1;         // exported via VIDIOC_EXPBUF (optional)
  int width = 0;
  int height = 0;
  uint64_t ts_ns = 0;
  std::vector<FaceBox> faces; // small metadata; reserve once
};

struct BitstreamPacket {
  const uint8_t* ptr = nullptr; // points to mmap'd encoder capture buffer
  size_t bytes = 0;
  uint64_t pts_ns = 0;
  bool keyframe = false;
};
