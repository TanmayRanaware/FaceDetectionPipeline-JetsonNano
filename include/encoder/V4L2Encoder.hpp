#pragma once
#include <string>
#include <vector>
#include <linux/videodev2.h>
#include "common/Types.hpp"

class V4L2Encoder {
public:
  ~V4L2Encoder();

  void openDevice(const std::string& dev);
  void configure(int w, int h, int bitrate, int fps);
  void initBuffers(int out_count, int cap_count);
  void start();

  // feed raw NV12 frame (MMAP pointer). For true zero-copy: use DMABUF with V4L2_MEMORY_DMABUF.
  void queueFrameNV12(const FrameRef& f);
  bool dequeueBitstream(BitstreamPacket& pkt);
  void releaseCaptureBuffer();  // call after writing pkt; re-queues capture buffer

  void stop();

private:
  int fd_ = -1;
  int w_ = 0, h_ = 0, bitrate_ = 0, fps_ = 0;

  struct CapBuf {
    void* ptr = nullptr;
    size_t len = 0;
  };
  std::vector<CapBuf> cap_;
  int cap_qd_ = 0;
  int last_cap_index_ = -1;  // index of last dequeued capture buffer

  void xioctl(unsigned long req, void* arg);
  void setControl(uint32_t id, int32_t value);
};
