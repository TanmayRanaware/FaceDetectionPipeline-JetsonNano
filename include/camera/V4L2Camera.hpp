#pragma once
#include <string>
#include <vector>
#include <linux/videodev2.h>
#include "common/Types.hpp"

class V4L2Camera {
public:
  ~V4L2Camera();
  void openDevice(const std::string& dev);
  void configure(int w, int h, int fps);
  void initMmapBuffers(int count);
  void start();
  FrameRef dequeue();
  void enqueue(int index);
  void stop();

  int width() const { return w_; }
  int height() const { return h_; }

private:
  int fd_ = -1;
  int w_ = 0, h_ = 0, fps_ = 0;

  struct Buf {
    void* ptr = nullptr;
    size_t len = 0;
    int dmabuf_fd = -1; // optional
  };
  std::vector<Buf> bufs_;

  void xioctl(unsigned long req, void* arg);
  int exportDmabufFd(int index);
};
