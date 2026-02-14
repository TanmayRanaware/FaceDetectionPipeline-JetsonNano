#include "camera/V4L2Camera.hpp"
#include "common/Logger.hpp"
#include "common/Timer.hpp"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

static void throw_sys(const char* msg) { throw std::runtime_error(msg); }

void V4L2Camera::xioctl(unsigned long req, void* arg) {
  if (ioctl(fd_, req, arg) < 0) throw_sys("ioctl failed");
}

V4L2Camera::~V4L2Camera() {
  try { stop(); } catch(...) {}
  if (fd_ >= 0) ::close(fd_);
}

void V4L2Camera::openDevice(const std::string& dev) {
  fd_ = ::open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd_ < 0) throw_sys("Failed to open camera device");

  v4l2_capability cap{};
  xioctl(VIDIOC_QUERYCAP, &cap);
  LOGI("Opened camera: %s", dev.c_str());
}

void V4L2Camera::configure(int w, int h, int fps) {
  w_ = w; h_ = h; fps_ = fps;

  v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = w_;
  fmt.fmt.pix.height = h_;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  xioctl(VIDIOC_S_FMT, &fmt);

  v4l2_streamparm parm{};
  parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  parm.parm.capture.timeperframe.numerator = 1;
  parm.parm.capture.timeperframe.denominator = fps_;
  xioctl(VIDIOC_S_PARM, &parm);

  LOGI("Camera configured: %dx%d@%d NV12", w_, h_, fps_);
}

int V4L2Camera::exportDmabufFd(int index) {
  v4l2_exportbuffer exp{};
  exp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  exp.index = index;
  exp.plane = 0;
  exp.flags = O_CLOEXEC;
  if (ioctl(fd_, VIDIOC_EXPBUF, &exp) < 0) {
    return -1; // not supported or failed
  }
  return exp.fd;
}

void V4L2Camera::initMmapBuffers(int count) {
  v4l2_requestbuffers req{};
  req.count = count;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  xioctl(VIDIOC_REQBUFS, &req);

  bufs_.resize(req.count);

  for (int i = 0; i < (int)req.count; i++) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    xioctl(VIDIOC_QUERYBUF, &buf);

    void* ptr = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (ptr == MAP_FAILED) throw_sys("mmap failed");

    bufs_[i].ptr = ptr;
    bufs_[i].len = buf.length;
    bufs_[i].dmabuf_fd = exportDmabufFd(i);

    // Queue all initially
    xioctl(VIDIOC_QBUF, &buf);
  }

  LOGI("MMAP buffers initialized: %zu", bufs_.size());
}

void V4L2Camera::start() {
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  xioctl(VIDIOC_STREAMON, &type);
  LOGI("Camera STREAMON");
}

FrameRef V4L2Camera::dequeue() {
  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;

  // Non-blocking device; poll-style: retry until frame available
  while (true) {
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) == 0) break;
    usleep(100); // tiny sleep (low latency). Could use poll() for better.
  }

  FrameRef f;
  f.cam_index = (int)buf.index;
  f.cam_ptr = bufs_[buf.index].ptr;
  f.cam_bytes = bufs_[buf.index].len;
  f.dmabuf_fd = bufs_[buf.index].dmabuf_fd;
  f.width = w_;
  f.height = h_;
  f.ts_ns = now_ns();
  f.faces.reserve(16);
  return f;
}

void V4L2Camera::enqueue(int index) {
  v4l2_buffer buf{};
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  buf.index = index;
  xioctl(VIDIOC_QBUF, &buf);
}

void V4L2Camera::stop() {
  if (fd_ < 0) return;
  int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  ioctl(fd_, VIDIOC_STREAMOFF, &type);

  for (auto& b : bufs_) {
    if (b.ptr) munmap(b.ptr, b.len);
    b.ptr = nullptr;
    if (b.dmabuf_fd >= 0) ::close(b.dmabuf_fd);
    b.dmabuf_fd = -1;
  }
  bufs_.clear();
}
