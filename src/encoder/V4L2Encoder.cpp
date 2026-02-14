#include "encoder/V4L2Encoder.hpp"
#include "common/Logger.hpp"
#include "common/Timer.hpp"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

static void throw_sys(const char* msg) { throw std::runtime_error(msg); }

void V4L2Encoder::xioctl(unsigned long req, void* arg) {
  if (ioctl(fd_, req, arg) < 0) throw_sys("encoder ioctl failed");
}

void V4L2Encoder::setControl(uint32_t id, int32_t value) {
  v4l2_control c{};
  c.id = id;
  c.value = value;
  ioctl(fd_, VIDIOC_S_CTRL, &c); // best-effort (diff jetpack)
}

V4L2Encoder::~V4L2Encoder() {
  try { stop(); } catch(...) {}
  if (fd_ >= 0) ::close(fd_);
}

void V4L2Encoder::openDevice(const std::string& dev) {
  fd_ = ::open(dev.c_str(), O_RDWR | O_NONBLOCK, 0);
  if (fd_ < 0) throw_sys("Failed to open encoder device");
  LOGI("Opened encoder: %s", dev.c_str());
}

void V4L2Encoder::configure(int w, int h, int bitrate, int fps) {
  w_ = w; h_ = h; bitrate_ = bitrate; fps_ = fps;

  v4l2_format out_fmt{};
  out_fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  out_fmt.fmt.pix.width = w_;
  out_fmt.fmt.pix.height = h_;
  out_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
  out_fmt.fmt.pix.field = V4L2_FIELD_NONE;
  xioctl(VIDIOC_S_FMT, &out_fmt);

  v4l2_format cap_fmt{};
  cap_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cap_fmt.fmt.pix.width = w_;
  cap_fmt.fmt.pix.height = h_;
  cap_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_H264;
  cap_fmt.fmt.pix.field = V4L2_FIELD_NONE;
  xioctl(VIDIOC_S_FMT, &cap_fmt);

  setControl(V4L2_CID_MPEG_VIDEO_BITRATE, bitrate_);
  setControl(V4L2_CID_MPEG_VIDEO_GOP_SIZE, fps_);
  setControl(V4L2_CID_MPEG_VIDEO_B_FRAMES, 0);

  LOGI("Encoder configured: %dx%d bitrate=%d fps=%d", w_, h_, bitrate_, fps_);
}

void V4L2Encoder::initBuffers(int out_count, int cap_count) {
  v4l2_requestbuffers cap_req{};
  cap_req.count = cap_count;
  cap_req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cap_req.memory = V4L2_MEMORY_MMAP;
  xioctl(VIDIOC_REQBUFS, &cap_req);

  cap_.resize(cap_req.count);

  for (int i = 0; i < (int)cap_req.count; i++) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = i;
    xioctl(VIDIOC_QUERYBUF, &b);

    void* ptr = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
    if (ptr == MAP_FAILED) throw_sys("encoder capture mmap failed");

    cap_[i].ptr = ptr;
    cap_[i].len = b.length;

    xioctl(VIDIOC_QBUF, &b);
    cap_qd_++;
  }

  v4l2_requestbuffers out_req{};
  out_req.count = out_count;
  out_req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  out_req.memory = V4L2_MEMORY_USERPTR;
  xioctl(VIDIOC_REQBUFS, &out_req);

  LOGI("Encoder buffers initialized: out=%d cap=%zu", out_count, cap_.size());
}

void V4L2Encoder::start() {
  int cap_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  int out_type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  xioctl(VIDIOC_STREAMON, &cap_type);
  xioctl(VIDIOC_STREAMON, &out_type);
  LOGI("Encoder STREAMON");
}

void V4L2Encoder::queueFrameNV12(const FrameRef& f) {
  v4l2_buffer b{};
  b.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  b.memory = V4L2_MEMORY_USERPTR;
  b.m.userptr = (unsigned long)f.cam_ptr;
  b.length = (uint32_t)f.cam_bytes;
  b.bytesused = (uint32_t)(f.width * f.height * 3 / 2);
  xioctl(VIDIOC_QBUF, &b);
}

bool V4L2Encoder::dequeueBitstream(BitstreamPacket& pkt) {
  v4l2_buffer b{};
  b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  b.memory = V4L2_MEMORY_MMAP;

  if (ioctl(fd_, VIDIOC_DQBUF, &b) != 0) {
    return false;
  }

  last_cap_index_ = b.index;
  pkt.ptr = (const uint8_t*)cap_[b.index].ptr;
  pkt.bytes = b.bytesused;
  pkt.pts_ns = now_ns();
  return true;
}

void V4L2Encoder::releaseCaptureBuffer() {
  if (last_cap_index_ < 0) return;
  v4l2_buffer b{};
  b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  b.memory = V4L2_MEMORY_MMAP;
  b.index = last_cap_index_;
  xioctl(VIDIOC_QBUF, &b);
  last_cap_index_ = -1;
}

void V4L2Encoder::stop() {
  if (fd_ < 0) return;
  int cap_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  int out_type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  ioctl(fd_, VIDIOC_STREAMOFF, &cap_type);
  ioctl(fd_, VIDIOC_STREAMOFF, &out_type);

  for (auto& c : cap_) {
    if (c.ptr) munmap(c.ptr, c.len);
    c.ptr = nullptr;
  }
  cap_.clear();
  last_cap_index_ = -1;
}
