#include "io/FileWriter.hpp"
#include "common/Logger.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

static void throw_sys(const char* msg) { throw std::runtime_error(msg); }

FileWriter::~FileWriter() {
  if (fd_ >= 0) ::close(fd_);
}

void FileWriter::open(const std::string& path) {
  fd_ = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd_ < 0) throw_sys("Failed to open output file");
  LOGI("Writing bitstream to: %s", path.c_str());
}

void FileWriter::write(const BitstreamPacket& pkt) {
  const uint8_t* p = pkt.ptr;
  size_t left = pkt.bytes;
  while (left) {
    ssize_t n = ::write(fd_, p, left);
    if (n <= 0) throw_sys("write failed");
    p += (size_t)n;
    left -= (size_t)n;
  }
}
