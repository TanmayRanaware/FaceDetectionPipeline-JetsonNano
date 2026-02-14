#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include "common/Types.hpp"

class FileWriter {
public:
  ~FileWriter();
  void open(const std::string& path);
  void write(const BitstreamPacket& pkt);

private:
  int fd_ = -1;
};
