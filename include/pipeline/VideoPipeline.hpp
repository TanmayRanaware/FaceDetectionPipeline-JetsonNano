#pragma once
#include <atomic>
#include <thread>
#include <string>
#include <cuda_runtime.h>

#include "common/ThreadSafeQueue.hpp"
#include "common/Types.hpp"
#include "camera/V4L2Camera.hpp"
#include "gpu/CudaPreprocessor.hpp"
#include "gpu/CudaOverlay.hpp"
#include "inference/TrtFaceDetector.hpp"
#include "encoder/V4L2Encoder.hpp"
#include "io/FileWriter.hpp"

struct PipelineConfig {
  std::string cam_dev = "/dev/video0";
  std::string enc_dev = "/dev/nvhost-msenc"; // may vary per JetPack
  std::string engine_path;
  std::string out_path = "out.h264";
  int w = 1280;
  int h = 720;
  int fps = 30;
  int bitrate = 4000000;
  int cam_bufs = 3;      // 2-3 low latency
  int trt_w = 320;       // model input
  int trt_h = 320;
  float conf_thresh = 0.6f;
};

class VideoPipeline {
public:
  explicit VideoPipeline(const PipelineConfig& cfg);

  void init();
  void start();
  void stop();

private:
  void captureLoop();
  void gpuLoop();
  void encodeLoop();

  PipelineConfig cfg_;
  std::atomic<bool> run_{false};

  // Components
  V4L2Camera cam_;
  CudaPreprocessor pre_;
  TrtFaceDetector trt_;
  CudaOverlay ov_;
  V4L2Encoder enc_;
  FileWriter writer_;

  // Queues (bounded => prevents latency explosion)
  ThreadSafeQueue<FrameRef> q_cap_{2};
  ThreadSafeQueue<FrameRef> q_gpu_{2};

  // Threads
  std::thread t_cap_;
  std::thread t_gpu_;
  std::thread t_enc_;

  // CUDA
  cudaStream_t stream_{};
};
