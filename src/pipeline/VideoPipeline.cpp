#include "pipeline/VideoPipeline.hpp"
#include "common/Logger.hpp"
#include "common/Timer.hpp"
#include "gpu/CudaUtils.hpp"
#include <unistd.h>

VideoPipeline::VideoPipeline(const PipelineConfig& cfg)
: cfg_(cfg), q_cap_(2), q_gpu_(2) {}

void VideoPipeline::init() {
  cam_.openDevice(cfg_.cam_dev);
  cam_.configure(cfg_.w, cfg_.h, cfg_.fps);
  cam_.initMmapBuffers(cfg_.cam_bufs);
  cam_.start();

  cudaCheck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate failed");

  trt_.loadEngine(cfg_.engine_path);
  trt_.setInputDims(3, cfg_.trt_h, cfg_.trt_w);
  trt_.allocate();

  pre_.init(cfg_.w, cfg_.h, cfg_.trt_w, cfg_.trt_h);
  ov_.init(cfg_.w, cfg_.h);

  enc_.openDevice(cfg_.enc_dev);
  enc_.configure(cfg_.w, cfg_.h, cfg_.bitrate, cfg_.fps);
  enc_.initBuffers(2, 4);
  enc_.start();

  writer_.open(cfg_.out_path);

  LOGI("Pipeline initialized");
}

void VideoPipeline::start() {
  run_ = true;
  t_cap_ = std::thread(&VideoPipeline::captureLoop, this);
  t_gpu_ = std::thread(&VideoPipeline::gpuLoop, this);
  t_enc_ = std::thread(&VideoPipeline::encodeLoop, this);
}

void VideoPipeline::stop() {
  run_ = false;
  q_cap_.stop();
  q_gpu_.stop();

  if (t_cap_.joinable()) t_cap_.join();
  if (t_gpu_.joinable()) t_gpu_.join();
  if (t_enc_.joinable()) t_enc_.join();

  if (stream_) cudaStreamDestroy(stream_);
  cam_.stop();
  enc_.stop();
  LOGI("Pipeline stopped");
}

void VideoPipeline::captureLoop() {
  while (run_) {
    FrameRef f = cam_.dequeue();
    q_cap_.push(std::move(f));
  }
}

void VideoPipeline::gpuLoop() {
  float sx = (float)cfg_.w / cfg_.trt_w;
  float sy = (float)cfg_.h / cfg_.trt_h;

  while (run_) {
    FrameRef f;
    if (!q_cap_.pop(f)) break;

    pre_.preprocessNV12ToCHW(f.cam_ptr, f.width, f.height,
                             trt_.dInput(), cfg_.trt_w, cfg_.trt_h, stream_);

    trt_.infer(stream_);

    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");

    f.faces = trt_.postprocessCPU(cfg_.conf_thresh);

    // Scale face boxes from model input space to frame size
    for (auto& b : f.faces) {
      b.x1 *= sx; b.y1 *= sy; b.x2 *= sx; b.y2 *= sy;
    }

    ov_.drawBoxesNV12(f.cam_ptr, f.width, f.height, f.faces, stream_);
    cudaCheck(cudaStreamSynchronize(stream_), "overlay sync failed");

    q_gpu_.push(std::move(f));
  }
}

void VideoPipeline::encodeLoop() {
  uint64_t last_log = now_ns();
  int frames = 0;

  while (run_) {
    FrameRef f;
    if (!q_gpu_.pop(f)) break;

    enc_.queueFrameNV12(f);

    BitstreamPacket pkt{};
    while (enc_.dequeueBitstream(pkt)) {
      writer_.write(pkt);
      enc_.releaseCaptureBuffer();
    }

    cam_.enqueue(f.cam_index);

    frames++;
    uint64_t t = now_ns();
    if (t - last_log > 1000000000ull) {
      LOGI("FPS approx: %d", frames);
      frames = 0;
      last_log = t;
    }
  }
}
