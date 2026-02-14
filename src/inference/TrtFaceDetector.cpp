#include "inference/TrtFaceDetector.hpp"
#include "common/Logger.hpp"
#include "gpu/CudaUtils.hpp"

#include <fstream>
#include <stdexcept>
#include <NvInfer.h>

using namespace nvinfer1;

namespace {
class TrtLogger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) LOGW("TRT: %s", msg);
  }
} gLogger;
}

static void throw_rt(const char* msg) { throw std::runtime_error(msg); }

std::vector<char> TrtFaceDetector::readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw_rt("Failed to open engine file");
  f.seekg(0, std::ios::end);
  size_t sz = (size_t)f.tellg();
  f.seekg(0);
  std::vector<char> buf(sz);
  f.read(buf.data(), sz);
  return buf;
}

TrtFaceDetector::~TrtFaceDetector() {
  if (d_input_) cudaFree(d_input_);
  if (d_output_) cudaFree(d_output_);
  if (ctx_) ctx_->destroy();
  if (engine_) engine_->destroy();
  if (runtime_) runtime_->destroy();
}

void TrtFaceDetector::loadEngine(const std::string& engine_path) {
  auto blob = readFile(engine_path);
  runtime_ = createInferRuntime(gLogger);
  if (!runtime_) throw_rt("createInferRuntime failed");

  engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
  if (!engine_) throw_rt("deserializeCudaEngine failed");

  ctx_ = engine_->createExecutionContext();
  if (!ctx_) throw_rt("createExecutionContext failed");

  int nb = engine_->getNbBindings();
  for (int i = 0; i < nb; i++) {
    if (engine_->bindingIsInput(i)) input_index_ = i;
    else output_index_ = i;
  }
  if (input_index_ < 0 || output_index_ < 0) throw_rt("Could not find input/output bindings");

  LOGI("TensorRT engine loaded. bindings=%d input=%d output=%d", nb, input_index_, output_index_);
}

void TrtFaceDetector::setInputDims(int c, int h, int w) {
  in_c_ = c; in_h_ = h; in_w_ = w;
}

void TrtFaceDetector::allocate() {
  d_input_bytes_ = (size_t)in_c_ * in_h_ * in_w_ * sizeof(float);
  cudaCheck(cudaMalloc(&d_input_, d_input_bytes_), "cudaMalloc input failed");

  size_t out_floats = 6000;
  d_output_bytes_ = out_floats * sizeof(float);
  cudaCheck(cudaMalloc(&d_output_, d_output_bytes_), "cudaMalloc output failed");

  bindings_.resize(engine_->getNbBindings(), nullptr);
  bindings_[input_index_] = d_input_;
  bindings_[output_index_] = d_output_;

  h_output_.resize(out_floats);
  LOGI("Allocated TRT buffers: input=%zu bytes output=%zu bytes", d_input_bytes_, d_output_bytes_);
}

void TrtFaceDetector::infer(cudaStream_t stream) {
  bool ok = ctx_->enqueueV2(bindings_.data(), stream, nullptr);
  if (!ok) throw_rt("enqueueV2 failed");
}

std::vector<FaceBox> TrtFaceDetector::postprocessCPU(float conf_thresh) {
  cudaCheck(cudaMemcpy(h_output_.data(), d_output_, d_output_bytes_, cudaMemcpyDeviceToHost),
            "cudaMemcpy output failed");

  std::vector<FaceBox> faces;
  for (size_t i = 0; i + 5 < h_output_.size(); i += 6) {
    float score = h_output_[i + 4];
    if (score < conf_thresh) continue;
    FaceBox b;
    b.x1 = h_output_[i + 0];
    b.y1 = h_output_[i + 1];
    b.x2 = h_output_[i + 2];
    b.y2 = h_output_[i + 3];
    b.score = score;
    faces.push_back(b);
    if (faces.size() >= 16) break;
  }
  return faces;
}
