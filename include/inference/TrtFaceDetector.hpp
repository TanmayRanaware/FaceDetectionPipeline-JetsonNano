#pragma once
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "common/Types.hpp"

namespace nvinfer1 { class IRuntime; class ICudaEngine; class IExecutionContext; }

class TrtFaceDetector {
public:
  ~TrtFaceDetector();

  void loadEngine(const std::string& engine_path);
  void setInputDims(int c, int h, int w);
  void allocate(); // alloc device buffers

  // Input is CHW float on GPU (d_input_)
  void infer(cudaStream_t stream);
  std::vector<FaceBox> postprocessCPU(float conf_thresh);

  float* dInput() { return d_input_; }

private:
  std::vector<char> readFile(const std::string& path);

  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* ctx_ = nullptr;

  int in_c_=3, in_h_=320, in_w_=320;

  // bindings
  std::vector<void*> bindings_;
  int input_index_ = -1;
  int output_index_ = -1;

  float* d_input_ = nullptr;
  float* d_output_ = nullptr;
  size_t d_input_bytes_ = 0;
  size_t d_output_bytes_ = 0;

  // host output for postprocess
  std::vector<float> h_output_;
};
