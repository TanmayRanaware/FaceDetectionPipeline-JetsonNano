#include "pipeline/VideoPipeline.hpp"
#include "common/Logger.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

static VideoPipeline* g_pipeline = nullptr;
static std::atomic<bool> g_done{false};

static void sig_handler(int) {
  LOGI("Caught signal, stopping...");
  g_done = true;
  if (g_pipeline) g_pipeline->stop();
}

static PipelineConfig parse(int argc, char** argv) {
  PipelineConfig cfg;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char* k){
      if (i + 1 >= argc) { std::cerr << "Missing value for " << k << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };
    if (a == "--engine") cfg.engine_path = need("--engine");
    else if (a == "--out") cfg.out_path = need("--out");
    else if (a == "--w") cfg.w = std::stoi(need("--w"));
    else if (a == "--h") cfg.h = std::stoi(need("--h"));
    else if (a == "--fps") cfg.fps = std::stoi(need("--fps"));
    else if (a == "--bitrate") cfg.bitrate = std::stoi(need("--bitrate"));
    else if (a == "--cam") cfg.cam_dev = need("--cam");
    else if (a == "--enc") cfg.enc_dev = need("--enc");
  }
  if (cfg.engine_path.empty()) {
    std::cerr << "Usage: --engine models/face_detector.engine [--out out.h264] [--w 1280 --h 720 --fps 30]\n";
    std::exit(1);
  }
  return cfg;
}

int main(int argc, char** argv) {
  try {
    PipelineConfig cfg = parse(argc, argv);
    VideoPipeline p(cfg);
    g_pipeline = &p;

    std::signal(SIGINT, sig_handler);
    std::signal(SIGTERM, sig_handler);

    p.init();
    p.start();

    LOGI("Running... press Ctrl+C to stop");
    while (!g_done) { sleep(1); }

  } catch (const std::exception& e) {
    LOGE("Fatal: %s", e.what());
    return 1;
  }
  return 0;
}
