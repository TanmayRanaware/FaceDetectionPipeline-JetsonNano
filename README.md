# Jetson Face Detect (V4L2 + CUDA + TensorRT + NVENC)

Low-latency pipeline:
Camera (CSI) -> ISP -> DMA -> V4L2 -> CUDA preprocess -> TensorRT detect -> CUDA overlay -> V4L2 M2M encoder (H264) -> file

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Run

```bash
./jetson-face-detect --engine ../models/face_detector.engine --out out.h264 --w 1280 --h 720 --fps 30
```

## Notes

- Capture uses V4L2 MMAP for speed.
- DMABUF export via VIDIOC_EXPBUF is included; CUDA zero-copy import requires EGL/CUDA interop (add later).
- Encoder uses V4L2 M2M template; device node/controls may vary across JetPack versions. Common encoder nodes: `/dev/nvhost-msenc`, `/dev/video1` (check `v4l2-ctl --list-devices` on your Jetson).
