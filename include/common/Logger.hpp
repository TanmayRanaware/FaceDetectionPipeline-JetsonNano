#pragma once
#include <cstdio>

#define LOGI(fmt, ...) std::fprintf(stdout, "[I] " fmt "\n", ##__VA_ARGS__)
#define LOGW(fmt, ...) std::fprintf(stdout, "[W] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) std::fprintf(stderr, "[E] " fmt "\n", ##__VA_ARGS__)
