#pragma once

struct Config {
  dim3 threads = dim3(16, 16);
};
typedef struct Config Config;
