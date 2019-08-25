#pragma once

struct Config {
  dim3 threads = dim3(16, 16);

  dim3 blocks(int w, int h) const { return dim3(w / threads.x + 1, h / threads.y + 1); }
};
typedef struct Config Config;
