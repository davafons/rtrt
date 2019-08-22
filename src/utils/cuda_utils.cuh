#pragma once

#include <iostream>
#include <string>

#include <algorithm>
#include <utility>

#define cudaCheckErr(ans)                                                      \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert" << cudaGetErrorString(code) << " " << file << " "
              << std::to_string(line) << std::endl;
    if (abort)
      exit(code);
  }
}

template <class T> T *cuda_malloc_managed(size_t size_in_bytes) {
  T *p = NULL;
  cudaCheckErr(cudaMallocManaged(&p, size_in_bytes));

  return p;
}

template <class T> T *cuda_malloc(size_t size_in_bytes) {
  T *p = NULL;
  cudaCheckErr(cudaMalloc(&p, size_in_bytes));

  return p;
}
