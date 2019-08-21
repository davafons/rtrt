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

class Managed {
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaCheckErr(cudaMallocManaged(&ptr, len));
    return ptr;
  }

  void operator delete(void *ptr) { cudaCheckErr(cudaFree(ptr)); }
};

template <class T> class managed_ptr {
public:
  explicit managed_ptr(T *data, void *mem) : data_(data), mem_(mem) {}
  ~managed_ptr() {
    data_->~T();
    cudaCheckErr(cudaFree(mem_));
  }

  managed_ptr(const managed_ptr &) = delete;
  managed_ptr &operator=(const managed_ptr &) = delete;

  managed_ptr(managed_ptr &&moving) noexcept { moving.swap(*this); }
  managed_ptr &operator=(managed_ptr &&moving) noexcept {
    moving.swap(*this);

    return *this;
  }

  T *operator->() const { return data_; }
  T &operator*() const { return *data_; }

  T *get() const { return data_; }
  explicit operator bool() const { return data_; }

  void swap(managed_ptr &rhs) noexcept {
    std::swap(data_, rhs.data_);
    std::swap(mem_, rhs.mem_);
  }

private:
  T *data_ = NULL;
  void *mem_ = NULL;
};

template <class T> void swap(managed_ptr<T> &lhs, managed_ptr<T> &rhs) {
  lhs.swap(rhs);
}

template <class T, typename... Args>
managed_ptr<T> make_managed(Args &&... args) {
  void *mem = cuda_malloc_managed<T>(sizeof(T));

  return managed_ptr<T>(new (mem) T(std::forward<Args>(args)...), mem);
}
