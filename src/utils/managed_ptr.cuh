#pragma once

template <class T> class managed_ptr {
public:
  explicit managed_ptr(T *data, void *mem)
      : data_(data), mem_(mem), count_(new int(1)) {}

  ~managed_ptr() {
    --(*count_);

    if (*count_ == 0) {
      data_->~T();
      cudaCheckErr(cudaFree(mem_));
    }
  }

  __host__ __device__ managed_ptr(const managed_ptr &copy)
      : data_(copy.data_), mem_(copy.mem_), count_(copy.count_) {
    ++(*count_);
  }

  __host__ __device__ managed_ptr &operator=(managed_ptr rhs) {
    rhs.swap(*this);

    return *this;
  }

  __host__ __device__ T *operator->() const { return data_; }
  __host__ __device__ T &operator*() const { return *data_; }

  __host__ __device__ T *get() const { return data_; }
  __host__ __device__ explicit operator bool() const { return data_; }

  __host__ __device__ void swap(managed_ptr &rhs) noexcept {
    std::swap(data_, rhs.data_);
    std::swap(mem_, rhs.mem_);
    std::swap(count_, rhs.count_);
  }

private:
  T *data_ = NULL;
  void *mem_ = NULL;
  int *count_;
};

template <class T, typename... Args>
managed_ptr<T> make_managed(Args &&... args) {
  void *mem = cuda_malloc_managed<T>(sizeof(T));

  return managed_ptr<T>(new (mem) T(std::forward<Args>(args)...), mem);
}
