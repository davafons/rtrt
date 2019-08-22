#pragma once

#include "utils/cuda_utils.cuh"

template <class T> class cuda_vector {
public:
  using value_type = T;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using iterator = T *;
  using const_iterator = const T *;
  using const_pointer = const T *;
  using size_type = size_t;

  __device__ cuda_vector(size_type size = 0)
      : data_(NULL), size_(size), capacity_(size) {
    data_ = new T[capacity_];
  }

  __device__ ~cuda_vector() {
    if (data_ != NULL) {
      delete[] data_;
      data_ = NULL;
    }
  }

  __device__ iterator begin() noexcept { return iterator(&data_[0]); }
  __device__ const_iterator begin() const noexcept {
    return const_iterator(&data_[0]);
  }

  __device__ iterator end() noexcept { return iterator(&data_[size_]); }
  __device__ const_iterator end() const noexcept {
    return const_iterator(&data_[size_]);
  }

  __device__ size_type size() const noexcept { return size_; }

  __device__ void resize(size_type new_size) {
    if (size_ > capacity_) {
      realloc_vector(new_size * 2);
    }

    size_ = new_size;
  }

  __device__ size_type capacity() const noexcept { return capacity_; }

  __device__ bool empty() const noexcept { return !size_; }

  __device__ void reserve(size_t new_capacity) {
    if (new_capacity > capacity_) {
      realloc_vector(new_capacity);
    }
  }

  __device__ reference operator[](size_type position) {
    return data_[position];
  }

  __device__ const_reference operator[](size_type position) const {
    return data_[position];
  }

  __device__ reference front() { return data_[0]; }
  __device__ const_reference front() const { return data_[0]; }

  __device__ reference back() { return data_[size_ - 1]; }
  __device__ const_reference back() const { return data_[size_ - 1]; }

  __device__ pointer data() { return data_; }
  __device__ const_pointer data() const { return data_; }

  __device__ void push_back(const_reference value) {
    if (size_ >= capacity_) {
      realloc_vector(size_ * 2);
    }

    data_[size_] = value;
    ++size_;
  }

  /* template <class... Args> */
  /* emplace(const_iterator position, Args && ... args) { */
  /*   size_type pos = position - data_; */
  /*  */
  /*   if(size_ >= capacity_) { */
  /*     realloc_vector(size_ * 2); */
  /*   } */
  /*  */
  /*   if(position != end() && size_ != 0) { */
  /*  */
  /*   }) */
  /*  */
  /* } */

private:
  __device__ void realloc_vector(size_type new_capacity) {
    if (new_capacity == 0) {
      ++new_capacity;
    }

    pointer new_data = new T[new_capacity];
    memcpy(new_data, data_, sizeof(T) * size_);

    delete[] data_;

    data_ = new_data;
    capacity_ = new_capacity;
  }

private:
  pointer data_;
  size_type size_;
  size_type capacity_ = 10;
};
