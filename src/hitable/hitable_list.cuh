#pragma once

#include "hitable/hitable.cuh"
#include "utils/cuda_vector.cuh"

class HitableList : public Hitable {

public:
  __device__ size_t size() const { return n_; }

  __device__ void push_back(Hitable *hitable) { c_list_.push_back(hitable); }

  __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                              HitRecord &rec) const;

  cuda_vector<Hitable *> c_list_;

private:
  size_t n_;
};
