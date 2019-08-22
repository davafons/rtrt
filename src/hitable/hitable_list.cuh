#pragma once

#include "hitable/hitable.cuh"
#include "utils/cuda_vector.cuh"

class HitableList : public Hitable {

public:
  __device__ HitableList(size_t n);
  __device__ ~HitableList();

  __device__ size_t size() const { return n_; }

  __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                              HitRecord &rec) const;
  Hitable **list_;
  cuda_vector<Hitable *> c_list_;

private:
  size_t n_;
};
