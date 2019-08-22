#include "hitable/hitable_list.cuh"
#include "hitable/sphere.cuh"
#include "utils/cuda_utils.cuh"

__device__ HitableList::HitableList(size_t n) : n_(n) {
  list_ = new Hitable *[n];
  list_[0] = new Sphere(Vec3(0, 0, -1), 0.5f);
  list_[1] = new Sphere(Vec3(0, -100.5f, -1), 100);

  n_ = 2;
}

__device__ HitableList::~HitableList() { delete[] list_; }

__device__ bool HitableList::hit(const Ray &r, float t_min, float t_max,
                                 HitRecord &rec) const {
  HitRecord tmp_rec;
  bool hit_anything = false;

  float closest_so_far = t_max;
  for (int i = 0; i < n_; ++i) {
    if (list_[i]->hit(r, t_min, closest_so_far, tmp_rec)) {
      hit_anything = true;
      closest_so_far = tmp_rec.t;
      rec = tmp_rec;
    }
  }

  return hit_anything;
}
