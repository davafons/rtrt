#include "hitable/hitable_list.cuh"
#include "hitable/sphere.cuh"
#include "utils/cuda_utils.cuh"

__device__ bool HitableList::hit(const Ray &r, float t_min, float t_max,
                                 HitRecord &rec) const {
  HitRecord tmp_rec;
  bool hit_anything = false;

  float closest_so_far = t_max;
  for (int i = 0; i < c_list_.size(); ++i) {
    if (c_list_[i]->hit(r, t_min, closest_so_far, tmp_rec)) {
      hit_anything = true;
      closest_so_far = tmp_rec.t;
      rec = tmp_rec;
    }
  }

  return hit_anything;
}
