#include "hitable/sphere.cuh"

__device__ bool Sphere::hit(const Ray &r, float t_min, float t_max,
                            HitRecord &rec) const {
  Vec3 oc = r.origin() - center_;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius_ * radius_;
  float discriminant = b * b - a * c;

  if (discriminant > 0) {
    float tmp = (-b - sqrt(discriminant)) / a;

    // if(t_min < tmp && tmp < t_max) {
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center_) / radius_;

      return true;
    }

    tmp = (-b + sqrt(discriminant)) / a;
    if (tmp < t_max && tmp > t_min) {
      rec.t = tmp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center_) / radius_;

      return true;
    }
  }

  return false;
}
