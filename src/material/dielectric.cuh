#pragma once

#include "material/material.cuh"
#include "math/math.cuh"

class Dielectric : public Material {
public:
  __device__ Dielectric(float refractive_indices)
      : ref_idx_(refractive_indices) {}

  __device__ virtual bool
  scatter(const Ray &r_in, const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, curandState *local_rand_state) const override {

    Vec3 outward_normal;
    Vec3 reflected = Math::reflect(r_in.direction(), rec.normal);

    float ni_over_nt;
    attenuation = Vec3(1.0f, 1.0f, 1.0f);

    Vec3 refracted;
    float reflect_prob;
    float cosine;

    if (dot(r_in.direction(), rec.normal) > 0) {
      outward_normal = -rec.normal;
      ni_over_nt = ref_idx_;

      cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
      cosine = sqrt(1.0f - ref_idx_ * ref_idx_ * (1 - cosine * cosine));

    } else {
      outward_normal = rec.normal;
      ni_over_nt = 1.0f / ref_idx_;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }

    if (Math::refract(r_in.direction(), outward_normal, ni_over_nt,
                      refracted)) {

      reflect_prob = Math::schlick(cosine, ref_idx_);

    } else {
      reflect_prob = 1.0f;
    }

    if (curand_uniform(local_rand_state) < reflect_prob) {
      scattered = Ray(rec.p, reflected);
    } else {
      scattered = Ray(rec.p, refracted);
    }

    return true;
  }

private:
  float ref_idx_;
};
