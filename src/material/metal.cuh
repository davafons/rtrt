#pragma once

#include "material/material.cuh"
#include "math/math.cuh"

class Metal : public Material {
public:
  __device__ Metal(const Vec3 &albedo, float fuzz)
      : albedo_(albedo), fuzz_(fuzz) {
    if (fuzz < 1) {
      fuzz_ = fuzz;
    } else {
      fuzz_ = 1;
    }
  }

  __device__ virtual bool scatter(const Ray &r_in, const HitRecord &rec,
                                  Vec3 &attenuation, Ray &scattered,
                                  curandState *local_rand_state) override {

    Vec3 reflected = Math::reflect(unit_vector(r_in.direction()), rec.normal);

    scattered =
        Ray(rec.p,
            reflected + fuzz_ * Math::random_in_unit_sphere(local_rand_state));
    attenuation = albedo_;

    return dot(scattered.direction(), rec.normal) > 0;
  }

private:
  Vec3 albedo_;
  float fuzz_;
};
