#pragma once

#include "material/material.cuh"
#include "math/math.cuh"

class Lambertian : public Material {
public:
  __device__ Lambertian(const Vec3 &albedo) : albedo_(albedo) {}

  __device__ virtual bool
  scatter(const Ray &r_in, const HitRecord &rec, Vec3 &attenuation,
          Ray &scattered, curandState *local_rand_state) const override {

    Vec3 target =
        rec.p + rec.normal + Math::random_in_unit_sphere(local_rand_state);
    scattered = Ray(rec.p, target - rec.p);
    attenuation = albedo_;

    return true;
  }

private:
  Vec3 albedo_;
};
