#pragma once

#include "math/vec3.cuh"

struct World {
  Vec3 lower_left_corner = Vec3(-2.0, -1.0, -1.0);
  Vec3 horizontal = Vec3(4.0, 0.0, 0.0);
  Vec3 vertical = Vec3(0.0, 2.0, 0.0);
  Vec3 origin = Vec3(0.0, 0.0, 0.0);
};
typedef struct World World;
