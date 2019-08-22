#pragma once

#include "math/vec3.cuh"

class World {
public:
  Vec3 lower_left_corner = Vec3(-2.0f, -1.0f, -1.0f);
  Vec3 horizontal = Vec3(4.0f, 0.0f, 0.0f);
  Vec3 vertical = Vec3(0.0f, 2.0f, 0.0f);
  Vec3 origin = Vec3(0.0f, 0.0f, 0.0f);
};
typedef struct World World;
