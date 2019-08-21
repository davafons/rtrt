#pragma once

class Ray;
class Vec3;

__device__ Vec3 color(const Ray &r);
