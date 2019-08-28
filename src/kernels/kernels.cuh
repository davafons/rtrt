#pragma once

#include <curand_kernel.h>

class World;
class TextureGPU;
class Hitable;
class Camera;

__global__ void raytracing(TextureGPU *tex, Camera camera,
                           Hitable **hitable_objects, curandState *rand_state);
