#pragma once

#include <curand_kernel.h>

class World;
class TextureGPU;
class Hitable;
class Camera;

__global__ void chapter_2_kernel(TextureGPU *tex, World world);
__global__ void chapter_3_kernel(TextureGPU *tex, World world);
__global__ void chapter_4_kernel(TextureGPU *tex, World world);
__global__ void chapter_5_kernel(TextureGPU *tex, World world,
                                 Hitable **hitable_objects);
__global__ void chapter_6_kernel(TextureGPU *tex, Camera camera,
                                 Hitable **hitable_objects, curandState*rand_state);
__global__ void chapter_7_kernel(TextureGPU *tex, Camera camera,
                                 Hitable **hitable_objects, curandState*rand_state);
__global__ void chapter_8_kernel(TextureGPU *tex, Camera camera,
                                 Hitable **hitable_objects, curandState*rand_state);
