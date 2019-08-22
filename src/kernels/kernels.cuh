#pragma once

#include "utils/cuda_utils.cuh"

class World;
class TextureGPU;
class HitableList;

__global__ void chapter_2_kernel(TextureGPU *tex, World world);
__global__ void chapter_3_kernel(TextureGPU *tex, World world);
__global__ void chapter_4_kernel(TextureGPU *tex, World world);
__global__ void chapter_5_kernel(TextureGPU *tex, World world,
                                 HitableList **hitable_objects);
