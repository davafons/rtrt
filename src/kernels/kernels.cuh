#pragma once

#include "frontend/texturegpu.cuh"
#include "utils/world.cuh"

__global__ void chapter_3_kernel(TextureGPU *tex, World world);
