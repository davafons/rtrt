#pragma once

#include "frontend/texturegpu.cuh"
#include "utils/world.cuh"

__global__ void sky(TextureGPU *tex, World world);
