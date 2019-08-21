#pragma once

class World;
class TextureGPU;

__global__ void chapter_2_kernel(TextureGPU *tex, World world);
__global__ void chapter_3_kernel(TextureGPU *tex, World world);
__global__ void chapter_4_kernel(TextureGPU *tex, World world);
