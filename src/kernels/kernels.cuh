#pragma once

class World;
class TextureGPU;
class Hitable;
template <class T> class managed_ptr;

__global__ void chapter_2_kernel(TextureGPU *tex, World world);
__global__ void chapter_3_kernel(TextureGPU *tex, World world);
__global__ void chapter_4_kernel(TextureGPU *tex, World world);
__global__ void chapter_5_kernel(TextureGPU* tex, World world,
                                 Hitable** hitable_objects);
