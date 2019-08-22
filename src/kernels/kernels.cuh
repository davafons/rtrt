#pragma once

class World;
class TextureGPU;
class HitableList;
template <class T> class managed_ptr;

__global__ void chapter_2_kernel(TextureGPU *tex, World world);
__global__ void chapter_3_kernel(TextureGPU *tex, World world);
__global__ void chapter_4_kernel(TextureGPU *tex, World world);
__global__ void chapter_5_kernel(managed_ptr<TextureGPU> tex, World world,
                                 HitableList **hitable_objects);
