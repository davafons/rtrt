#include <iostream>

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define TX 32
#define TY 32

////////////////////777

__device__ unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__ void distanceKernel(uchar4 *d_out, int w, int h) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x >= w) || (y >= h))
    return;                // Check if within image bounds
  const int i = y * w + x; // 1D indexing

  d_out[i].x = 255;
  d_out[i].y = 255;
  d_out[i].z = 0;
  d_out[i].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h) {
  const dim3 blockSize(TX, TY);
  const dim3 gridSize = dim3((w + TX - 1) / TX, (h + TY - 1) / TY);
  distanceKernel<<<gridSize, blockSize>>>(d_out, w, h);
}

////////////////////777

void framebuffer_size_callback(GLFWwindow *window, int w, int h);
void processInput(GLFWwindow *window);

int v_width = 800;
int v_height = 600;
GLuint pbo = 0;
GLuint tex = 0;
cudaGraphicsResource_t cuda_pbo_resource = 0;

void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,
                                       cuda_pbo_resource);

  kernelLauncher(d_out, v_width, v_height);

  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, v_width, v_height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  // clang-format off
  glTexCoord2f(0.0f, 0.0f); glVertex2f(-1, -1);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(-1, v_height);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(v_width, v_height);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(v_width, 1);
  // clang-format on
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

int main() {
  try {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow *window =
        glfwCreateWindow(v_width, v_height, "Raytracing", NULL, NULL);

    if (!window) {
      throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
      throw std::runtime_error("Failed to initialize GLAD");
    }

    glViewport(0, 0, v_width, v_height);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 4 * v_width * v_height * sizeof(GLubyte), 0, GL_STREAM_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                                 cudaGraphicsMapFlagsWriteDiscard);

    while (!glfwWindowShouldClose(window)) {
      processInput(window);

      glClearColor(0.2f, 0.3f, 0.3f, 1.0f);

      // Set every pixel in the frame buffer to the current clear color.
      glClear(GL_COLOR_BUFFER_BIT);

      render();
      drawTexture();

      glfwSwapBuffers(window);
      glfwPollEvents();
    }

  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwTerminate();
  return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int w, int h) {
  glViewport(0, 0, w, h);
  v_width = w;
  v_height = h;
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
}
