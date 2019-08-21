cmake_minimum_required(VERSION 3.7)

project(rtrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


find_package(CUDA REQUIRED)
find_package(SDL2 REQUIRED)


add_executable(${PROJECT_NAME} src/main.cu src/frontend/window.cpp src/kernels/sky.cu
  src/frontend/texturegpu.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${SDL2_INCLUDE_DIRS}
  "${CMAKE_SOURCE_DIR}/src/")
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
