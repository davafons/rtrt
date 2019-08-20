cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(rtrt LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Dirs
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty")

# Source files
file(GLOB_RECURSE SOURCE_FILES
  ${CMAKE_SOURCE_DIR}/src/*.c
  ${CMAKE_SOURCE_DIR}/src/*.cpp
  ${CMAKE_SOURCE_DIR}/src/*.cu)

file(GLOB_RECURSE HEADER_FILES
  ${CMAKE_SOURCE_DIR}/src/*.h
  ${CMAKE_SOURCE_DIR}/src/*.hpp
  ${CMAKE_SOURCE_DIR}/src/*.cuh)


# Define the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})


# Find external libraries
find_package(CUDA REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

set(GLAD_DIR "${LIB_DIR}/glad")
add_library(glad "${GLAD_DIR}/src/glad.c")
target_include_directories(glad PRIVATE "${GLAD_DIR}/include")


# Includes and link
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_BINARY_DIR}/src
  "${GLAD_DIR}/include"
  ${OPENGL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} glfw glad ${OPENGL_LIBRARIES} ${CMAKE_DL_LIBS})


# Cuda properties
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 14)
