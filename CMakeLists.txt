cmake_minimum_required(VERSION 3.18)
project(gem_blockset LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

add_library(gem_blockset INTERFACE)
target_include_directories(gem_blockset INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
    $<INSTALL_INTERFACE:include>
)

install(
    DIRECTORY models/
    DESTINATION include
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.cu"
        PATTERN "*.cuh"
)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
