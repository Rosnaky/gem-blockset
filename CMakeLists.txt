cmake_minimum_required(VERSION 3.18)
project(gem_blockset LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

add_library(gem_blockset_headers INTERFACE)
target_include_directories(gem_blockset_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/abstract_primitives>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/primitives>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/composites>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models/estimators>
    $<INSTALL_INTERFACE:include>
)

option(GEM_BUILD_FFI "Build the extern C FFI library" OFF)
if(GEM_BUILD_FFI)
    add_library(gem_blockset STATIC models/ffi.cu)
    target_link_libraries(gem_blockset PUBLIC gem_blockset_headers)
    install(TARGETS gem_blockset DESTINATION lib)
endif()

install(
    DIRECTORY models/
    DESTINATION include
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.c"
        PATTERN "*.cpp"
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
