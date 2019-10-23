cmake_minimum_required(VERSION 3.9)

project(FastANN VERSION 1.0
                DESCRIPTION "Accelerated Approximate Nearest Neighbor Search"
                LANGUAGES CXX)

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "Cannot build in source directory, "
            "use a separate build directory")
endif()

option(BUILD_EXPERIMENTS "Build experiments" ON)

add_library(fast_ann INTERFACE)
target_compile_features(fast_ann INTERFACE cxx_std_11)
target_include_directories(fast_ann INTERFACE ${PROJECT_SOURCE_DIR}/include)

if(BUILD_EXPERIMENTS)
    add_subdirectory(experiments)
endif(BUILD_EXPERIMENTS)
