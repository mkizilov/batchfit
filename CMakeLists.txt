cmake_minimum_required(VERSION 3.24)
project(batchfit VERSION 0.1.0 LANGUAGES CXX)

option(BATCHFIT_METAL "Build the Apple Metal backend" ${APPLE})
option(BATCHFIT_BUILD_TESTS "Build tests (GoogleTest via FetchContent)" ON)
option(BATCHFIT_BUILD_BENCHMARKS "Build Google Benchmark smoke benchmark" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(batchfit STATIC src/cpu/reference.cpp)
add_library(batchfit::batchfit ALIAS batchfit)
target_compile_features(batchfit PUBLIC cxx_std_20)
set_target_properties(batchfit PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(batchfit PUBLIC include PRIVATE src)
target_compile_definitions(batchfit PUBLIC BATCHFIT_HAS_METAL=$<BOOL:${BATCHFIT_METAL}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(batchfit PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(FetchContent)
if(BATCHFIT_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "BATCHFIT_METAL=ON requires macOS and Apple Metal frameworks")
    endif()
    # The versioned Apple archive is downloaded only for Metal builds.
    FetchContent_Declare(metal_cpp
        URL https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip
        SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/metal-cpp"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(metal_cpp)
    if(NOT EXISTS "${metal_cpp_SOURCE_DIR}/Metal/Metal.hpp")
        message(FATAL_ERROR "metal-cpp download is missing Metal/Metal.hpp")
    endif()
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    # Runtime shader compilation avoids requiring the optional offline metal toolchain.
    file(READ "${PROJECT_SOURCE_DIR}/src/metal/residuals.metal" BATCHFIT_SHADER_SOURCE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/metal/residuals.metal)
    configure_file(src/metal/shader_source.hpp.in generated/shader_source.hpp @ONLY)
    target_sources(batchfit PRIVATE src/metal/backend.cpp)
    target_include_directories(batchfit SYSTEM PRIVATE "${metal_cpp_SOURCE_DIR}")
    target_include_directories(batchfit PRIVATE "${PROJECT_BINARY_DIR}/generated")
    target_link_libraries(batchfit PRIVATE "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
endif()

if(BATCHFIT_BUILD_TESTS)
    enable_testing()
    add_executable(batchfit_cpu_smoke tests/cpu_smoke.cpp)
    target_include_directories(batchfit_cpu_smoke PRIVATE src)
    target_link_libraries(batchfit_cpu_smoke PRIVATE batchfit::batchfit)
    add_test(NAME cpu_smoke COMMAND batchfit_cpu_smoke)
    add_executable(batchfit_models_check tests/models_check.cpp)
    target_link_libraries(batchfit_models_check PRIVATE batchfit::batchfit)
    add_test(NAME models_check COMMAND batchfit_models_check)
    if(BATCHFIT_METAL)
        enable_language(OBJCXX)
        add_executable(batchfit_models_metal tests/models_metal.mm)
        target_compile_options(batchfit_models_metal PRIVATE -fobjc-arc)
        target_link_libraries(batchfit_models_metal PRIVATE batchfit::batchfit
            "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
        set_target_properties(batchfit_models_metal PROPERTIES OBJCXX_STANDARD 20
            OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
        add_test(NAME models_metal COMMAND batchfit_models_metal "${PROJECT_SOURCE_DIR}")
    endif()
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_executable(batchfit_tests tests/backend_test.cpp tests/models_test.cpp)
    target_include_directories(batchfit_tests PRIVATE src)
    target_link_libraries(batchfit_tests PRIVATE batchfit::batchfit GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(batchfit_tests)
endif()

if(BATCHFIT_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
        URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googlebenchmark)
    add_executable(batchfit_bench bench/residuals.cpp)
    target_include_directories(batchfit_bench PRIVATE src)
    target_link_libraries(batchfit_bench PRIVATE batchfit::batchfit benchmark::benchmark_main)
endif()
