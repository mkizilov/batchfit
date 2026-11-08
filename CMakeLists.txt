cmake_minimum_required(VERSION 3.24)
project(batchfit VERSION 0.1.0 LANGUAGES CXX)

option(BATCHFIT_METAL "Build the Apple Metal backend" ${APPLE})
option(BATCHFIT_BUILD_TESTS "Build tests (GoogleTest via FetchContent)" ON)
option(BATCHFIT_BUILD_BENCHMARKS "Build Google Benchmark CPU/residual benchmarks" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
add_library(batchfit STATIC src/cpu/reference.cpp src/cpu/npy.cpp src/cpu/synthetic.cpp
    src/cpu/thread_pool.cpp src/metal/runtime.cpp)
add_library(batchfit::batchfit ALIAS batchfit)
target_compile_features(batchfit PUBLIC cxx_std_20)
target_link_libraries(batchfit PUBLIC Threads::Threads)
set_target_properties(batchfit PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(batchfit PUBLIC include PRIVATE src)
target_compile_definitions(batchfit PUBLIC BATCHFIT_HAS_METAL=$<BOOL:${BATCHFIT_METAL}>)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(batchfit PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(batchfit_generate tools/generate.cpp)
target_link_libraries(batchfit_generate PRIVATE batchfit::batchfit)

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
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(OUTPUT "${PROJECT_BINARY_DIR}/generated/lm_source.hpp"
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/embed_metal.py"
            "${PROJECT_BINARY_DIR}/generated/lm_source.hpp"
        DEPENDS tools/embed_metal.py src/metal/lm.metal include/batchfit/models.hpp
            include/batchfit/metal/types.hpp VERBATIM)
    target_sources(batchfit PRIVATE "${PROJECT_BINARY_DIR}/generated/lm_source.hpp")
    add_custom_command(OUTPUT "${PROJECT_BINARY_DIR}/generated/profile_source.hpp"
        COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/embed_metal.py"
            "${PROJECT_BINARY_DIR}/generated/profile_source.hpp" --profile
        DEPENDS tools/embed_metal.py src/metal/lm.metal src/metal/profile_variants.metal
            include/batchfit/models.hpp include/batchfit/metal/types.hpp VERBATIM)
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
    add_executable(batchfit_solver_check tests/solver_check.cpp)
    target_link_libraries(batchfit_solver_check PRIVATE batchfit::batchfit)
    add_test(NAME solver_check COMMAND batchfit_solver_check)
    add_executable(batchfit_initial_guess_check tests/initial_guess_check.cpp)
    target_link_libraries(batchfit_initial_guess_check PRIVATE batchfit::batchfit)
    add_test(NAME initial_guess_check COMMAND batchfit_initial_guess_check)
    add_executable(batchfit_batch_check tests/batch_check.cpp)
    target_link_libraries(batchfit_batch_check PRIVATE batchfit::batchfit)
    add_test(NAME batch_check COMMAND batchfit_batch_check)
    set_tests_properties(batch_check PROPERTIES PROCESSORS 2 TIMEOUT 120)
    add_executable(batchfit_batch_failure_check tests/batch_failure_check.cpp)
    target_link_libraries(batchfit_batch_failure_check PRIVATE batchfit::batchfit)
    add_test(NAME batch_failure_check COMMAND batchfit_batch_failure_check)
    set_tests_properties(batch_failure_check PROPERTIES PROCESSORS 2 TIMEOUT 120)
    add_executable(batchfit_solver_allocation_check tests/solver_allocation_check.cpp)
    target_link_libraries(batchfit_solver_allocation_check PRIVATE batchfit::batchfit)
    add_test(NAME solver_allocation_check COMMAND batchfit_solver_allocation_check)
    add_executable(batchfit_synthetic_check tests/synthetic_check.cpp)
    target_link_libraries(batchfit_synthetic_check PRIVATE batchfit::batchfit)
    add_test(NAME synthetic_check COMMAND batchfit_synthetic_check)
    set_tests_properties(synthetic_check PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    if(BATCHFIT_METAL)
        enable_language(OBJCXX)
        add_executable(batchfit_models_metal tests/models_metal.mm)
        target_compile_options(batchfit_models_metal PRIVATE -fobjc-arc)
        target_link_libraries(batchfit_models_metal PRIVATE batchfit::batchfit
            "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
        set_target_properties(batchfit_models_metal PROPERTIES OBJCXX_STANDARD 20
            OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
        add_test(NAME models_metal COMMAND batchfit_models_metal "${PROJECT_SOURCE_DIR}")
        add_executable(batchfit_metal_lm_check tests/metal_lm_check.cpp)
        target_link_libraries(batchfit_metal_lm_check PRIVATE batchfit::batchfit)
        add_test(NAME metal_lm_check COMMAND batchfit_metal_lm_check)
    else()
        add_executable(batchfit_metal_disabled_check tests/metal_disabled_check.cpp)
        target_link_libraries(batchfit_metal_disabled_check PRIVATE batchfit::batchfit)
        add_test(NAME metal_disabled_check COMMAND batchfit_metal_disabled_check)
    endif()
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
        URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    add_executable(batchfit_tests tests/backend_test.cpp tests/models_test.cpp tests/synthetic_test.cpp
        tests/solver_test.cpp tests/initial_guess_test.cpp tests/batch_test.cpp)
    target_include_directories(batchfit_tests PRIVATE src)
    target_link_libraries(batchfit_tests PRIVATE batchfit::batchfit GTest::gtest_main)
    if(BATCHFIT_METAL)
        target_sources(batchfit_tests PRIVATE tests/metal_lm_test.cpp)
    endif()
    include(GoogleTest)
    gtest_discover_tests(batchfit_tests WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" PROPERTIES PROCESSORS 2)
endif()

if(BATCHFIT_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.tar.gz
        URL_HASH SHA256=b334658edd35efcf06a99d9be21e4e93e092bd5f95074c1673d5c8705d95c104
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(googlebenchmark)
    add_executable(batchfit_bench bench/residuals.cpp bench/cpu_batch.cpp)
    target_include_directories(batchfit_bench PRIVATE src)
    target_link_libraries(batchfit_bench PRIVATE batchfit::batchfit benchmark::benchmark_main)
    add_executable(batchfit_cpu_baseline bench/cpu_baseline.cpp)
    target_link_libraries(batchfit_cpu_baseline PRIVATE batchfit::batchfit)
    if(BATCHFIT_METAL)
        target_sources(batchfit_bench PRIVATE bench/metal_batch.cpp)
        add_executable(batchfit_metal_baseline bench/metal_baseline.cpp)
        target_link_libraries(batchfit_metal_baseline PRIVATE batchfit::batchfit)
        enable_language(OBJCXX)
        add_executable(batchfit_metal_profile bench/metal_profile.mm
            "${PROJECT_BINARY_DIR}/generated/profile_source.hpp")
        target_include_directories(batchfit_metal_profile PRIVATE src/metal "${PROJECT_BINARY_DIR}/generated")
        target_compile_options(batchfit_metal_profile PRIVATE -fobjc-arc)
        set_target_properties(batchfit_metal_profile PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED ON)
        target_link_libraries(batchfit_metal_profile PRIVATE batchfit::batchfit "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")
    endif()
endif()
