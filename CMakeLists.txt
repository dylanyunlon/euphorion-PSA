cmake_minimum_required(VERSION 3.14)

project(euphorion-psa VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -Wall -Wextra -Wno-unused-parameter")

# ── Upstream mosib library ─────────────────────────────────
include_directories(upstream/mosib/src)
aux_source_directory(upstream/mosib/src MOSIB_SOURCES)
add_library(mosib_lib ${MOSIB_SOURCES})

# ── Upstream mosib executables (preserved) ─────────────────
add_executable(local_exact upstream/mosib/run/run_local_exact.cpp)
target_link_libraries(local_exact mosib_lib)

add_executable(global_exact upstream/mosib/run/run_global_exact.cpp)
target_link_libraries(global_exact mosib_lib)

add_executable(global_app upstream/mosib/run/run_global_app.cpp)
target_link_libraries(global_app mosib_lib)

# ── Euphorion benchmark ───────────────────────────────────
include_directories(src/core src/scheduler src/sampling src/comm)

add_executable(benchmark_euphorion src/benchmark/benchmark_euphorion.cpp)
target_link_libraries(benchmark_euphorion mosib_lib pthread)

add_executable(benchmark_hardware src/benchmark/benchmark_hardware.cpp)
target_link_libraries(benchmark_hardware mosib_lib pthread)

add_executable(generate_data src/benchmark/generate_data.cpp)
target_link_libraries(generate_data mosib_lib pthread)

add_executable(benchmark_engine src/benchmark/benchmark_engine.cpp)
target_link_libraries(benchmark_engine mosib_lib pthread)
