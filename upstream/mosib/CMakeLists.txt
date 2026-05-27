cmake_minimum_required(VERSION 2.8)

project(mosib)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-std=c++14 -O3 -fopenmp")

include_directories(./src)

aux_source_directory(./src INCLUDES)

add_library(INCLUDES_LIB ${INCLUDES})

add_executable(local_exact ./run/run_local_exact.cpp)
target_link_libraries(local_exact INCLUDES_LIB)

add_executable(global_exact ./run/run_global_exact.cpp)
target_link_libraries(global_exact INCLUDES_LIB)

add_executable(global_app ./run/run_global_app.cpp)
target_link_libraries(global_app INCLUDES_LIB)
