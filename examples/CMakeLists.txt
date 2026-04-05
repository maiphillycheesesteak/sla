cmake_minimum_required(VERSION 3.10)
project(SLA_Examples C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find OpenMP and require it
find_package(OpenMP REQUIRED)

add_executable(benchmark benchmark.c)

# Link to OpenMP and Math library
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_C m)

# Include directories
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Optimization flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(benchmark PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(benchmark PRIVATE /O2)
endif()
