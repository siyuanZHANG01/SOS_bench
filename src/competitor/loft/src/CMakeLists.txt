project(sim)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
# Enable C++11
find_package(OpenMP)
cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 17)

set(MKL_LINK_DIRECTORY "/opt/intel/mkl/lib/intel64")
set(MKL_INCLUDE_DIRECTORY "/opt/intel/mkl/include")
set(JEMALLOC_DIR "/usr/lib/x86_64-linux-gnu")

include_directories(${PROJECT_SOURCE_DIR} "/home/userspace-rcu/include" )
include_directories(${MKL_INCLUDE_DIRECTORY})

ADD_DEFINITIONS(-std=c++17)

set(CMAKE_CXX_FLAGS "-O2 -g -DNDEBUG -march=native")

link_directories(${MKL_LINK_DIRECTORY} "/usr/local/lib")
link_libraries(jemalloc)

add_executable(microbench ../bench/microbench.cpp)

target_link_libraries(microbench OpenMP::OpenMP_CXX mkl_rt urcu-qsbr pthread)


