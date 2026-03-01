cmake_minimum_required (VERSION 3.14)
project (SOS)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

include(FetchContent)
include(ExternalProject)

find_package(OpenMP)
find_package(MKL REQUIRED)
find_package(JeMalloc REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${MKL_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${JEMALLOC_INCLUDE_DIR})
include_directories(SYSTEM src/competitor/hot/src/libs/hot/commons/include)
include_directories(SYSTEM src/competitor/hot/src/libs/hot/rowex/include)
include_directories(SYSTEM src/competitor/hot/src/libs/hot/single-threaded/include)
include_directories(SYSTEM src/competitor/hot/src/libs/idx/content-helpers/include)
include_directories(SYSTEM src/competitor/dytis/src)
include_directories(SYSTEM src/competitor/loft/src)
include_directories(SYSTEM src/competitor/carmi/src/src/include)
add_compile_options(-faligned-new -march=native -g -O3)

# add_subdirectory(src/competitor/wormhole)
add_subdirectory(src/competitor/wormhole_u64)
add_subdirectory(src/competitor/masstree)
add_subdirectory(src/competitor/dili)

# microbench
add_executable(microbench
        ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/microbench.cpp
    )

target_link_libraries(microbench
        PRIVATE
        # WORMHOLE
        WORMHOLEU64
        MASSTREE
        DILI
)

find_library(URCU_QSBR_LIB urcu-qsbr)
if (URCU_QSBR_LIB)
    target_compile_definitions(microbench PUBLIC GRE_HAVE_URCU_QSBR=1)
    target_link_libraries(microbench PUBLIC ${URCU_QSBR_LIB})
else()
    message(WARNING "urcu-qsbr not found; LOFT adapter will be disabled at compile time.")
    target_compile_definitions(microbench PUBLIC GRE_HAVE_URCU_QSBR=0)
endif()

target_link_libraries(microbench PUBLIC OpenMP::OpenMP_CXX Threads::Threads ${JEMALLOC_LIBRARIES} ${MKL_LIBRARIES} ${TBB_LIBRARIES})
