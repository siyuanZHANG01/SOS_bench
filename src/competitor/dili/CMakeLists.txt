cmake_minimum_required(VERSION 3.14)

# GRE wrapper for DILI: build as a normal library (avoid including *.cpp from headers).
add_library(DILI
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/dili/DILI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/dili/diliNode.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/utils/linux_sys_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/utils/data_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/utils/file_utils.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/global/global.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/global/linearReg.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/butree/interval_utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/butree/linearRegressor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/butree/interval.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src/butree/buInterval.cpp
)

target_include_directories(DILI
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/src
)

target_compile_features(DILI PUBLIC cxx_std_17)


