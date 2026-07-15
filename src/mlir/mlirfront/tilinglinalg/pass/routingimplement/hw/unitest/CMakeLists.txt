# /******************************************************************************
# * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# * SPDX-License-Identifier: Apache-2.0
# ******************************************************************************/

cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------------
# 1.  Test sources in this folder
# ------------------------------------------------------------------
file(GLOB TEST_SRC CONFIGURE_DEPENDS
     *.cpp)                        # e.g. test.cpp

# ------------------------------------------------------------------
# 2.  Build test executable
# ------------------------------------------------------------------
add_executable(hwresource_ut 
            ${TEST_SRC}
            ../hwresource.cpp
            ../ResourceManager.cpp)

# Link against the production library defined one level up
#target_link_libraries(hwresource_ut
#    PRIVATE
#    "${CMAKE_CURRENT_SOURCE_DIR}/../build/libhwresource.a"
#)

project(aie_hwresource_ut LANGUAGES CXX)

target_include_directories(hwresource_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hwresource_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_include_directories(hwresource_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)

message(STATUS "Using ${CMAKE_CURRENT_SOURCE_DIR}/../")

target_compile_features(hwresource_ut PRIVATE cxx_std_17)

# ------------------------------------------------------------------
# 3.  Register with CTest
# ------------------------------------------------------------------
enable_testing()
add_test(NAME hwresource_ut COMMAND hwresource_ut)
