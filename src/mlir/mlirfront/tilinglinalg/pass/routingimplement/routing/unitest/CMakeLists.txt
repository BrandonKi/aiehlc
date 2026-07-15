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
add_executable(routing_ut 
    ${TEST_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/../routingpath.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/hwresource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/ResourceManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../routingtopology.cpp
    )

# Link against the production library defined one level up
#target_link_libraries(routing_ut
#    PRIVATE
#    "${CMAKE_CURRENT_SOURCE_DIR}/../build/librouting.a"
#)

project(aie_routing_ut LANGUAGES CXX)

target_include_directories(routing_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(routing_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_include_directories(routing_ut PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

message(STATUS "Using ${CMAKE_CURRENT_SOURCE_DIR}/../")

target_compile_features(routing_ut PRIVATE cxx_std_17)

# ------------------------------------------------------------------
# 3.  Register with CTest
# ------------------------------------------------------------------
enable_testing()
add_test(NAME routing_ut COMMAND routing_ut)
