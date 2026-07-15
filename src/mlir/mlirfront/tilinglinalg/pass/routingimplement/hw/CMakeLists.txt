# /******************************************************************************
# * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# * SPDX-License-Identifier: Apache-2.0
# ******************************************************************************/

# resource/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
# This file is processed from the top-level CMakeLists.txt via
#     add_subdirectory(resource)

# ------------------------------------------------------------
# 1.  Production library
# ------------------------------------------------------------
set(HWRESOURCE_SRC
    hwresource.cpp          # add more .cpp files here as needed
    ResourceManager.cpp
)

project(aie_hwresource LANGUAGES CXX)

add_library(hwresource STATIC ${HWRESOURCE_SRC})

target_include_directories(hwresource
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}   # allows #include "hw_resource.hpp"
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/
)

target_compile_features(hwresource PUBLIC cxx_std_17)

# ------------------------------------------------------------
# 2.  Unit tests (optional)
# ------------------------------------------------------------
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/unitest/CMakeLists.txt")
    add_subdirectory(unitest)         # builds tests when present
endif()
