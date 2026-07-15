###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

cmake_minimum_required(VERSION 3.5)
project(mlirtest)
#set(CMAKE_C_COMPILER "/scratch/staff/huaj/mlirdocker/llvm-project/build/bin/clang")
#set(CMAKE_CXX_COMPILER "/scratch/staff/huaj/mlirdocker/llvm-project/build/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${CMAKE_SOURCE_DIR}/../Parser)
file(GLOB PARSER_FILES ${CMAKE_SOURCE_DIR}/../Parser/*.cpp)
message("Parser src Directory: ${PARSER_FILES}")

#fiind antlr include
find_path(ANTLR4_INCLUDE_DIR antlr4-runtime/antlr4-runtime.h PATH_SUFFIXES include)
if(ANTLR4_INCLUDE_DIR)
    message("ANTLR4 Include Directory: ${ANTLR4_INCLUDE_DIR}")
    include_directories(${ANTLR4_INCLUDE_DIR}/antlr4-runtime/)
else()
    message(FATAL_ERROR "ANTLR4 runtime header not found.")
endif()
#find antlr lib
execute_process(COMMAND dpkg-query -L libantlr4-runtime-dev
                OUTPUT_VARIABLE ANTLR4_PACKAGE_FILES
                RESULT_VARIABLE DPkg_RESULT)

string(STRIP ${ANTLR4_PACKAGE_FILES} ANTLR4_PACKAGE_FILES)

string(REPLACE "\n" ";" ANTLR4_PACKAGE_FILES_LIST ${ANTLR4_PACKAGE_FILES})

foreach(FILE_PATH IN LISTS ANTLR4_PACKAGE_FILES_LIST)
    if(FILE_PATH MATCHES "\\.so$")
        list(APPEND ANTLR4_LIBRARY_PATHS_LIST ${FILE_PATH})
    endif()
endforeach()

message("ANTLR4 Library Paths: ${ANTLR4_LIBRARY_PATHS_LIST}")

set(SOURCE_FILES ./AieFrontEnd.cc)
list(APPEND SOURCE_FILES ./AieDialect.cc)

add_executable(mlirtest ${SOURCE_FILES} ${PARSER_FILES})
#add_executable(mlirtest ./AieDialect.cc)

#find_package(Curses REQUIRED)
#include_directories(${CURSES_INCLUDE_DIR})

#message(STATUS "curse include is ${CURSES_INCLUDE_DIR} lib is ${CURSES_LIBRARIES}")

set(LLVM_PATH /scratch/staff/huaj/mlirdocker/)
#set(LLVM_LIBRARY_DIR ${LLVM_PATH}/acompiler/download/llvm/lib/)
set(LLVM_LIBRARY_DIR ${LLVM_PATH}/llvm-project/build/lib/)
if(NOT MLIR_DIR)
	set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()
message(STATUS "${LLVM_LIBRARY_DIR}/cmake/mlir")
 
# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

message(STATUS "mlir include is ${MLIR_INCLUDE_DIRS} lib is ${MLIR_LIB_DIRS}")
#need -DLLVM_ENABLE_RTTI=ON to enable RTTI, as some thirdparty lib like antlr4 use RTTI
#and LLVM default disable RTTI , this will cause the typeinfo error without do the said
#enable
#target_compile_options(mlirtest PRIVATE -fno-rtti)
target_link_libraries(mlirtest PUBLIC
	${ANTLR4_LIBRARY_PATHS_LIST}
MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
	MLIRIR
	MLIRControlFlowToLLVM
	MLIRBytecodeWriter
	MLIRPass
	MLIRTransforms
	MLIRLLVMDialect
	MLIRSupport
	MLIRTargetLLVMIRExport
	MLIRMathToLLVM
			#MLIRROCDLToLLVMIRTranslation
			#MLIRGPUDialect
			# MLIRSCFToControlFlow
			#MLIRIndexToLLVM
			#MLIRGPUToROCDLTransforms
 
			#LLVM
     LLVMPasses
      LLVMNVPTXCodeGen
      #LLVMNVPTXAsmPrinter
       LLVMAMDGPUCodeGen
       LLVMAMDGPUAsmParser
      dl
)
target_link_options(mlirtest PRIVATE ${LLVM_LDFLAGS})

