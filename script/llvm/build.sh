###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

cd ./build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_TERMINFO=OFF \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_CCACHE_BUILD=ON \
	 -DLLVM_ENABLE_RTTI=ON
   #-DLLVM_ENABLE_PROJECTS=clang
cmake --build . --target check-mlir
