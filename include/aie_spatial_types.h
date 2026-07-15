/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef AIE_SPATIAL_TYPES_H
#define AIE_SPATIAL_TYPES_H

/// Spatial type wrappers for __global__ kernel parameters.
///
/// These struct templates annotate tensor parameters with their mesh
/// partitioning strategy — they control which mesh axis owns and replicates
/// the tensor, NOT which tensor dimension to split. splitDim always defaults
/// to 0; the spatial tag determines only hwAxisOwner and replicateOn.
///
///   __global__ void matmul(
///       aie::row_broadcast_in<input_window_int8*>  A,
///       aie::col_broadcast_in<input_window_int8*>  B,
///       aie::row_major_out<output_window_int8*>    C);
///
/// The Clang AST preserves the wrapper type (ClassTemplateSpecializationType)
/// so the compiler can extract both the spatial tag name and the inner type T.
///
/// Bare types (without wrapper) default to row_broadcast_in (input) or
/// row_major_out (output) for backward compatibility.

namespace aie {

// --- Input distribution types ---

/// Partition mesh by row, broadcast input to each row of tiles.
/// splitDim=0 (default), hwAxisOwner="row", replicateOn="col"
template <typename T> struct row_broadcast_in {
    using type = T;
};

/// Partition mesh by col, broadcast input to each col of tiles.
/// splitDim=0 (default), hwAxisOwner="col", replicateOn="row"
template <typename T> struct col_broadcast_in {
    using type = T;
};

/// 1-to-1 unique tile mapping (no broadcast).
/// splitDim=0, hwAxisOwner="row", replicateOn=""
template <typename T> struct tiled_in {
    using type = T;
};

// --- Output recomposition types ---

/// Gather output in row-major order.
/// splitDim=0 (default), hwAxisOwner="row", replicateOn="col"
template <typename T> struct row_major_out {
    using type = T;
};

/// Gather output in col-major order.
/// splitDim=0 (default), hwAxisOwner="col", replicateOn="row"
template <typename T> struct col_major_out {
    using type = T;
};

/// Cascade reduction along row axis.
/// splitDim=0, hwAxisOwner="row", replicateOn=""
template <typename T> struct row_reduce_out {
    using type = T;
};

} // namespace aie

#endif // AIE_SPATIAL_TYPES_H
