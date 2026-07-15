# Conv2d Im2col Design for AIEHLC

## Overview

This document describes how to add Conv2d support to the AIEHLC tilinglinalg pipeline by using the **im2col** (image-to-column) transformation to convert convolution into matrix multiplication (GEMM), then leveraging the existing GEMM pipeline with **DMA-accelerated im2col** via AIE multi-dimensional BD addressing.

## 1. Conv2d Operation

A 2D convolution slides a filter over an input feature map:

```
Input:  [H, W, C]      e.g. [8, 8, 1]     (NHWC layout, N=1 omitted)
Filter: [KH, KW, C, F] e.g. [3, 3, 1, 1]  (height, width, input channels, output filters)
Output: [OH, OW, F]    e.g. [6, 6, 1]      (OH = H-KH+1, OW = W-KW+1 for stride=1, pad=0)
```

General output dimensions:
```
OH = (H + 2*pad - KH) / stride + 1
OW = (W + 2*pad - KW) / stride + 1
```

In MLIR linalg, this is represented as:
```mlir
linalg.conv_2d_nhwc_hwcf
  ins(%input, %filter : tensor<1x8x8x1xf32>, tensor<3x3x1x1xf32>)
  outs(%output : tensor<1x6x6x1xf32>)
```

The naive implementation is a 6-nested loop:
```
for oc in [0, F):          // output channel
  for oh in [0, OH):       // output row
    for ow in [0, OW):     // output col
      acc = 0
      for ic in [0, C):    // input channel
        for kh in [0, KH): // kernel row
          for kw in [0, KW):  // kernel col
            acc += input[ic, oh*stride+kh, ow*stride+kw] * filter[kh, kw, ic, oc]
      output[oc, oh, ow] = acc
```

## 2. Im2col Transformation

Im2col converts convolution into matrix multiplication by rearranging the input data.

### Step 1: Extract Sliding Window Patches

For each output position `(oh, ow)`, extract the `KH * KW * C` patch from the input:

```
im2col_matrix[oh*OW + ow, :] = input[oh:oh+KH, ow:ow+KW, :].flatten()
```

Result: `im2col_matrix` has shape **[OH*OW, KH*KW*C]**

### Step 2: Reshape Filter

```
filter_matrix = filter.reshape(KH*KW*C, F)
```

Result: `filter_matrix` has shape **[KH*KW*C, F]**

### Step 3: Matrix Multiply

```
output_matrix = im2col_matrix @ filter_matrix
              = [OH*OW, KH*KW*C] x [KH*KW*C, F]
              = [OH*OW, F]
output = output_matrix.reshape(OH, OW, F)
```

### Concrete Examples

**Example 1: Simple 3x3 conv (target for first implementation)**
```
Input:  [8, 8, 1], Filter: [3, 3, 1, 1], stride=1, pad=0
OH = 6, OW = 6
Im2col matrix: [36, 9]    (36 output positions, 9 elements per patch)
Filter matrix: [9, 1]
Matmul:        [36, 9] x [9, 1] = [36, 1]
Reshape:       [6, 6, 1]
```

**Example 2: Multi-channel 3x3 conv**
```
Input:  [8, 8, 3], Filter: [3, 3, 3, 16], stride=1, pad=1
OH = 8, OW = 8  (same-padding)
Im2col matrix: [64, 27]   (64 positions, 27 = 3*3*3 elements per patch)
Filter matrix: [27, 16]
Matmul:        [64, 27] x [27, 16] = [64, 16]
Reshape:       [8, 8, 16]
```

**Example 3: ResNet conv1 (7x7 stride=2)**
```
Input:  [230, 230, 3], Filter: [7, 7, 3, 64], stride=2, pad=0
OH = 112, OW = 112
Im2col matrix: [12544, 147]  (12544 positions, 147 = 7*7*3)
Filter matrix: [147, 64]
Matmul:        [12544, 147] x [147, 64] = [12544, 64]
Reshape:       [112, 112, 64]
```

## 3. Why Im2col on AIE

### 3.1 Reuses Existing GEMM Pipeline

The tilinglinalg pipeline already handles matmul end-to-end:
- Tensor partitioning across mesh (`SplitModel::gemm()`)
- Routing, dataflow mapping, scheduling, DMA BD generation
- Host/kernel/routing code emission

Im2col converts conv2d into GEMM, so the entire pipeline works unchanged for the compute path.

### 3.2 AIE DMA Multi-Dim Addressing Does Im2col in Hardware

The key advantage: **AIE DMA can perform the im2col data rearrangement in hardware** using multi-dimensional BD addressing (up to 4 dimensions with stride/wrap per dimension). No need to materialize the im2col matrix in memory.

The existing `xaie_conv2d_2core_dataflow_test.c` already proves this with 4D descriptors:
```c
// From xaie_conv2d_2core_dataflow_test.c — MemTile output DMA
XAie_DmaDimDesc dims[4] = {
    {.AieMlDimDesc = {1, 7}},        // dim0: stride=1, wrap=7
    {.AieMlDimDesc = {230, 7}},       // dim1: stride=230 (W), wrap=7 (KH)
    {.AieMlDimDesc = {2530, 3}},      // dim2: stride=2530 (W*KH?), wrap=3 (C)
    {.AieMlDimDesc = {2, 0}},         // dim3: stride=2 (stride), wrap=0
};
```

### 3.3 Separation of Concerns

- **DMA** (shim/memtile): performs im2col patch extraction — data rearrangement
- **Core tile**: runs standard matmul kernel — compute
- **Host**: orchestrates DMA and kernel launch — control

This maps naturally to the AIE architecture where DMA engines handle data movement independently from compute cores.

## 4. DMA-Accelerated Im2col Design

### 4.1 DMA Descriptor Math

For an input `[H, W, C]` with filter `[KH, KW]`, stride `S`, the shim/memtile DMA BD extracts im2col patches using multi-dim addressing:

```
// Innermost dimensions extract one patch (KH * KW * C elements):
dim0: stride = 1,       wrap = KW        // scan KW elements within a row
dim1: stride = W,       wrap = KH        // jump to next row, repeat KH times
dim2: stride = W*KH,    wrap = C         // across input channels (if C > 1)

// Outer dimensions slide the window across output positions:
dim3: stride = S,        wrap = OW       // slide window right by stride, OW times

// For advancing down rows (OH iterations):
// Use BD iteration: iter_step_size = W * S * elementBytes, iter_wrap = OH
```

**For the target case (3x3, stride=1, [8,8,1] input, C=1):**
```
dim0: stride = 1,   wrap = 3    // 3 elements per row of kernel
dim1: stride = 8,   wrap = 3    // 3 rows (W=8 stride to next row)
dim2: stride = 1,   wrap = 6    // slide right, OW=6 positions

iter_step_size = 8 * 4 = 32 bytes (W * stride * sizeof(int32))
iter_wrap = 6  (OH=6 output rows)
```

This produces a stream of 36 patches, each 9 elements = 324 total elements, which is exactly the im2col matrix flattened row-by-row.

### 4.2 Existing Infrastructure

The pipeline already has the multi-dim DMA plumbing:

| Level | Op | Attributes | File |
|-------|----|------------|------|
| Blueprint | `TransferManifestOp` | `shim_dim_strides`, `shim_dim_wraps` | `dfscheblueprintop.td:203-204` |
| Schedule | `ConfigDmaBdOp` | `dim_strides`, `dim_wraps`, `iter_step_size`, `iter_wrap` | `dfscheduleop.td:167-175` |
| API emit | `DfscheduleToApiPass` | Generates `XAie_DmaSetMultiDimAddr` | `passdfscheduletoapi.cpp` |

The `BlueprintToSchedulePass` already propagates `shim_dim_strides`/`shim_dim_wraps` from blueprint to schedule. The `DfscheduleToApiPass` already emits `XAie_DmaSetMultiDimAddr` when these attributes are present. So we only need to **set** these attributes during IR construction.

## 5. Implementation Plan

### Step 1: Add `Conv2dParams` and `SplitModel::conv2d()`

**File**: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h`

```cpp
/// Conv2d operation parameters for im2col transformation.
struct Conv2dParams {
    int inputH, inputW, inputC;  // input spatial dimensions
    int kernelH, kernelW;        // filter spatial dimensions
    int numFilters;              // F (output channels)
    int stride;
    int pad;                     // zero-padding (0 for initial implementation)

    // Derived im2col dimensions (computed from above)
    int outputH() const { return (inputH + 2*pad - kernelH) / stride + 1; }
    int outputW() const { return (inputW + 2*pad - kernelW) / stride + 1; }
    int im2colM() const { return outputH() * outputW(); }        // OH*OW
    int im2colK() const { return kernelH * kernelW * inputC; }   // KH*KW*C
    int im2colN() const { return numFilters; }                    // F
};
```

Add to `SplitModel`:
```cpp
/// Factory: Conv2d via im2col (same distribution as GEMM)
static SplitModel conv2d() { return gemm(); }
```

The split model is identical to GEMM because after im2col, the problem IS a GEMM: im2col matrix is partitioned like A, filter like B, output like C.

### Step 2: Add `buildConv2dRoutingIR()`

**File**: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h` (declaration)
**File**: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp` (implementation)

```cpp
/// Build routing IR for Conv2d via im2col.
/// Converts conv2d parameters into GEMM-equivalent tensor dimensions,
/// attaches im2col DMA hints as module attributes, then calls buildRoutingIR.
static mlir::ModuleOp buildConv2dRoutingIR(
    mlir::MLIRContext &ctx,
    int meshRows, int meshCols,
    const Conv2dParams &conv,
    const PartitionDesc &partition = {},
    const std::string &aieGen = "Gen2");
```

Implementation:
1. Compute im2col dimensions: `M = OH*OW`, `K = KH*KW*C`, `N = F`
2. Create 3 TensorParams with GEMM shapes:
   - Input (im2col'd): `{M, K}`, isInput=true
   - Filter (reshaped): `{K, N}`, isInput=true
   - Output:            `{M, N}`, isInput=false
3. Call `buildRoutingIR(ctx, meshRows, meshCols, tensors, SplitModel::conv2d())`
4. Attach module attribute `conv2d.im2col_config`:
```cpp
module->setAttr("conv2d.im2col_config", DictionaryAttr::get(&ctx, {
    {"input_h",      builder.getI32IntegerAttr(conv.inputH)},
    {"input_w",      builder.getI32IntegerAttr(conv.inputW)},
    {"input_c",      builder.getI32IntegerAttr(conv.inputC)},
    {"kernel_h",     builder.getI32IntegerAttr(conv.kernelH)},
    {"kernel_w",     builder.getI32IntegerAttr(conv.kernelW)},
    {"stride",       builder.getI32IntegerAttr(conv.stride)},
    {"pad",          builder.getI32IntegerAttr(conv.pad)},
    {"num_filters",  builder.getI32IntegerAttr(conv.numFilters)},
    {"tensor_index", builder.getI32IntegerAttr(0)},  // input tensor
}));
```

### Step 3: Set Im2col DMA Descriptors in DmaphopTodfscheblueprintPass

**File**: `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`

When creating `TransferManifestOp` for the input tensor's shim tile, read the `conv2d.im2col_config` module attribute and set `shim_dim_strides` / `shim_dim_wraps`:

```cpp
// Check for conv2d im2col config on the module
if (auto im2colConfig = module->getAttrOfType<DictionaryAttr>("conv2d.im2col_config")) {
    int inputW  = im2colConfig.getAs<IntegerAttr>("input_w").getInt();
    int kernelH = im2colConfig.getAs<IntegerAttr>("kernel_h").getInt();
    int kernelW = im2colConfig.getAs<IntegerAttr>("kernel_w").getInt();
    int stride  = im2colConfig.getAs<IntegerAttr>("stride").getInt();
    int inputC  = im2colConfig.getAs<IntegerAttr>("input_c").getInt();
    int OW = (im2colConfig.getAs<IntegerAttr>("input_w").getInt() +
              2 * im2colConfig.getAs<IntegerAttr>("pad").getInt() - kernelW) / stride + 1;
    int OH = (im2colConfig.getAs<IntegerAttr>("input_h").getInt() +
              2 * im2colConfig.getAs<IntegerAttr>("pad").getInt() - kernelH) / stride + 1;

    // Build im2col DMA descriptors
    SmallVector<int32_t> dimStrides, dimWraps;
    if (inputC == 1) {
        // Single channel: 3 dimensions
        dimStrides = {1,        inputW,   stride};
        dimWraps   = {kernelW,  kernelH,  OW};
    } else {
        // Multi-channel: 4 dimensions (AIE max)
        dimStrides = {1,        inputW,   inputW*kernelH,  stride};
        dimWraps   = {kernelW,  kernelH,  inputC,          OW};
    }

    // Set on the TransferManifestOp for the input tensor's shim tile
    manifestOp.setShimDimStridesAttr(builder.getI32ArrayAttr(dimStrides));
    manifestOp.setShimDimWrapsAttr(builder.getI32ArrayAttr(dimWraps));
    // Set iteration for row advancement
    // iter_step_size = inputW * stride * elementBytes
    // iter_wrap = OH
}
```

### Step 4: Unit Test

**File**: `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp`

Add `testConv2d()` test function and CLI dispatch:
```cpp
void testConv2d(MLIRContext &ctx) {
    // 8x8x1 input, 3x3x1x1 filter, stride=1, pad=0
    Conv2dParams conv = {8, 8, 1, 3, 3, 1, 1, 0};
    auto module = TilingLinalgPipeline::buildConv2dRoutingIR(ctx, 2, 2, conv);
    std::string workDir = setupWorklocalDir();
    TilingLinalgPipeline::runPipeline(ctx, module, workDir, ...);
}

// In main():
if (testName == "conv2d") { testConv2d(ctx); return 0; }
```

### Step 5: Conv2d Example

**File**: `example/tileprogram/ccode/simpleconv2d.cc` (new file)

```cpp
// Conv2d via im2col: the kernel receives pre-rearranged data from DMA
// and performs standard matrix multiplication.
//
// Input:  [8, 8, 1] — 8x8 single-channel image
// Filter: [3, 3, 1, 1] — 3x3 single filter
// Output: [6, 6, 1] — 6x6 output
//
// After im2col by DMA:
//   im2col matrix: [36, 9]  (A)
//   filter matrix: [9, 1]   (B)
//   output:        [36, 1]  (C)
//   C = A * B

__global__ void conv2d_matmul(
    input_window_int32 *window_im2col,
    input_window_int32 *window_filter,
    output_window_int32 *window_out
) {
    int32_t *A = (int32_t *)acquire_input_window(window_im2col);
    int32_t *B = (int32_t *)acquire_input_window(window_filter);
    int32_t *C = (int32_t *)acquire_output_window(window_out);

    // Standard matmul: C[i,j] += A[i,k] * B[k,j]
    // Dimensions: M=36/num_tiles, K=9, N=1
    for (int i = 0; i < TILE_M; i++) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            acc += A[i * K + k] * B[k];
        }
        C[i] = acc;
    }

    release_input_window(window_im2col);
    release_input_window(window_filter);
    release_output_window(window_out);
}

int main() {
    aieDim mesh(2, 2);
    int32_t input[8*8], filter[3*3], output[6*6];

    // Fill input with test data
    for (int i = 0; i < 64; i++) input[i] = i + 1;
    for (int i = 0; i < 9; i++) filter[i] = 1;
    memset(output, 0, sizeof(output));

    conv2d_matmul<<<mesh>>>(input, sizeof(input),
                             filter, sizeof(filter),
                             output, sizeof(output));

    // Verify: each output[oh][ow] = sum of 3x3 patch (filter is all 1s)
    return 0;
}
```

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Host (DDR Memory)                           │
│                                                                 │
│  input[8][8]          filter[3][3]         output[6][6]         │
│  (original layout)    (flat layout)        (result)             │
└──────┬────────────────────┬────────────────────┬────────────────┘
       │ Shim DMA           │ Shim DMA           │ Shim DMA
       │ (multi-dim BD      │ (flat BD)          │ (flat BD)
       │  does im2col)      │                    │
       ▼                    ▼                    ▲
┌──────────────────────────────────────────────────────────────────┐
│                     Stream Switch Network                       │
│                                                                 │
│  im2col patches         filter data          output data        │
│  [9 elem/patch          [9 elements]         [tile_M elements]  │
│   x tile_M patches]                                             │
└──────┬────────────────────┬────────────────────┬────────────────┘
       │                    │                    │
       ▼                    ▼                    ▲
┌──────────────────────────────────────────────────────────────────┐
│                     Core Tile (row R, col C)                    │
│                                                                 │
│  Kernel: standard matmul                                        │
│  C[i] = sum_k( A[i*K + k] * B[k] )                            │
│                                                                 │
│  A = im2col patches (tile_M x K)                               │
│  B = filter (K x 1)                                            │
│  C = output (tile_M x 1)                                       │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Pipeline Flow

```
buildConv2dRoutingIR()
  │  Computes im2col dimensions: M=36, K=9, N=1
  │  Creates TensorParams: {36,9}, {9,1}, {36,1}
  │  Attaches conv2d.im2col_config module attribute
  │  Calls buildRoutingIR() with GEMM-equivalent shapes
  ▼
RoutingUnrollingLowerPass        (unchanged — unrolls routing ops)
  ▼
RoutingToDmapPass                (unchanged — routing → logical dataflow)
  ▼
DmapToDmaphopPass                (unchanged — logical → physical hops)
  ▼
DmaphopTodfscheblueprintPass     ★ MODIFIED: reads conv2d.im2col_config,
  │                                sets shim_dim_strides/wraps on input
  │                                tensor's TransferManifestOp
  ▼
BlueprintToSchedulePass          (unchanged — propagates dim_strides/wraps
  │                                to ConfigDmaBdOp)
  ▼
DfscheduleToApiPass              (unchanged — emits XAie_DmaSetMultiDimAddr
  │                                when dim_strides/wraps present)
  ▼
EmitC → host.cc                  (shim BD has multi-dim im2col addressing)
         kernel.cc               (standard matmul — no conv2d awareness)
         routing.cc              (stream switch configuration)
```

## 8. Files Summary

### Modified Files
| File | Change |
|------|--------|
| `pass/tilinglinalg_pipeline.h` | Add `Conv2dParams`, `SplitModel::conv2d()`, `buildConv2dRoutingIR()` |
| `pass/tilinglinalg_pipeline.cpp` | Implement `buildConv2dRoutingIR()` |
| `pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp` | Read im2col config, set `shim_dim_strides`/`shim_dim_wraps` |
| `pass/unitest/test.cpp` | Add `testConv2d()` and CLI dispatch |

### New Files
| File | Purpose |
|------|---------|
| `example/tileprogram/ccode/simpleconv2d.cc` | Conv2d example using `kernel<<<mesh>>>` syntax |

### Unchanged Files (already support multi-dim DMA)
- `passblueprinttoschedule.cpp` — propagates shim_dim_strides/wraps to schedule
- `passdfscheduletoapi.cpp` — emits `XAie_DmaSetMultiDimAddr`
- `routing/routingmanager.cpp` — `createroutingfuncBySplitModel` (GEMM split works for conv2d)

## 9. Verification

1. **Build**: `cd pass/unitest && mkdir -p build && cd build && cmake .. && make -j4`
2. **Run**: `./test conv2d` → generates `worklocal/{host.cc, kernel.cc, routing.cc}`
3. **Check host.cc**: Shim BD for input tensor should have:
   - `XAie_DmaSetMultiDimAddr` with strides `{1, 8, 1}` and wraps `{3, 3, 6}`
   - `XAie_DmaSetBdIteration` with `iter_step_size=32`, `iter_wrap=6`
4. **Check kernel.cc**: Should be standard matmul (no conv2d-specific code)
5. **CPU validation**: `conv2d_naive(input, filter)` == `matmul(im2col(input), reshape(filter))`
6. **HW run**: Deploy via `piplinerun.sh` on AIE board

## 10. Future Extensions

- **Padding support**: Zero-padding via `XAie_DmaSetPadding` on the BD (already supported by the driver)
- **Stride > 1**: Adjust dim2 stride from `1` to `S` in the im2col DMA descriptor
- **Multi-channel**: Use 4D descriptor (dim2 for channel iteration)
- **Multiple filters**: N > 1, filter broadcast across mesh columns (already handled by GEMM split)
- **MemTile caching**: Use MemTile as im2col buffer for large inputs (matches `xaie_conv2d_2core_dataflow_test.c` pattern)
- **Fused ops**: Conv2d + BN + ReLU fusion at the kernel level (matches `conv2d_fuse_relu.cc` pattern)

## 11. Revised Architecture: Generic DmaAddressing (replaces conv2d-specific approach)

### 11.1 Key Insight: Conv2d Doesn't Need Special Routing

After im2col, conv2d IS a GEMM. The routing topology, mesh partitioning, broadcast/gather flows, packet IDs, DMA hops — all identical to `simplematmul.cc`. The pipeline doesn't know or care it's conv2d until the very last moment when `DmaphopTodfscheblueprintPass` sets the shim BD's multi-dim strides/wraps instead of flat addressing.

Therefore:
- **`SplitModel::conv2d()`** is unnecessary — `SplitModel::gemm()` works directly
- **`buildConv2dRoutingIR()`** is unnecessary — the caller just calls `buildRoutingIR()` with GEMM-shaped tensors
- **No new spatial types** — SpatialPolicy describes data distribution (broadcast, gather), which is the same for conv2d and matmul. Im2col is a DMA addressing concern, orthogonal to spatial distribution.

### 11.2 The Real Problem: Non-Contiguous DMA Addressing

Conv2d im2col is one instance of a general problem: the shim DMA sometimes needs to read DDR data in a non-contiguous pattern. Other instances include:

| Operation | DDR Buffer | Kernel Sees | DMA Pattern |
|-----------|-----------|-------------|-------------|
| Im2col | `[H, W, C]` | `[OH*OW, KH*KW*C]` | Overlapping patches |
| Transpose | `[M, N]` | `[N, M]` | Column-major read |
| Strided read | `[H, W]` | `[H/S, W/S]` | Skip every S-th element |
| Pooling | `[H, W, C]` | `[OH*OW, KH*KW*C]` | Similar to im2col |
| Depthwise conv | `[H, W, C]` | per-channel patches | Channel-separated |

All of these are just different stride/wrap configurations on the shim DMA's multi-dim BD. The pipeline should handle **generic DMA addressing**, not conv2d-specific logic.

### 11.3 Unified Solution: Generic `DmaAddressing` on `TensorParam`

```cpp
/// Generic multi-dim DMA descriptor.
/// Works for im2col, transpose, strided reads, pooling, etc.
/// The pipeline never knows what operation produced these — it just
/// sees "tensor N has non-flat DMA addressing, here are the dims."
struct DmaAddressing {
    std::vector<std::pair<int,int>> dims;  // {stride, wrap} pairs, up to 3-4 dims
    int iter_step = 0;                      // BD iteration step size (bytes)
    int iter_wrap = 0;                      // BD iteration count
    std::vector<int64_t> ddrShape;          // actual DDR buffer shape (may differ from GEMM shape)
};

struct TensorParam {
    std::vector<int64_t> shape;          // GEMM shape [36,9] — used for tiling/partitioning
    int elementBitWidth;                 // e.g. 8 for int8_t
    bool isInput;
    DmaAddressing shimDma;               // optional: non-flat DDR addressing
};
```

### 11.4 Data Flow

```
                     operation-aware         generic from here on
                    ┌────────────────┐      ┌──────────────────────────────┐
User/Frontend       │  Caller        │      │  Pipeline                    │
{H,W,KH,KW,S,P} ──►│  computes      │      │                              │
  or                │  strides/wraps │      │  buildRoutingIR()            │
{transpose dims}    │  into generic  │──►   │  stores DmaAddressing as     │
  or                │  DmaAddressing │      │  per-tensor module attribute │
{pool params}       │                │      │  "tensor_N.shim_dma"         │
                    └────────────────┘      │                              │
                                            │  DmaphopTodfscheblueprintPass│
                                            │  reads generic strides/wraps │
                                            │  sets on FlowConfigOp        │
                                            │                              │
                                            │  (no conv2d / transpose /    │
                                            │   pooling knowledge here)    │
                                            └──────────────────────────────┘
```

### 11.5 Caller-Side Helpers (operation-specific knowledge isolated here)

```cpp
/// Im2col DMA addressing — the ONLY place with conv2d knowledge.
/// Lives in the caller (frontend, test), NOT in the pipeline.
DmaAddressing im2colAddressing(int H, int W, int C, int KH, int KW, int S, int P) {
    int OH = (H + 2*P - KH) / S + 1;
    int OW = (W + 2*P - KW) / S + 1;
    DmaAddressing addr;
    if (C == 1) {
        addr.dims = {{1, KW}, {W, KH}, {S, OW}};
    } else {
        addr.dims = {{1, KW}, {W, KH}, {W*KH, C}, {S, OW}};
    }
    addr.iter_step = W * S;   // in elements (pipeline multiplies by elementBytes)
    addr.iter_wrap = OH;
    addr.ddrShape = {H, W, C};
    return addr;
}

/// Transpose DMA addressing — same generic mechanism, different strides.
DmaAddressing transposeAddressing(int M, int N) {
    return {
        .dims = {{M, N}},     // stride=M (column-major), wrap=N columns
        .iter_step = 1,
        .iter_wrap = M,
        .ddrShape = {M, N}
    };
}
```

### 11.6 Usage Example

```cpp
// Conv2d caller — computes GEMM dims + DMA addressing, pipeline sees generic strides
int M = OH * OW, K = KH * KW * C, N = F;

auto tensors = {
    TensorParam{{M, K}, 8, true,  im2colAddressing(8,8,1, 3,3, 1,0)},  // input: im2col DMA
    TensorParam{{K, N}, 8, true,  {}},                                   // filter: flat DMA
    TensorParam{{M, N}, 8, false, {}}                                    // output: flat DMA
};

// Same buildRoutingIR as matmul — routing doesn't care it's conv2d
auto module = TilingLinalgPipeline::buildRoutingIR(ctx, 2, 2, tensors, SplitModel::gemm());
TilingLinalgPipeline::runPipeline(ctx, module, outputDir, ...);
```

### 11.7 Pipeline Changes (Minimal)

| Component | Change |
|-----------|--------|
| `TensorParam` | Add `DmaAddressing shimDma` field |
| `buildRoutingIR()` | Store non-empty `shimDma` as `"tensor_N.shim_dma"` module attribute |
| `DmaphopTodfscheblueprintPass` | Read `"tensor_N.shim_dma"`, set `shim_dim_strides`/`shim_dim_wraps` on `FlowConfigOp` |

**Everything else unchanged**: no new dialects, no new ops, no new TD attributes, no SpatialPolicy changes, no `BlueprintToSchedulePass` changes, no `DfscheduleToApiPass` changes.

### 11.8 Why This Is Better Than the Conv2d-Specific Approach (Sections 5-8)

| Concern | Conv2d-specific (Section 5) | Generic DmaAddressing (Section 11) |
|---------|---------------------------|-------------------------------------|
| Pipeline knows conv2d? | Yes — reads `conv2d.im2col_config` | No — reads generic strides/wraps |
| Supports transpose? | No — needs another module attr | Yes — different strides/wraps |
| Supports pooling? | No — needs another module attr | Yes — different strides/wraps |
| New functions? | `buildConv2dRoutingIR()` | None — existing `buildRoutingIR()` |
| IR changes? | Module attr per op type | One `DmaAddressing` for all |
| Pass changes? | Per-op-type logic in pass | One generic read path |
| Operation knowledge in pipeline? | Yes | No — isolated to caller helpers |

### 11.9 Conv2d Example File

Created at `example/tileprogram/ccode/simpleconv2d.cc` with matching header `simpleconv2d.h`:

- **Kernel**: identical to `simplematmul.cc` — same `aie::port`, `aie::SpatialPolicy`, `acquire`/`release` pattern. The kernel name is `conv2d_im2col` but the code is pure matmul.
- **Host**: allocates input `[8,8,1]`, filter `[3,3,1,1]`, output `[6,6,1]`; launches kernel with GEMM dims M=36, K=9, N=1.
- **Verification**: three-level — im2col equivalence check (im2col+matmul == naive conv2d), naive conv2d reference, and AIE output comparison.
- **Data type**: `int8_t` with `int16_t` accumulation and saturation, matching `simplematmul.cc`.

### 11.10 Revised Files Summary

| File | Change |
|------|--------|
| `pass/tilinglinalg_pipeline.h` | Add `DmaAddressing` struct; add `shimDma` field to `TensorParam` |
| `pass/tilinglinalg_pipeline.cpp` | `buildRoutingIR()`: store `shimDma` as module attr when non-empty |
| `pass/passdmaphoptodfscheblueprint.cpp` | Read generic `"tensor_N.shim_dma"` attr, set strides/wraps on `FlowConfigOp` |
| `pass/unitest/test.cpp` | Add `testConv2d()` using `im2colAddressing()` helper + `buildRoutingIR()` |
| `example/tileprogram/ccode/simpleconv2d.cc` | Conv2d example (parameterized kernel, int8) |
| `example/tileprogram/ccode/simpleconv2d.h` | Conv2d params, im2col utility, verification |

## 12. Halo Handling: Why Im2col Eliminates It

### 12.1 The Halo Problem in Traditional Spatial Tiling

In traditional spatial tiling of conv2d (without im2col), the input image is split across tiles. Each tile owns a contiguous region of input rows. Because the convolution kernel overlaps tile boundaries, neighboring tiles need extra "halo" rows from each other:

```
Traditional spatial split of input[8,8] with KH=3, stride=1:

  Tile 0 owns rows 0-3 → needs rows 0-5 to compute outputs for oh=0..3
                          must GET rows 4-5 from tile 1 (halo)
  Tile 1 owns rows 4-7 → needs rows 2-7 to compute outputs for oh=3..5
                          must GET rows 2-3 from tile 0 (halo)

  Overlap = KH - stride = 3 - 1 = 2 rows at each boundary
  → requires inter-tile communication, halo buffers, synchronization
```

This is complex, error-prone, and wastes local memory on duplicated halo data.

### 12.2 Im2col Eliminates Halo at the Spatial Level

With im2col, the DMA extracts complete patches from DDR before any tiling occurs. After im2col, the problem is a standard GEMM where each row of A is a self-contained patch with no dependency on any other row:

```
Im2col spatial split (2x2 mesh, M=36):

  Tile row 0: receives im2col rows 0-17   (patches for output positions 0-17)
  Tile row 1: receives im2col rows 18-35  (patches for output positions 18-35)

  Each patch is complete (9 elements) — no overlap between tile partitions
  → zero halo, zero inter-tile communication
```

The "overlap" still exists in the source data (neighboring patches share input elements), but it's handled by the DMA re-reading from DDR — not by inter-tile exchange.

### 12.3 Halo Is Also Not a Problem with tile_m Temporal Tiling

When `tile_m` splits the per-core work into multiple kernel launches, the same principle holds. Each round's DMA independently reads its patches from DDR:

```
tile_m = 6 (= OW), 6 rounds per core:

Round 0: output row oh=0, DMA reads input rows 0,1,2
Round 1: output row oh=1, DMA reads input rows 1,2,3
Round 2: output row oh=2, DMA reads input rows 2,3,4
Round 3: output row oh=3, DMA reads input rows 3,4,5
Round 4: output row oh=4, DMA reads input rows 4,5,6
Round 5: output row oh=5, DMA reads input rows 5,6,7
```

Input rows 1-6 are each read by multiple rounds. But:
- Each round is a separate host-launched kernel invocation
- Each round's DMA BD points to DDR with an adjusted base address
- No round depends on another round's data — all reads come from DDR
- No halo exchange, no inter-round communication

### 12.4 Total DMA Volume Is Unchanged by tile_m

The total number of DMA reads from DDR is the same regardless of tile_m:

```
No tile_m:     36 patches × 9 elements = 324 reads, 1 continuous stream
tile_m=6:       6 patches × 9 elements =  54 reads/round × 6 rounds = 324 total
tile_m=12:     12 patches × 9 elements = 108 reads/round × 3 rounds = 324 total
tile_m=18:     18 patches × 9 elements = 162 reads/round × 2 rounds = 324 total
```

The overlap is inherent to convolution (324 element reads from 64 input elements ≈ 5x redundancy). tile_m doesn't increase it — it just distributes the same redundant reads across rounds.

### 12.5 Comparison Summary

| Approach | Halo? | How overlap is handled | Complexity |
|----------|-------|----------------------|------------|
| Spatial tiling (no im2col) | Yes | Inter-tile exchange: tiles send/receive halo rows | High — requires halo buffers, sync, communication |
| Im2col, no tile_m | No | DMA re-reads overlapping elements from DDR | None — DMA handles it transparently |
| Im2col + tile_m | No | DMA re-reads per round from DDR | None — each round is independent |

### 12.6 The Trade-off

Im2col trades **inter-tile communication** for **redundant DDR reads**:

- **Without im2col**: each input element is read once from DDR, but tiles must exchange halo data through the stream network or shared memory
- **With im2col**: overlapping input elements are read multiple times from DDR, but no inter-tile communication is needed

This trade-off favors im2col on AIE because:
1. DDR bandwidth via NoC is high and DMA engines handle re-reads without CPU intervention
2. Inter-tile halo exchange would consume stream switch bandwidth and require synchronization
3. The kernel stays a simple matmul — no halo-aware boundary logic

### 12.7 tile_m Alignment Constraint

The only constraint tile_m introduces for conv2d is alignment:

```
tile_m must be a multiple of OW (output width)
```

This is because the im2col DMA sweeps OW positions horizontally before advancing to the next output row. Partial output rows would require a different DMA descriptor mid-stream, which AIE multi-dim BD doesn't support.

Valid tile_m values for the 8x8x1 / 3x3 example (OW=6):
```
tile_m = 6   → 1 output row per round,  iter_wrap = 1
tile_m = 12  → 2 output rows per round, iter_wrap = 2
tile_m = 18  → 3 output rows per round, iter_wrap = 3  (= tileRows for 2-row mesh)
tile_m = 36  → all 6 output rows,       iter_wrap = 6  (no temporal tiling)
```

This constraint should be enforced by the `DmaAddressing` struct:

```cpp
struct DmaAddressing {
    std::vector<std::pair<int,int>> dims;
    int iter_step = 0;
    int iter_wrap = 0;
    std::vector<int64_t> ddrShape;
    int tile_m_alignment = 1;  // tile_m must be multiple of this (OW for im2col)
};
```

## 13. General DmaTransform Architecture (Implemented)

The `DmaTransform` type provides a general, hardware-aligned abstraction for multi-dimensional DMA addressing. It mirrors the AIEML BD hardware capabilities (up to 4 dims + iteration) and is operation-agnostic -- the pipeline sees only strides and wraps, never "im2col" or "transpose".

### Type Design

```cpp
// In aie:: namespace (emitted by aiehlc.cc)
struct DmaTransform {
    struct Dim { int stride; int wrap; };
    Dim dims[4] = {};       // up to 4 dims (AIEML hardware limit)
    int num_dims = 0;
    int iter_step = 0;      // BD iteration step
    int iter_wrap = 0;      // BD iteration count

    static constexpr DmaTransform flat() { return {}; }
    static constexpr DmaTransform im2col(int H, int W, int C,
                                          int KH, int KW, int S, int P);
    static constexpr DmaTransform transpose(int rows, int cols);
};

// port: 3 template params, DmaTransform defaults to flat()
template<typename T, SpatialPolicy P, DmaTransform D = DmaTransform::flat()>
struct port { using type = T; };
```

### User API

```cpp
// matmul -- flat DMA (default, no 3rd param):
aie::port<input_window_int8 *, RowBC> win_a;

// conv2d -- im2col factory:
constexpr auto im2col = aie::DmaTransform::im2col(8, 8, 1, 3, 3, 1, 0);
aie::port<input_window_int8 *, RowBC, im2col> win_a;

// transpose (future):
constexpr auto xpose = aie::DmaTransform::transpose(64, 64);
aie::port<input_window_int8 *, RowBC, xpose> win_a;
```

### Pipeline Integration

```
User C++ (simpleconv2d.cc)
  constexpr auto im2col = aie::DmaTransform::im2col(8,8,1, 3,3, 1,0);
  aie::port<T*, RowBC, im2col> win_a;
       |
       v  Clang constexpr evaluation
  DmaTransform{ dims=[{1,3},{8,3},{1,6}], num_dims=3, iter={8,6} }
       |
       v  aiehlc.cc AST extraction (targs[2])
  ParsedTensorInfo.shimDma = {dims, iter_step, iter_wrap}
       |
       v  aiehlc.cc TensorParam construction
  TensorParam{ shape, bitWidth, isInput, shimDma }
       |
       v  tilinglinalg_pipeline.cpp buildRoutingIR()
  module->setAttr("tensor_0.shim_dma", {strides, wraps, iter_step, iter_wrap})
       |
       v  passdmaphoptodfscheblueprint.cpp
  FlowConfigOp gets shim_dim_strides, shim_dim_wraps from module attr
       |
       v  (existing infrastructure -- no changes below)
  BlueprintToSchedulePass -> DfscheduleToApiPass -> host.cc
  XAie_DmaSetMultiDimAddr(...)
```

### Files Modified

| File | Change |
|------|--------|
| `src/llvm/aiehlc.cc` | Emit DmaTransform struct (3 locations); port template 3 params; extract targs[2]; build shimDma on TensorParam |
| `src/mlir/.../tilinglinalg_pipeline.h` | Add `DmaAddressing` struct; add `shimDma` to `TensorParam` |
| `src/mlir/.../tilinglinalg_pipeline.cpp` | Store shimDma as `tensor_N.shim_dma` module attr; emit DmaTransform in single-kernel host.cc |
| `src/mlir/.../passdmaphoptodfscheblueprint.cpp` | Read `tensor_N.shim_dma` attr; set strides/wraps on FlowConfigOp |
| `example/tileprogram/ccode/simpleconv2d.cc` | Use `DmaTransform::im2col(...)` on input port |
| `src/mlir/.../pass/unitest/test.cpp` | Add `testConv2d()` command |

### Properties

| Property | Benefit |
|----------|---------|
| Hardware-aligned | `DmaTransform` = AIEML BD descriptor: 4 dims + iteration |
| Operation-agnostic | Pipeline sees only strides/wraps, never "im2col" or "transpose" |
| User-friendly | Factories hide stride/wrap math |
| Uniform AST extraction | Always same struct layout, factory-independent |
| Extensible | Add `pool()`, `dilated_conv()` -- no type/pipeline changes |
| Backward compatible | `SpatialPolicy` unchanged; matmul code needs no changes |
