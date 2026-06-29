// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT
//
// AIE sim test kernel: element-wise int32 add C[i] = A[i] + B[i]
// Used with reload.cc to demonstrate runtime kernel swap on the same tile.
//
// Memory layout matches matmul_kernel.cc (same DM offsets):
//   DM offset 0x2000 → kernel abs addr 0x72000 : A array  (16 int32)
//   DM offset 0x2040 → kernel abs addr 0x72040 : B array  (16 int32)
//   DM offset 0x2080 → kernel abs addr 0x72080 : C result (16 int32)
//   DM offset 0x20C0 → kernel abs addr 0x720C0 : done flag

#define DM_BASE    0x70000
#define N          16
#define A_ADDR     (DM_BASE + 0x2000)
#define B_ADDR     (DM_BASE + 0x2040)
#define C_ADDR     (DM_BASE + 0x2080)
#define DONE_ADDR  (DM_BASE + 0x20C0)
#define DONE_MAGIC 0xCAFECAFE

int main(void) {
    *(volatile unsigned int *)DONE_ADDR = 0;

    volatile int *A = (volatile int *)A_ADDR;
    volatile int *B = (volatile int *)B_ADDR;
    volatile int *C = (volatile int *)C_ADDR;

    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];

    *(volatile unsigned int *)DONE_ADDR = DONE_MAGIC;
    return 0;
}
