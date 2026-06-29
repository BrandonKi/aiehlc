// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT
//
// AIE sim test kernel: 4x4 int32 matrix multiply (direct DM access).
//
// Memory layout (tile local DM, from the host's perspective as DM offsets):
//   DM offset 0x2000 → kernel absolute addr 0x72000 : A matrix, 4x4 int32 (64 bytes)
//   DM offset 0x2040 → kernel absolute addr 0x72040 : B matrix, 4x4 int32 (64 bytes)
//   DM offset 0x2080 → kernel absolute addr 0x72080 : C matrix, 4x4 int32 (64 bytes)
//   DM offset 0x20C0 → kernel absolute addr 0x720C0 : done flag (0xCAFECAFE when done)
//
// AIE-ML core tile DM base = 0x70000 (from inside the core).
// Host uses DM offsets (0x2000, etc.) for XAie_DataMemBlockRead/MoveData*.

#define DM_BASE    0x70000   /* AIE-ML core tile DM base (from inside the core) */
#define N          4
#define A_ADDR     (DM_BASE + 0x2000)
#define B_ADDR     (DM_BASE + 0x2040)
#define C_ADDR     (DM_BASE + 0x2080)
#define DONE_ADDR  (DM_BASE + 0x20C0)
#define DONE_MAGIC 0xCAFECAFE

int main(void) {
    *(volatile unsigned int *)DONE_ADDR = 0;   /* clear done flag */

    volatile int *A = (volatile int *)A_ADDR;
    volatile int *B = (volatile int *)B_ADDR;
    volatile int *C = (volatile int *)C_ADDR;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }

    *(volatile unsigned int *)DONE_ADDR = DONE_MAGIC;   /* signal host */
    return 0;
}
