/* Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 *
 * Shim header for aie-rt xaie_sim.c: declares the ess_* ESS (Execution
 * Simulation System) interface that our PS wrapper (ioutils.cpp) implements.
 * This replaces the Vitis-supplied main_rts.h which conflicts with our
 * aie-rt internal headers.
 */
#ifndef MAIN_RTS_IO_H
#define MAIN_RTS_IO_H

#include <stdint.h>

typedef unsigned int uint;

void ess_Write32(uint64_t Addr, uint Data);
uint ess_Read32(uint64_t Addr);

void ess_Write128(uint64_t Addr, uint32_t *Data);
void ess_Read128(uint64_t Addr, uint32_t *Data);

void ess_WriteCmd(unsigned char Command, unsigned char ColId, unsigned char RowId, unsigned int CmdWd0,
                  unsigned int CmdWd1, unsigned char *CmdStr);

void ess_NpiWrite32(uint64_t Addr, uint Data);
uint ess_NpiRead32(uint64_t Addr);

void ess_WriteGM(uint64_t addr, const void *data, uint64_t size);
void ess_ReadGM(uint64_t addr, void *data, uint64_t size);

void start_plios(void);

/* uc_driver stubs for PS-side simulation (uc_driver/*.c uses reg_read32
 * which dereferences raw hardware addresses - unsafe on x86-64 PS side).
 * The ISS resolves these symbols from our .so; stubs prevent the crash. */
void uc_dma_init(void);
void uc_dma_process_responses(void);
void uc_dma_process_dm2mm_responses(void);
void uc_dma_create_bd(void *, unsigned short, unsigned int, unsigned int, unsigned int, unsigned int);
void uc_dma_create_bd_simple(void *, unsigned short, unsigned int, unsigned int);
unsigned short uc_dma_dm2mm_push_task(unsigned int bd_ptr);
unsigned short uc_dma_mm2dm_push_task(unsigned int bd_ptr);
int uc_dma_atomic_op(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);

#endif /* MAIN_RTS_IO_H */
