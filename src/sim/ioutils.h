// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>

void setWrite32Ptr   (void (*func)(uint64_t, unsigned int));
void setRead32Ptr    (unsigned int (*func)(uint64_t));
void setWriteCmdPtr  (void (*func)(unsigned char, unsigned char, unsigned char,
                                   unsigned int, unsigned int, unsigned char*));
void setNpiWrite32Ptr(void (*func)(uint64_t, unsigned int));
void setNpiRead32Ptr (unsigned int (*func)(uint64_t));
void setWriteGMPtr   (void (*func)(uint64_t, const void*, uint64_t));
void setReadGMPtr    (void (*func)(uint64_t, void*, uint64_t));
void setPLIOStartPtr (void (*func)());

extern "C" {
void         ess_Write32   (uint64_t Addr, unsigned int Data);
unsigned int ess_Read32    (uint64_t Addr);
void         ess_WriteGM   (uint64_t addr, const void* data, uint64_t size);
void         ess_ReadGM    (uint64_t addr, void* data, uint64_t size);
void         ess_NpiWrite32(uint64_t Addr, unsigned int Data);
unsigned int ess_NpiRead32 (uint64_t Addr);
void         ess_WriteCmd  (unsigned char Command,
                            unsigned char ColId, unsigned char RowId,
                            unsigned int CmdWd0, unsigned int CmdWd1,
                            unsigned char* CmdStr);
void         ess_debug     (const char* msg);

void XPl_Write32(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint32_t value);
void XPl_Write64(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint64_t value);
void XPl_Read32 (uint64_t base, uint32_t stride, int idx, uint32_t offset, uint32_t& value);
void XPl_Read64 (uint64_t base, uint32_t stride, int idx, uint32_t offset, uint64_t& value);
void XPl_IOstart();
} // extern "C"
