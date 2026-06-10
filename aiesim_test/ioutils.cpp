// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ioutils.h"
#include <cstdio>

static void          (*s_Write32)(uint64_t, unsigned int)                      = nullptr;
static unsigned int  (*s_Read32)(uint64_t)                                     = nullptr;
static void          (*s_WriteCmd)(unsigned char, unsigned char, unsigned char,
                                   unsigned int, unsigned int, unsigned char*) = nullptr;
static void          (*s_NpiWrite32)(uint64_t, unsigned int)                   = nullptr;
static unsigned int  (*s_NpiRead32)(uint64_t)                                  = nullptr;
static void          (*s_WriteGM)(uint64_t, const void*, uint64_t)             = nullptr;
static void          (*s_ReadGM)(uint64_t, void*, uint64_t)                    = nullptr;
static void          (*s_PLIOStart)()                                          = nullptr;

void setWrite32Ptr(void (*f)(uint64_t, unsigned int)) { s_Write32 = f; }
void setRead32Ptr(unsigned int (*f)(uint64_t))        { s_Read32  = f; }
void setWriteCmdPtr(void (*f)(unsigned char, unsigned char, unsigned char,
                               unsigned int, unsigned int, unsigned char*)) {
    s_WriteCmd = f;
}
void setNpiWrite32Ptr(void (*f)(uint64_t, unsigned int))  { s_NpiWrite32 = f; }
void setNpiRead32Ptr (unsigned int (*f)(uint64_t))        { s_NpiRead32  = f; }
void setWriteGMPtr   (void (*f)(uint64_t, const void*, uint64_t)) { s_WriteGM = f; }
void setReadGMPtr    (void (*f)(uint64_t, void*, uint64_t))       { s_ReadGM  = f; }
void setPLIOStartPtr (void (*f)()) { s_PLIOStart = f; }

extern "C" {

void         ess_Write32   (uint64_t a, unsigned int d) { if (s_Write32) s_Write32(a,d); }
unsigned int ess_Read32    (uint64_t a)                 { return s_Read32 ? s_Read32(a) : 0; }
void         ess_WriteGM   (uint64_t a, const void* d, uint64_t n) { if (s_WriteGM) s_WriteGM(a,d,n); }
void         ess_ReadGM    (uint64_t a, void* d, uint64_t n)       { if (s_ReadGM) s_ReadGM(a,d,n); }
void         ess_NpiWrite32(uint64_t a, unsigned int d) { if (s_NpiWrite32) s_NpiWrite32(a,d); }
unsigned int ess_NpiRead32 (uint64_t a)                 { return s_NpiRead32 ? s_NpiRead32(a) : 0; }
void         ess_WriteCmd  (unsigned char c, unsigned char col, unsigned char row,
                            unsigned int w0, unsigned int w1, unsigned char* s) {
    if (s_WriteCmd) s_WriteCmd(c,col,row,w0,w1,s);
}
void ess_debug(const char* msg) {
    fprintf(stderr, "[aeg] %s\n", msg ? msg : "");
}

void XPl_Write32(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint32_t value) {
    ess_Write32(base + (uint64_t)idx * stride + offset, value);
}
void XPl_Write64(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint64_t value) {
    uint64_t addr = base + (uint64_t)idx * stride + offset;
    ess_Write32(addr,     *((uint32_t*)&value));
    ess_Write32(addr + 4, *(((uint32_t*)&value) + 1));
}
void XPl_Read32(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint32_t& value) {
    value = ess_Read32(base + (uint64_t)idx * stride + offset);
}
void XPl_Read64(uint64_t base, uint32_t stride, int idx, uint32_t offset, uint64_t& value) {
    uint64_t addr = base + (uint64_t)idx * stride + offset;
    *((uint32_t*)&value)       = ess_Read32(addr);
    *(((uint32_t*)&value) + 1) = ess_Read32(addr + 4);
}

void XPl_IOstart() {
    if (s_PLIOStart) s_PLIOStart();
}

} // extern "C"
