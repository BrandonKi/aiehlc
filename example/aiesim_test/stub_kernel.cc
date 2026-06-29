// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT
//
// Stub kernel for AIE simulator tile activation.
//
// The aiesimulator only creates a core ISS execution model for tiles that
// have an ELF pre-loaded in Work/aie/<row>_<col>/Release/ at startup.
// Populating ALL core tile directories with this stub activates every core,
// allowing runtime XAie_LoadElfMem calls to target any tile freely.
//
// The stub does nothing — it just returns.  The actual kernel ELF is
// loaded over it at runtime via XAie_LoadElfMem.
//
// Uses the same stub_kernel.bcf (stack at 0x70000) as other kernels so
// the compiler places the stack away from the data region.

int main(void) { return 0; }
