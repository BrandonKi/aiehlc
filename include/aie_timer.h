/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#ifndef AIE_TIMER_H
#define AIE_TIMER_H

#ifdef __AIESIM__
#include <stdint.h>
#include <time.h>
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000000ULL
static inline void XTime_GetTime(XTime *t) {
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    *t = (uint64_t)_ts.tv_sec * 1000000000ULL + (uint64_t)_ts.tv_nsec;
}
#elif defined(AIE_GEN) && AIE_GEN == 5
#include "xiltimer.h"
#else
#include "xtime_l.h"
#endif

/* [exp40] Independent PS wall counter, read-only (no system-register writes ⇒ cannot
 * trap/hang). XTime's xiltimer source is frozen on some boots (raw counts read 0);
 * CNTVCT_EL0 is the ARM architectural counter, readable at EL1+. If it advances while
 * XTime does not, it revives the end-to-end wall metric. CNTFRQ_EL0 gives its tick
 * frequency for ms conversion. */
#if defined(__aarch64__) && !defined(__AIESIM__)
static inline unsigned long long __ps_cntvct(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}
static inline unsigned long long __ps_cntfrq(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
    return v;
}
#else
static inline unsigned long long __ps_cntvct(void) { return 0ULL; }
static inline unsigned long long __ps_cntfrq(void) { return 0ULL; }
#endif

/* [exp41] PMU cycle counter (PMCCNTR_EL0) fallback to exp40's CNTVCT. When the system
 * generic timer is frozen this boot, CNTVCT_EL0 (same source) also reads 0. PMCCNTR_EL0
 * counts CPU-core-clock cycles from a *different* clock tree, so it survives a dead
 * generic timer. It needs a one-time enable sequence (system-register WRITES): if the
 * ELF runs at EL0 these msr writes trap → hang, so this is gated behind exp40 being
 * proven insufficient. Reports raw cycles (nonzero = alive); ms only with a known CPU Hz. */
#if defined(__aarch64__) && !defined(__AIESIM__)
static inline void __ps_pmccntr_enable(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmcr_el0" : "=r"(v));
    v |= (1ULL << 0) | (1ULL << 2); /* E: enable counters, C: reset cycle counter */
    __asm__ volatile("msr pmcr_el0, %0" : : "r"(v));
    __asm__ volatile("msr pmcntenset_el0, %0" : : "r"(1ULL << 31)); /* enable dedicated cycle counter */
    __asm__ volatile("isb");
}
static inline unsigned long long __ps_pmccntr(void) {
    unsigned long long v;
    __asm__ volatile("mrs %0, pmccntr_el0" : "=r"(v));
    return v;
}
#else
static inline void __ps_pmccntr_enable(void) {}
static inline unsigned long long __ps_pmccntr(void) { return 0ULL; }
#endif

#endif /* AIE_TIMER_H */
