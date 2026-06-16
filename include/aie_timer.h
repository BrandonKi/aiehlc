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

#endif /* AIE_TIMER_H */
