/******************************************************************************
 * Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
#ifndef BSPCONFIG_H
#define BSPCONFIG_H

#include "xmem_config.h"
#define XPAR_XILTIMER_ENABLED
#include "xparameters_ps.h"

#if (defined(__aarch64__) || defined(ARMA53_32)) && !defined(FREERTOS_BSP)
#define EL3   1
#define EL1_NONSECURE  0
#define HYP_GUEST  0
#endif

#define versal
/* #undef VERSAL */
#define VERSAL_NET  
#define VERSAL_AIEPG2
#define VERSAL_2VE_2VM
/* #undef SPARTANUP */
/* #undef PSU_PMU */
/* #undef PLATFORM_ZYNQMP */
/* #undef XCLOCKING */
/* #undef PLATFORM_ZYNQ */
/* #undef VERSAL_PLM */
/* #undef VERSALNET_PLM */
/* #undef PLATFORM_MB */
/* #undef ASUFW */
#define XPAR_CPU_ID 0
#define XIL_INTERRUPT
/* #undef XPM_SUPPORT */
#define ENABLE_MINIMAL_XLAT_TBL
/* #undef ENABLE_4K_PAGES_OCM_TCM */
/* #undef XPAR_STDIN_IS_UARTLITE */
/* #undef XPAR_STDIN_IS_UARTNS550 */
/* #undef XPAR_STDIN_IS_UARTPS */
#define XPAR_STDIN_IS_UARTPSV
/* #undef XPAR_STDIN_IS_CORESIGHTPS_DCC */
/* #undef XPAR_STDIN_IS_IOMODULE */
#define STDIN_BASEADDRESS 0xf1920000
#define STDOUT_BASEADDRESS 0xf1920000

#endif /* BSPCONFIG_H */
