// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#ifndef AIEHLC_DBG_SERVER_H
#define AIEHLC_DBG_SERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned int (*read32)(uint64_t addr);
    void (*write32)(uint64_t addr, unsigned int data);
    unsigned int (*npi_read32)(uint64_t addr);
    void (*npi_write32)(uint64_t addr, unsigned int data);
} aiehlc_dbg_callbacks_t;

typedef struct {
    uint64_t base_address;
    uint32_t column_shift;
    uint32_t row_shift;
    int aie_gen;
} aiehlc_dbg_addr_info_t;

void aiehlc_dbg_set_callbacks(const aiehlc_dbg_callbacks_t *cb);

void aiehlc_dbg_set_wake(void (*wake)(void));

int aiehlc_dbg_start(const aiehlc_dbg_addr_info_t *addr);

int aiehlc_dbg_drain(void);

uint64_t aiehlc_dbg_service_count(void);

void aiehlc_dbg_stop(void);

int aiehlc_dbg_active(void);

#ifdef __cplusplus
}
#endif

#endif // AIEHLC_DBG_SERVER_H
