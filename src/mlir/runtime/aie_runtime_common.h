/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef AIE_RUNTIME_COMMON_H
#define AIE_RUNTIME_COMMON_H

#include "xaiengine.h"
#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif

/* --- Out-of-order DMA transfer modes --- */
#define MOVEDATA_MODE_BLOCKING 0   /* Blocking: wait for S2MM completion */
#define MOVEDATA_MODE_OOO 1        /* OOO: non-blocking, contiguous receive */
#define MOVEDATA_MODE_OOO_STRIDE 2 /* OOO: non-blocking, stride/wrap receive */

#define MOVEDATA_MAX_PENDING 16
#define MOVEDATA_MAX_SOURCES 16

typedef struct {
    XAie_LocType src_tile;       /* source tile location */
    uint32_t src_addr;           /* source memory offset */
    uint8_t src_ch;              /* source DMA channel */
    uint8_t src_bd;              /* source BD ID */
    uint8_t dst_bd;              /* destination BD ID on the shim */
    uint32_t data_bytes;         /* bytes this source sends */
    uint32_t dst_len;            /* dst BD length (0 = use data_bytes) */
    int src_pkt_id;              /* packet ID (>=0), or <0 for circuit-switched */
    int dst_num_dims;            /* 0 = linear, >0 = stride/wrap on dst BD */
    XAie_DmaDimDesc dst_dims[4]; /* stride/wrap per dimension for dst BD */
    /* [in] caller-provided destination receive address */
    XAie_MemInst *recv_buf; /* DDR MemInst (for sync) */
    u64 recv_phy;           /* DDR physical address for this dst BD */
} MovedataSrcDesc;

typedef struct {
    int mode;                /* MOVEDATA_MODE_* */
    int num_dims;            /* 0-4, used when mode == OOO_STRIDE */
    XAie_DmaDimDesc dims[4]; /* stride/wrap per dimension */
} MovedataOpt;

/**
 * Runtime_Movedata - Move data between tile DMAs (mem tile or shim tile).
 *
 * Configures MM2S on src_tile and S2MM on dst_tile, starts the transfer.
 * Behavior depends on opt->mode:
 *   MOVEDATA_MODE_BLOCKING   - waits for S2MM completion before returning.
 *   MOVEDATA_MODE_OOO        - non-blocking; receive BD uses contiguous addr.
 *   MOVEDATA_MODE_OOO_STRIDE - non-blocking; receive BD uses stride/wrap
 *                               addressing (opt->num_dims / opt->dims[]).
 *
 * For OOO modes, call Runtime_Movedata_WaitAll() to drain all pending
 * transfers before reading the data.
 *
 * For shim dst (row==0): allocates a DDR buffer, returns it via
 *   out_recv_buf/out_recv_phy so the caller can read back the data.
 * For mem dst (row>0): uses dst_addr as the local tile memory offset.
 *
 * @param opt       Transfer options (mode + optional stride/wrap). NULL for
 *                  blocking (backward-compatible).
 * @param pkt_id    Packet ID for packet-switched paths (>=0), or <0 for
 *                  circuit-switched (no packet header).
 * @param out_recv_buf  [out] DDR MemInst when dst is shim (may be NULL).
 * @param out_recv_phy  [out] DDR physical address when dst is shim (may be NULL).
 *
 * @return XAIE_OK on success, error code otherwise.
 */
AieRC Runtime_Movedata(XAie_DevInst *DevInst, XAie_LocType src_tile, uint32_t src_addr, uint8_t src_ch, uint8_t src_bd,
                       XAie_LocType dst_tile, uint32_t dst_addr, uint8_t dst_ch, uint8_t dst_bd, uint32_t data_bytes,
                       int pkt_id, XAie_MemInst **out_recv_buf, u64 *out_recv_phy, MovedataOpt *opt = nullptr);

/**
 * Runtime_Movedata_WaitAll - Wait for all pending OOO transfers to complete.
 *
 * Iterates through all pending entries and waits for each DMA channel to
 * finish. Entries may be S2MM (normal transfers) or MM2S (SingleDstBd
 * where the self-looping S2MM never idles — we wait on sources instead).
 * Resets the pending count to 0 after all waits succeed.
 *
 * @return XAIE_OK if all transfers completed, first error code otherwise.
 */
AieRC Runtime_Movedata_WaitAll(XAie_DevInst *DevInst);

/**
 * Runtime_Movedata_ManyToOne - Move data from multiple source tiles to a
 * single destination shim tile using out-of-order (OOO) BD selection.
 *
 * Uses hardware OOO DMA:
 *   - Only source MM2S BDs get XAie_DmaSetOutofOrderBdId (srcs[i].dst_bd
 *     as the OOO BD ID). Destination S2MM BDs do NOT need it.
 *   - The S2MM channel is configured with XAie_DmaChannelEnOutofOrder so
 *     it selects the correct receive BD based on the OOO BD ID embedded
 *     in incoming packets, rather than processing BDs from the queue.
 *   - The S2MM channel is started once with BD 0 pushed to the queue.
 *
 * The caller provides the destination DDR address per source via
 * srcs[i].recv_phy and srcs[i].recv_buf. Each destination BD is configured
 * with that address (with optional stride/wrap from dst_dims/dst_num_dims).
 *
 * After this call, use Runtime_Movedata_WaitAll() to wait for all transfers.
 *
 * @param srcs          Array of source descriptors. Each srcs[i].recv_phy
 *                      and srcs[i].recv_buf must be set by the caller.
 *                      srcs[i].dst_bd is used as the OOO BD ID on the
 *                      source side.
 * @param num_srcs      Number of sources (1..MOVEDATA_MAX_SOURCES).
 * @param dst_tile      Single destination shim tile (row==0).
 * @param dst_ch        Destination DMA channel (S2MM).
 *
 * @return XAIE_OK on success, error code otherwise.
 */
AieRC Runtime_Movedata_ManyToOne(XAie_DevInst *DevInst, MovedataSrcDesc *srcs, int num_srcs, XAie_LocType dst_tile,
                                 uint8_t dst_ch);

/**
 * Runtime_Movedata_ManyToOne_SingleDstBd - Move data from multiple source
 * tiles to a single destination shim tile using ONE destination BD with
 * out-of-order (OOO) BD selection.
 *
 * Uses hardware OOO DMA (same pattern as Runtime_Movedata_ManyToOne):
 *   - Source MM2S BDs get XAie_DmaSetOutofOrderBdId(dst_bd) so the
 *     destination S2MM channel knows which BD to write into.
 *   - The S2MM channel is configured with XAie_DmaChannelEnOutofOrder
 *     and started via XAie_DmaChannelSetStartQueue with repeat=num_srcs.
 *   - Iteration wrap = num_srcs with iter_step_size: the write address
 *     advances by iter_step_size words after each transaction.
 *   - Buffer length = per_src_bytes (per transaction, not total).
 *   - Optionally uses multi-dim addressing (dst_dims/dst_num_dims) for
 *     scatter-write patterns within each transaction.
 *
 * After this call, use Runtime_Movedata_WaitAll() to wait for completion.
 *
 * @param srcs           Array of source descriptors (src_tile, src_addr,
 *                       src_ch, src_bd, data_bytes, src_pkt_id used).
 * @param num_srcs       Number of sources (1..MOVEDATA_MAX_SOURCES).
 * @param dst_tile       Destination shim tile (row==0).
 * @param dst_ch         Destination DMA channel (S2MM).
 * @param dst_bd         Single destination BD ID (NextBd→self).
 * @param dst_addr       DDR physical base address for the destination BD.
 * @param per_src_bytes  Bytes per transaction (one source's data).
 * @param dst_num_dims   Number of multi-dim dimensions (0 = linear).
 * @param dst_dims       Multi-dim descriptors (may be NULL if dst_num_dims==0).
 * @param iter_step_size Iteration step size in 32-bit words for address
 *                       advancement between transactions. 0 (default) =
 *                       auto-compute as per_src_bytes/4 (contiguous).
 *
 * @return XAIE_OK on success, error code otherwise.
 */
AieRC Runtime_Movedata_ManyToOne_SingleDstBd(XAie_DevInst *DevInst, MovedataSrcDesc *srcs, int num_srcs,
                                             XAie_LocType dst_tile, uint8_t dst_ch, uint8_t dst_bd, u64 dst_addr,
                                             uint32_t per_src_bytes, int dst_num_dims = 0,
                                             XAie_DmaDimDesc *dst_dims = nullptr, int iter_step_size = 0);

#ifdef __cplusplus
}
#endif

#endif /* AIE_RUNTIME_COMMON_H */
