/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aie_runtime_common.h"

/* --- Pending OOO transfer tracker --- */
typedef struct {
    XAie_LocType tile;
    uint8_t ch;
    XAie_DmaDirection dir; /* DMA_S2MM or DMA_MM2S */
} MovedataPending;

static MovedataPending g_pending[MOVEDATA_MAX_PENDING];
static int g_pending_count = 0;

AieRC Runtime_Movedata(XAie_DevInst *DevInst, XAie_LocType src_tile, uint32_t src_addr, uint8_t src_ch, uint8_t src_bd,
                       XAie_LocType dst_tile, uint32_t dst_addr, uint8_t dst_ch, uint8_t dst_bd, uint32_t data_bytes,
                       int pkt_id, XAie_MemInst **out_recv_buf, u64 *out_recv_phy, MovedataOpt *opt) {
    AieRC RC = XAIE_OK;
    int dst_is_shim = (dst_tile.Row == 0);
    int src_is_shim = (src_tile.Row == 0);
    int mode = (opt != NULL) ? opt->mode : MOVEDATA_MODE_BLOCKING;

    printf("  Runtime_Movedata: (%d,%d) MM2S ch%d -> (%d,%d) S2MM ch%d, %u bytes, mode=%d", src_tile.Col, src_tile.Row,
           src_ch, dst_tile.Col, dst_tile.Row, dst_ch, data_bytes, mode);
    if (pkt_id >= 0)
        printf(", pkt_id=%d", pkt_id);
    printf("\n");

    /* ---- Configure destination (S2MM) ---- */
    {
        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, dst_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: dst DmaDescInit: %d\n", RC);
            return RC;
        }

        if (dst_is_shim) {
            /* Shim tile: allocate DDR buffer */
            XAie_MemInst *buf = XAie_MemAllocate(DevInst, data_bytes, XAIE_MEM_CACHEABLE);
            if (!buf) {
                printf("ERROR: DDR alloc for dst shim failed\n");
                return XAIE_ERR;
            }
            u64 phy = (u64)XAie_MemGetDevAddr(buf);
            /* Clear buffer */
            uint32_t i;
            for (i = 0; i < data_bytes / sizeof(uint32_t); i++)
                ((uint32_t *)phy)[i] = 0xDEADBEEF;
            XAie_MemSyncForDev(buf);

            if (mode == MOVEDATA_MODE_OOO_STRIDE && opt->num_dims > 0) {
                /* Stride/wrap addressing on receive BD */
                XAie_DmaTensor tensor;
                tensor.NumDim = opt->num_dims;
                tensor.Dim = opt->dims;
                RC = XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, phy, data_bytes);
            } else {
                RC = XAie_DmaSetAddrLen(&DmaInst, phy, data_bytes);
            }
            if (RC != XAIE_OK) {
                printf("ERROR: dst DmaSetAddr: %d\n", RC);
                return RC;
            }

            if (out_recv_buf)
                *out_recv_buf = buf;
            if (out_recv_phy)
                *out_recv_phy = phy;
        } else {
            /* Mem tile: use local memory address */
            if (mode == MOVEDATA_MODE_OOO_STRIDE && opt->num_dims > 0) {
                XAie_DmaTensor tensor;
                tensor.NumDim = opt->num_dims;
                tensor.Dim = opt->dims;
                RC = XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, dst_addr, data_bytes);
            } else {
                RC = XAie_DmaSetAddrLen(&DmaInst, dst_addr, data_bytes);
            }
            if (RC != XAIE_OK) {
                printf("ERROR: dst DmaSetAddr: %d\n", RC);
                return RC;
            }
        }

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, dst_tile, dst_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: dst DmaWriteBd: %d\n", RC);
            return RC;
        }

        RC = XAie_DmaChannelPushBdToQueue(DevInst, dst_tile, dst_ch, DMA_S2MM, dst_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: dst PushBdToQueue: %d\n", RC);
            return RC;
        }

        RC = XAie_DmaChannelEnable(DevInst, dst_tile, dst_ch, DMA_S2MM);
        if (RC != XAIE_OK) {
            printf("ERROR: dst DmaChannelEnable: %d\n", RC);
            return RC;
        }
    }

    /* ---- Configure source (MM2S) ---- */
    {
        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, src_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: src DmaDescInit: %d\n", RC);
            return RC;
        }

        if (src_is_shim) {
            RC = XAie_DmaSetAddrLen(&DmaInst, (u64)src_addr, data_bytes);
        } else {
            RC = XAie_DmaSetAddrLen(&DmaInst, src_addr, data_bytes);
        }
        if (RC != XAIE_OK) {
            printf("ERROR: src DmaSetAddrLen: %d\n", RC);
            return RC;
        }

        /* Attach packet header if needed */
        if (pkt_id >= 0) {
            XAie_Packet pkt = XAie_PacketInit(pkt_id, 0);
            RC = XAie_DmaSetPkt(&DmaInst, pkt);
            if (RC != XAIE_OK) {
                printf("ERROR: src DmaSetPkt: %d\n", RC);
                return RC;
            }
        }

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, src_tile, src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: src DmaWriteBd: %d\n", RC);
            return RC;
        }

        RC = XAie_DmaChannelPushBdToQueue(DevInst, src_tile, src_ch, DMA_MM2S, src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: src PushBdToQueue: %d\n", RC);
            return RC;
        }

        RC = XAie_DmaChannelEnable(DevInst, src_tile, src_ch, DMA_MM2S);
        if (RC != XAIE_OK) {
            printf("ERROR: src DmaChannelEnable: %d\n", RC);
            return RC;
        }
    }

    /* ---- Blocking mode: wait for destination S2MM to complete ---- */
    if (mode == MOVEDATA_MODE_BLOCKING) {
        RC = XAie_DmaWaitForDone(DevInst, dst_tile, dst_ch, DMA_S2MM, 5000000 /*5s*/);
        if (RC != XAIE_OK) {
            printf("ERROR: dst S2MM wait timed out: %d\n", RC);
            return RC;
        }
    } else {
        /* OOO mode: record pending transfer for later WaitAll */
        if (g_pending_count >= MOVEDATA_MAX_PENDING) {
            printf("ERROR: too many pending OOO transfers (max %d)\n", MOVEDATA_MAX_PENDING);
            return XAIE_ERR;
        }
        g_pending[g_pending_count].tile = dst_tile;
        g_pending[g_pending_count].ch = dst_ch;
        g_pending[g_pending_count].dir = DMA_S2MM;
        g_pending_count++;
    }

    return XAIE_OK;
}

AieRC Runtime_Movedata_WaitAll(XAie_DevInst *DevInst) {
    AieRC RC = XAIE_OK;
    int i;

    printf("  Runtime_Movedata_WaitAll: waiting for %d pending transfers...\n", g_pending_count);

    for (i = 0; i < g_pending_count; i++) {
        AieRC rc = XAie_DmaWaitForDone(DevInst, g_pending[i].tile, g_pending[i].ch, g_pending[i].dir, 5000000 /*5s*/);
        if (rc != XAIE_OK) {
            printf("ERROR: WaitAll: transfer %d (tile(%d,%d) ch%d %s) timed out: %d\n", i, g_pending[i].tile.Col,
                   g_pending[i].tile.Row, g_pending[i].ch, g_pending[i].dir == DMA_S2MM ? "S2MM" : "MM2S", rc);
            if (RC == XAIE_OK)
                RC = rc; /* report first error */
        }
    }

    g_pending_count = 0;
    printf("  Runtime_Movedata_WaitAll: done.\n");
    return RC;
}

AieRC Runtime_Movedata_ManyToOne(XAie_DevInst *DevInst, MovedataSrcDesc *srcs, int num_srcs, XAie_LocType dst_tile,
                                 uint8_t dst_ch) {
    AieRC RC = XAIE_OK;
    int i;

    if (num_srcs <= 0 || num_srcs > MOVEDATA_MAX_SOURCES) {
        printf("ERROR: ManyToOne: invalid num_srcs=%d (max %d)\n", num_srcs, MOVEDATA_MAX_SOURCES);
        return XAIE_ERR;
    }

    if (dst_tile.Row != 0) {
        printf("ERROR: ManyToOne: dst_tile (%d,%d) is not a shim tile\n", dst_tile.Col, dst_tile.Row);
        return XAIE_ERR;
    }

    if (g_pending_count + 1 > MOVEDATA_MAX_PENDING) {
        printf("ERROR: ManyToOne: would exceed max pending (%d + 1 > %d)\n", g_pending_count, MOVEDATA_MAX_PENDING);
        return XAIE_ERR;
    }

    printf("  0506 Runtime_Movedata_ManyToOne: %d sources → Shim(%d,%d) S2MM ch%d\n", num_srcs, dst_tile.Col,
           dst_tile.Row, dst_ch);

    /* ---- 1: Configure all destination BDs (S2MM on dst_tile) ---- */
    /* Destination BDs do NOT need XAie_DmaSetOutofOrderBdId.               */
    /* The OOO BD ID is only set on the source MM2S BDs; the S2MM channel   */
    /* uses the OOO BD ID from incoming packets to select which BD to use.  */

    /* -- Pass 1: Group sources by dst_bd, count per group, validate --     */
    /* bd_group[bd_id] = {count, dst_len, recv_phy, dst_num_dims,           */
    /*                    first_src_index}                                   */
    typedef struct {
        int count;
        uint32_t dst_len;
        u64 recv_phy;
        int dst_num_dims;
        int first_src_idx;
    } BdGroupInfo;
    std::unordered_map<int, BdGroupInfo> bd_groups;

    for (i = 0; i < num_srcs; i++) {
        MovedataSrcDesc *s = &srcs[i];
        uint32_t dst_len = (s->dst_len > 0) ? s->dst_len : s->data_bytes;

        auto it = bd_groups.find(s->dst_bd);
        if (it != bd_groups.end()) {
            BdGroupInfo &g = it->second;
            if (g.dst_len != dst_len) {
                printf("ERROR: ManyToOne[%d]: dst_len mismatch for bd%d: existing=%u new=%u\n", i, s->dst_bd, g.dst_len,
                       dst_len);
                return XAIE_ERR;
            }
            if (g.recv_phy != s->recv_phy) {
                printf("ERROR: ManyToOne[%d]: recv_phy mismatch for bd%d: existing=0x%llx new=0x%llx\n", i, s->dst_bd,
                       (unsigned long long)g.recv_phy, (unsigned long long)s->recv_phy);
                return XAIE_ERR;
            }
            g.count++;
        } else {
            BdGroupInfo g;
            g.count = 1;
            g.dst_len = dst_len;
            g.recv_phy = s->recv_phy;
            g.dst_num_dims = s->dst_num_dims;
            g.first_src_idx = i;
            bd_groups[s->dst_bd] = g;
        }
    }

    /* -- Pass 2: Write each unique dst_bd exactly once --                  */
    for (auto &kv : bd_groups) {
        int bd_id = kv.first;
        BdGroupInfo &g = kv.second;
        MovedataSrcDesc *repr = &srcs[g.first_src_idx];

        printf("    [dst bd%d] addr=0x%llx len=%u, srcs_sharing=%d", bd_id, (unsigned long long)g.recv_phy, g.dst_len,
               g.count);
        if (g.dst_num_dims > 0)
            printf(", dims=%d", g.dst_num_dims);
        printf("\n");

        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, dst_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne: dst DmaDescInit for bd%d: %d\n", bd_id, RC);
            return RC;
        }

        if (g.dst_num_dims > 0) {
            XAie_DmaTensor tensor;
            tensor.NumDim = g.dst_num_dims;
            tensor.Dim = repr->dst_dims;
            RC = XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, g.recv_phy, g.dst_len);
        } else {
            RC = XAie_DmaSetAddrLen(&DmaInst, g.recv_phy, g.dst_len);
        }
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne: dst DmaSetAddr for bd%d: %d (addr=0x%llx, len=%u, num_dims=%d)\n", bd_id, RC,
                   (unsigned long long)g.recv_phy, g.dst_len, g.dst_num_dims);
            return RC;
        }

        /* Iteration: StepSize in 32-bit words (byte→word: /4), Wrap = group count */
        if (g.count > 1) {
            RC = XAie_DmaSetBdIteration(&DmaInst, g.dst_len / 4, g.count, 0);
            if (RC != XAIE_OK) {
                printf("ERROR: ManyToOne: dst DmaSetBdIteration for bd%d: %d\n", bd_id, RC);
                return RC;
            }
        }
        /* Single-source groups don't need iteration (default: no D3 wrap) */

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, dst_tile, bd_id);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne: dst DmaWriteBd for bd%d: %d\n", bd_id, RC);
            return RC;
        }
    }

    /* ---- 2: Enable OOO on S2MM channel, push BD 0, start once ---- */
    /* With OOO enabled the channel selects BDs based on the OOO BD ID     */
    /* in incoming packets, not from the sequential queue. We push BD 0     */
    /* to satisfy the channel start requirement.                            */
    {
        XAie_DmaChannelDesc DmaChannelDescInst;
        XAie_DmaChannelDescInit(DevInst, &DmaChannelDescInst, dst_tile);
        XAie_DmaChannelEnOutofOrder(&DmaChannelDescInst, XAIE_ENABLE);
        RC = XAie_DmaWriteChannel(DevInst, &DmaChannelDescInst, dst_tile, dst_ch, DMA_S2MM);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne: S2MM OOO channel enable: %d\n", RC);
            return RC;
        }
        printf("    S2MM ch%d OOO enabled on Shim(%d,%d)\n", dst_ch, dst_tile.Col, dst_tile.Row);
    }
    // as OOO drived by src bd list, the detination need repeat N time the n is the source bd transcatin number
    //  by using this way each first arrive BD can be processed, and each process will remove the packet header
    //  and process per packet length
    RC = XAie_DmaChannelSetStartQueue(DevInst, dst_tile, dst_ch /*channel*/, DMA_S2MM, 0 /*bdid*/, num_srcs,
                                      XAIE_DISABLE);
    // RC = XAie_DmaChannelPushBdToQueue(DevInst, dst_tile, dst_ch, DMA_S2MM, 0 /*bd 0*/);
    if (RC != XAIE_OK) {
        printf("ERROR: ManyToOne: dst XAie_DmaChannelSetStartQueue: %d\n", RC);
        return RC;
    }

    // RC = XAie_DmaChannelEnable(DevInst, dst_tile, dst_ch, DMA_S2MM);
    // if (RC != XAIE_OK) {
    //     printf("ERROR: ManyToOne: dst DmaChannelEnable: %d\n", RC);
    //     return RC;
    /// }

    /* ---- 3: Configure each source BD (MM2S) with OOO BD ID ---- */
    /* Only source BDs need XAie_DmaSetOutofOrderBdId. The OOO BD ID       */
    /* tells the destination S2MM channel which BD to write into.           */
    for (i = 0; i < num_srcs; i++) {
        MovedataSrcDesc *s = &srcs[i];

        printf("    [src %d] tile(%d,%d) MM2S ch%d bd%d, %u bytes", i, s->src_tile.Col, s->src_tile.Row, s->src_ch,
               s->src_bd, s->data_bytes);
        if (s->src_pkt_id >= 0)
            printf(", pkt_id=%d", s->src_pkt_id);
        printf(", ooo_bd_id=%d\n", s->dst_bd);

        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, s->src_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src DmaDescInit: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaSetAddrLen(&DmaInst, s->src_addr, s->data_bytes);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src DmaSetAddrLen: %d\n", i, RC);
            return RC;
        }

        if (s->src_pkt_id >= 0) {
            XAie_Packet pkt = XAie_PacketInit(s->src_pkt_id, 0);
            RC = XAie_DmaSetPkt(&DmaInst, pkt);
            if (RC != XAIE_OK) {
                printf("ERROR: ManyToOne[%d]: src DmaSetPkt: %d\n", i, RC);
                return RC;
            }
        }

        /* OOO BD ID on source — tells the destination S2MM which BD to use */
        RC = XAie_DmaSetOutofOrderBdId(&DmaInst, s->dst_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src DmaSetOutofOrderBdId: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, s->src_tile, s->src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src DmaWriteBd: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaChannelPushBdToQueue(DevInst, s->src_tile, s->src_ch, DMA_MM2S, s->src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src PushBdToQueue: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaChannelEnable(DevInst, s->src_tile, s->src_ch, DMA_MM2S);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne[%d]: src DmaChannelEnable: %d\n", i, RC);
            return RC;
        }
    }

    /* ---- 4: Record single S2MM pending entry for WaitAll ---- */
    g_pending[g_pending_count].tile = dst_tile;
    g_pending[g_pending_count].ch = dst_ch;
    g_pending[g_pending_count].dir = DMA_S2MM;
    g_pending_count++;

    return XAIE_OK;
}

AieRC Runtime_Movedata_ManyToOne_SingleDstBd(XAie_DevInst *DevInst, MovedataSrcDesc *srcs, int num_srcs,
                                             XAie_LocType dst_tile, uint8_t dst_ch, uint8_t dst_bd, u64 dst_addr,
                                             uint32_t per_src_bytes, int dst_num_dims, XAie_DmaDimDesc *dst_dims,
                                             int iter_step_size) {
    AieRC RC = XAIE_OK;
    int i;

    if (num_srcs <= 0 || num_srcs > MOVEDATA_MAX_SOURCES) {
        printf("ERROR: ManyToOne_SingleDstBd: invalid num_srcs=%d (max %d)\n", num_srcs, MOVEDATA_MAX_SOURCES);
        return XAIE_ERR;
    }

    if (dst_tile.Row != 0) {
        printf("ERROR: ManyToOne_SingleDstBd: dst_tile (%d,%d) is not a shim tile\n", dst_tile.Col, dst_tile.Row);
        return XAIE_ERR;
    }

    if (g_pending_count + 1 > MOVEDATA_MAX_PENDING) {
        printf("ERROR: ManyToOne_SingleDstBd: would exceed max pending (%d + 1 > %d)\n", g_pending_count,
               MOVEDATA_MAX_PENDING);
        return XAIE_ERR;
    }

    /* Default iter_step_size: advance by per_src_bytes/4 words (contiguous) */
    if (iter_step_size <= 0)
        iter_step_size = per_src_bytes / 4;

    printf("  Runtime_Movedata_ManyToOne_SingleDstBd: %d sources → Shim(%d,%d) S2MM ch%d bd%d\n", num_srcs,
           dst_tile.Col, dst_tile.Row, dst_ch, dst_bd);
    printf("    dst_addr=0x%llx per_src_bytes=%u iter_step=%d iter_wrap=%d\n", (unsigned long long)dst_addr,
           per_src_bytes, iter_step_size, num_srcs);

    /* ---- 1: Configure the single destination BD (S2MM on dst_tile) ---- */
    /*    Uses OOO logic: the S2MM channel selects this BD based on OOO    */
    /*    BD ID from incoming packets. Iteration wrap = num_srcs with      */
    /*    iter_step_size for address advancement between transactions.     */
    {
        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, dst_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd: dst DmaDescInit: %d\n", RC);
            return RC;
        }

        if (dst_num_dims > 0 && dst_dims != NULL) {
            XAie_DmaTensor tensor;
            tensor.NumDim = dst_num_dims;
            tensor.Dim = dst_dims;
            RC = XAie_DmaSetMultiDimAddr(&DmaInst, &tensor, dst_addr, per_src_bytes);
        } else {
            RC = XAie_DmaSetAddrLen(&DmaInst, dst_addr, per_src_bytes);
        }
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd: dst DmaSetAddr: %d\n", RC);
            return RC;
        }

        /* Iteration: address advances iter_step_size words per transaction */
        RC = XAie_DmaSetBdIteration(&DmaInst, iter_step_size, num_srcs, 0);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd: dst DmaSetBdIteration: %d\n", RC);
            return RC;
        }

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, dst_tile, dst_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd: dst DmaWriteBd: %d\n", RC);
            return RC;
        }
    }

    /* ---- 2: Enable OOO on S2MM channel, use SetStartQueue ---- */
    /* With OOO enabled the channel selects BDs based on the OOO BD ID     */
    /* in incoming packets. We use SetStartQueue with repeat=num_srcs      */
    /* so the single BD is re-used for each source transaction.            */
    {
        XAie_DmaChannelDesc DmaChannelDescInst;
        XAie_DmaChannelDescInit(DevInst, &DmaChannelDescInst, dst_tile);
        XAie_DmaChannelEnOutofOrder(&DmaChannelDescInst, XAIE_ENABLE);
        RC = XAie_DmaWriteChannel(DevInst, &DmaChannelDescInst, dst_tile, dst_ch, DMA_S2MM);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd: S2MM OOO channel enable: %d\n", RC);
            return RC;
        }
        printf("    S2MM ch%d OOO enabled on Shim(%d,%d)\n", dst_ch, dst_tile.Col, dst_tile.Row);
    }
    RC = XAie_DmaChannelSetStartQueue(DevInst, dst_tile, dst_ch, DMA_S2MM, dst_bd, num_srcs /*1*/, XAIE_DISABLE);
    if (RC != XAIE_OK) {
        printf("ERROR: ManyToOne_SingleDstBd: dst XAie_DmaChannelSetStartQueue: %d\n", RC);
        return RC;
    }

    /* ---- 3: Configure each source BD (MM2S) with OOO BD ID ---- */
    /* Only source BDs need XAie_DmaSetOutofOrderBdId. The OOO BD ID       */
    /* tells the destination S2MM channel which BD to write into.           */
    /* Since SingleDstBd uses a single dst BD, all sources use dst_bd.      */
    for (i = 0; i < num_srcs; i++) {
        MovedataSrcDesc *s = &srcs[i];

        printf("    [src %d] tile(%d,%d) MM2S ch%d bd%d, %u bytes", i, s->src_tile.Col, s->src_tile.Row, s->src_ch,
               s->src_bd, s->data_bytes);
        if (s->src_pkt_id >= 0)
            printf(", pkt_id=%d", s->src_pkt_id);
        printf(", ooo_bd_id=%d\n", dst_bd);

        XAie_DmaDesc DmaInst;
        RC = XAie_DmaDescInit(DevInst, &DmaInst, s->src_tile);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaDescInit: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaSetAddrLen(&DmaInst, s->src_addr, s->data_bytes);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaSetAddrLen: %d\n", i, RC);
            return RC;
        }

        if (s->src_pkt_id >= 0) {
            XAie_Packet pkt = XAie_PacketInit(s->src_pkt_id, 0);
            RC = XAie_DmaSetPkt(&DmaInst, pkt);
            if (RC != XAIE_OK) {
                printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaSetPkt: %d\n", i, RC);
                return RC;
            }
        }

        /* OOO BD ID on source — tells the destination S2MM which BD to use */
        RC = XAie_DmaSetOutofOrderBdId(&DmaInst, dst_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaSetOutofOrderBdId: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaEnableBd(&DmaInst);
        RC = XAie_DmaWriteBd(DevInst, &DmaInst, s->src_tile, s->src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaWriteBd: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaChannelPushBdToQueue(DevInst, s->src_tile, s->src_ch, DMA_MM2S, s->src_bd);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src PushBdToQueue: %d\n", i, RC);
            return RC;
        }

        RC = XAie_DmaChannelEnable(DevInst, s->src_tile, s->src_ch, DMA_MM2S);
        if (RC != XAIE_OK) {
            printf("ERROR: ManyToOne_SingleDstBd[%d]: src DmaChannelEnable: %d\n", i, RC);
            return RC;
        }
    }

    /* ---- 4: Record single S2MM pending entry for WaitAll ---- */
    g_pending[g_pending_count].tile = dst_tile;
    g_pending[g_pending_count].ch = dst_ch;
    g_pending[g_pending_count].dir = DMA_S2MM;
    g_pending_count++;

    return XAIE_OK;
}
