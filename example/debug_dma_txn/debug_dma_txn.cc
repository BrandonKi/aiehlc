/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
/*
 * DMA transaction-count debug example (AIE-ML / gen 2).
 *
 * Exercises XAie_DmaTxnCount{Enable,Disable,Get} and
 * XAie_Partition{Initialize,Teardown}_v2:
 *   1. Group of tiles + DMA channel group
 *   2. Enable / disable the transaction counters
 *   3. Partition init v2: DmaTxnCountChGroup on (arm) and 0 (leave off)
 *
 * AIE tile MEM-mod has 2 perf counters and the driver binds one counter per
 * channel (FINISHED_BD), so up to 2 channels can be monitored per tile. These
 * tests monitor MM2S ch 0.
 *
 * Build (gen 2 sim):
 *   source script/aiehlc.sh --aie-version 2 --platform sim \
 *       --runtime-source-file ./example/debug_dma_txn/debug_dma_txn.cc
 *   bash script/runsim.sh aout
 */

#include "xaiengine.h"
#include <stdio.h>

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#define XAIE_BASE_ADDR 0x20000000000ULL
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#define XAIE_BASE_ADDR 0x20000000000ULL
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4
#endif

#define TILE_COL 4U
#define TILE_ROW0 XAIE_AIE_TILE_ROW_START
#define TILE_ROW1 (XAIE_AIE_TILE_ROW_START + 1U)

#define DATA_MEM_IN 0x4000U
#define DATA_MEM_OUT 0x3000U
#define NUM_ELEMS 32U
#define NUM_BYTES (NUM_ELEMS * (u32)sizeof(uint32_t))
/*
 * Poll budgets are deliberately small: a 128-byte BD retires in tens of cycles,
 * so anything beyond this is a real failure. Every poll is an MMIO read, which
 * costs milliseconds under the SystemC simulator, so a large budget turns a
 * failed test into an apparent hang instead of a fast diagnostic.
 */
#define PENDING_RETRY 200U
#define COUNT_POLL_RETRY 200U
#define LOCK_FOR_WRITE 0
#define LOCK_FOR_READ 1
#define CH_GROUP XAIE_DMA_CH_MM2S_0

/* Dummy kernel so aiehlc --platform sim extracts and host-compiles this file. */
__global__ void dummy_kernel(input_window_int32 *in
                             __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:16"))),
                             output_window_int32 *out
                             __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:16")))) {}

/*
 * Every step is logged and flushed: on a board hang the last line printed is
 * the call that stalled, which is the only way to localize an AXI stall.
 */
#define STEP(tag)                                                                                  \
	do {                                                                                       \
		printf("[debug_dma_txn]   . %s\n", (tag));                                         \
		fflush(stdout);                                                                    \
	} while (0)

#define TRY_RC(tag, call)                                                                          \
	do {                                                                                       \
		AieRC _rc = (call);                                                                \
		if (_rc != XAIE_OK) {                                                              \
			printf("[debug_dma_txn] FAIL: %s (RC=0x%x)\n", (tag), _rc);                \
			fflush(stdout);                                                            \
			return _rc;                                                                \
		}                                                                                  \
	} while (0)

static int fail(const char *msg, AieRC rc) {
	printf("[debug_dma_txn] FAIL: %s (RC=0x%x)\n", msg, rc);
	fflush(stdout);
	return -1;
}

/*
 * Read back what the counter is actually bound to, so a zero count can be told
 * apart from a counter that was never armed. Counter 0 is the slot the driver
 * assigns to the lowest set bit of the channel group (MM2S ch 0 here).
 */
/* Returns 1 if cnt0 is bound to MM2S0 FINISHED_BD, 0 if not, -1 on read error. */
static int probe_counter_binding(XAie_DevInst *Dev, XAie_LocType Loc,
				  const char *tag) {
	XAie_Events Start = XAIE_EVENT_NONE_CORE;
	XAie_Events Stop = XAIE_EVENT_NONE_CORE;
	XAie_Events Reset = XAIE_EVENT_NONE_CORE;
	u32 Val = 0U;
	AieRC RC;
	int Armed;

	RC = XAie_PerfCounterGetControlConfig(Dev, Loc, XAIE_MEM_MOD, 0U,
					      &Start, &Stop, &Reset);
	if (RC != XAIE_OK) {
		printf("[probe] %s tile (%u,%u) GetControlConfig RC=0x%x\n",
		       tag, Loc.Col, Loc.Row, RC);
		fflush(stdout);
		return -1;
	}
	(void)XAie_PerfCounterGet(Dev, Loc, XAIE_MEM_MOD, 0U, &Val);
	Armed = (Start == XAIE_EVENT_DMA_MM2S_0_FINISHED_BD_MEM &&
		 Stop == XAIE_EVENT_DMA_MM2S_0_FINISHED_BD_MEM) ? 1 : 0;
	printf("[probe] %s tile (%u,%u) cnt0 start=%d stop=%d reset=%d value=%u"
	       " (FINISHED_BD=%d armed=%d)\n",
	       tag, Loc.Col, Loc.Row, (int)Start, (int)Stop, (int)Reset, Val,
	       (int)XAIE_EVENT_DMA_MM2S_0_FINISHED_BD_MEM, Armed);
	fflush(stdout);
	return Armed;
}

static int print_mm2s_counts(XAie_DevInst *Dev, XAie_LocType Loc, u32 ChGroup,
			     u32 Target, u32 *EndOut) {
	u32 End = 0U;
	u32 retry;

	for (retry = 0U; retry < COUNT_POLL_RETRY; retry++) {
		AieRC RC = XAie_DmaTxnCountGet(Dev, Loc, ChGroup,
					       XAIE_DMA_CH_MM2S_0, &End);
		if (RC != XAIE_OK) {
			return fail("XAie_DmaTxnCountGet", RC);
		}
		if (End >= Target) {
			break;
		}
	}
	printf("[debug_dma_txn] tile (%u,%u) MM2S ch0 completed BDs=%u\n",
	       Loc.Col, Loc.Row, End);
	fflush(stdout);
	if (EndOut != NULL) {
		*EndOut = End;
	}
	return 0;
}

static AieRC setup_loopback_stream(XAie_DevInst *Dev, XAie_LocType T0,
				   XAie_LocType T1) {
	TRY_RC("StrmConn T0 DMA->NORTH", XAie_StrmConnCctEnable(Dev, T0, DMA, 0, NORTH, 0));
	TRY_RC("StrmConn T1 SOUTH->DMA", XAie_StrmConnCctEnable(Dev, T1, SOUTH, 0, DMA, 0));
	TRY_RC("StrmConn T1 DMA->SOUTH", XAie_StrmConnCctEnable(Dev, T1, DMA, 0, SOUTH, 0));
	TRY_RC("StrmConn T0 NORTH->DMA", XAie_StrmConnCctEnable(Dev, T0, NORTH, 0, DMA, 0));
	return XAIE_OK;
}

/*
 * Write handshake locks before every loopback. After column reset,
 * LockGetValue can already read 0 while acquire still stalls; a prior test
 * can also leave a non-zero value. XAie_LockSetValue makes the lock usable.
 */
static AieRC reset_loopback_locks(XAie_DevInst *Dev, XAie_LocType T0,
				  XAie_LocType T1) {
	u32 V0 = 0U, V1 = 0U;

	(void)XAie_LockGetValue(Dev, T0, XAie_LockInit(6U, 0), &V0);
	(void)XAie_LockGetValue(Dev, T1, XAie_LockInit(5U, 0), &V1);
	printf("[debug_dma_txn] lock pre-reset: T0 lock6=%u T1 lock5=%u\n",
	       V0, V1);
	fflush(stdout);

	TRY_RC("LockSetValue T0 lock6", XAie_LockSetValue(Dev, T0, XAie_LockInit(6U, 0)));
	TRY_RC("LockSetValue T1 lock5", XAie_LockSetValue(Dev, T1, XAie_LockInit(5U, 0)));
	return XAIE_OK;
}

static AieRC start_oneshot_loopback(XAie_DevInst *Dev, XAie_LocType T0,
				    XAie_LocType T1) {
	XAie_DmaDesc D0_MM2S, D0_S2MM, D1_MM2S, D1_S2MM;
	uint32_t data[NUM_ELEMS];
	u8 i;

	for (i = 0U; i < NUM_ELEMS; i++) {
		data[i] = 0xA0000000U + i;
	}

	TRY_RC("reset_loopback_locks", reset_loopback_locks(Dev, T0, T1));

	STEP("DataMemBlockWrite");
	TRY_RC("XAie_DataMemBlockWrite",
	       XAie_DataMemBlockWrite(Dev, T0, DATA_MEM_IN, data, NUM_BYTES));

	STEP("stream switch");
	TRY_RC("setup_loopback_stream", setup_loopback_stream(Dev, T0, T1));

	STEP("DmaDescInit");
	TRY_RC("DmaDescInit D0_MM2S", XAie_DmaDescInit(Dev, &D0_MM2S, T0));
	TRY_RC("DmaDescInit D0_S2MM", XAie_DmaDescInit(Dev, &D0_S2MM, T0));
	TRY_RC("DmaDescInit D1_MM2S", XAie_DmaDescInit(Dev, &D1_MM2S, T1));
	TRY_RC("DmaDescInit D1_S2MM", XAie_DmaDescInit(Dev, &D1_S2MM, T1));

	STEP("DmaSetAddrLen / SetLock / EnableBd");
	TRY_RC("SetAddrLen D0_MM2S", XAie_DmaSetAddrLen(&D0_MM2S, DATA_MEM_IN, NUM_BYTES));
	TRY_RC("SetAddrLen D0_S2MM", XAie_DmaSetAddrLen(&D0_S2MM, DATA_MEM_OUT, NUM_BYTES));
	TRY_RC("SetAddrLen D1_MM2S", XAie_DmaSetAddrLen(&D1_MM2S, DATA_MEM_IN, NUM_BYTES));
	TRY_RC("SetAddrLen D1_S2MM", XAie_DmaSetAddrLen(&D1_S2MM, DATA_MEM_IN, NUM_BYTES));

	TRY_RC("SetLock D1_S2MM", XAie_DmaSetLock(&D1_S2MM, XAie_LockInit(5U, LOCK_FOR_WRITE),
						  XAie_LockInit(5U, LOCK_FOR_READ)));
	TRY_RC("SetLock D1_MM2S", XAie_DmaSetLock(&D1_MM2S, XAie_LockInit(5U, LOCK_FOR_READ),
						  XAie_LockInit(5U, LOCK_FOR_WRITE)));
	TRY_RC("SetLock D0_MM2S", XAie_DmaSetLock(&D0_MM2S, XAie_LockInit(6U, LOCK_FOR_WRITE),
						  XAie_LockInit(6U, LOCK_FOR_READ)));
	TRY_RC("SetLock D0_S2MM", XAie_DmaSetLock(&D0_S2MM, XAie_LockInit(6U, LOCK_FOR_READ),
						  XAie_LockInit(6U, LOCK_FOR_WRITE)));

	TRY_RC("EnableBd D0_MM2S", XAie_DmaEnableBd(&D0_MM2S));
	TRY_RC("EnableBd D0_S2MM", XAie_DmaEnableBd(&D0_S2MM));
	TRY_RC("EnableBd D1_MM2S", XAie_DmaEnableBd(&D1_MM2S));
	TRY_RC("EnableBd D1_S2MM", XAie_DmaEnableBd(&D1_S2MM));

	STEP("DmaWriteBd");
	TRY_RC("WriteBd D0_MM2S", XAie_DmaWriteBd(Dev, &D0_MM2S, T0, 1U));
	TRY_RC("WriteBd D0_S2MM", XAie_DmaWriteBd(Dev, &D0_S2MM, T0, 9U));
	TRY_RC("WriteBd D1_MM2S", XAie_DmaWriteBd(Dev, &D1_MM2S, T1, 1U));
	TRY_RC("WriteBd D1_S2MM", XAie_DmaWriteBd(Dev, &D1_S2MM, T1, 9U));

	STEP("PushBdToQueue");
	TRY_RC("Push T0 MM2S", XAie_DmaChannelPushBdToQueue(Dev, T0, 0U, DMA_MM2S, 1U));
	TRY_RC("Push T0 S2MM", XAie_DmaChannelPushBdToQueue(Dev, T0, 0U, DMA_S2MM, 9U));
	TRY_RC("Push T1 MM2S", XAie_DmaChannelPushBdToQueue(Dev, T1, 0U, DMA_MM2S, 1U));
	TRY_RC("Push T1 S2MM", XAie_DmaChannelPushBdToQueue(Dev, T1, 0U, DMA_S2MM, 9U));

	STEP("DmaChannelEnable");
	TRY_RC("ChEnable T0 MM2S", XAie_DmaChannelEnable(Dev, T0, 0U, DMA_MM2S));
	TRY_RC("ChEnable T0 S2MM", XAie_DmaChannelEnable(Dev, T0, 0U, DMA_S2MM));
	TRY_RC("ChEnable T1 MM2S", XAie_DmaChannelEnable(Dev, T1, 0U, DMA_MM2S));
	TRY_RC("ChEnable T1 S2MM", XAie_DmaChannelEnable(Dev, T1, 0U, DMA_S2MM));
	return XAIE_OK;
}

/*
 * Self-linked BDs with no locks: each channel free-runs its single BD, so
 * FINISHED_BD keeps ticking after one task push. This is the Next-BD case
 * where throughput is otherwise invisible.
 */
static AieRC start_circular_loopback(XAie_DevInst *Dev, XAie_LocType T0,
				     XAie_LocType T1) {
	XAie_DmaDesc D0_MM2S, D0_S2MM, D1_MM2S, D1_S2MM;
	uint32_t data[NUM_ELEMS];
	u8 i;

	for (i = 0U; i < NUM_ELEMS; i++) {
		data[i] = 0xB0000000U + i;
	}

	TRY_RC("reset_loopback_locks", reset_loopback_locks(Dev, T0, T1));

	STEP("DataMemBlockWrite");
	TRY_RC("XAie_DataMemBlockWrite",
	       XAie_DataMemBlockWrite(Dev, T0, DATA_MEM_IN, data, NUM_BYTES));

	STEP("stream switch");
	TRY_RC("setup_loopback_stream", setup_loopback_stream(Dev, T0, T1));

	STEP("DmaDescInit");
	TRY_RC("DmaDescInit D0_MM2S", XAie_DmaDescInit(Dev, &D0_MM2S, T0));
	TRY_RC("DmaDescInit D0_S2MM", XAie_DmaDescInit(Dev, &D0_S2MM, T0));
	TRY_RC("DmaDescInit D1_MM2S", XAie_DmaDescInit(Dev, &D1_MM2S, T1));
	TRY_RC("DmaDescInit D1_S2MM", XAie_DmaDescInit(Dev, &D1_S2MM, T1));

	STEP("DmaSetAddrLen / SetNextBd / EnableBd");
	TRY_RC("SetAddrLen D0_MM2S", XAie_DmaSetAddrLen(&D0_MM2S, DATA_MEM_IN, NUM_BYTES));
	TRY_RC("SetAddrLen D0_S2MM", XAie_DmaSetAddrLen(&D0_S2MM, DATA_MEM_OUT, NUM_BYTES));
	TRY_RC("SetAddrLen D1_MM2S", XAie_DmaSetAddrLen(&D1_MM2S, DATA_MEM_IN, NUM_BYTES));
	TRY_RC("SetAddrLen D1_S2MM", XAie_DmaSetAddrLen(&D1_S2MM, DATA_MEM_IN, NUM_BYTES));

	TRY_RC("SetNextBd D0_MM2S", XAie_DmaSetNextBd(&D0_MM2S, 1U, XAIE_ENABLE));
	TRY_RC("SetNextBd D0_S2MM", XAie_DmaSetNextBd(&D0_S2MM, 9U, XAIE_ENABLE));
	TRY_RC("SetNextBd D1_MM2S", XAie_DmaSetNextBd(&D1_MM2S, 1U, XAIE_ENABLE));
	TRY_RC("SetNextBd D1_S2MM", XAie_DmaSetNextBd(&D1_S2MM, 9U, XAIE_ENABLE));

	TRY_RC("EnableBd D0_MM2S", XAie_DmaEnableBd(&D0_MM2S));
	TRY_RC("EnableBd D0_S2MM", XAie_DmaEnableBd(&D0_S2MM));
	TRY_RC("EnableBd D1_MM2S", XAie_DmaEnableBd(&D1_MM2S));
	TRY_RC("EnableBd D1_S2MM", XAie_DmaEnableBd(&D1_S2MM));

	STEP("DmaWriteBd");
	TRY_RC("WriteBd D0_MM2S", XAie_DmaWriteBd(Dev, &D0_MM2S, T0, 1U));
	TRY_RC("WriteBd D0_S2MM", XAie_DmaWriteBd(Dev, &D0_S2MM, T0, 9U));
	TRY_RC("WriteBd D1_MM2S", XAie_DmaWriteBd(Dev, &D1_MM2S, T1, 1U));
	TRY_RC("WriteBd D1_S2MM", XAie_DmaWriteBd(Dev, &D1_S2MM, T1, 9U));

	STEP("PushBdToQueue");
	TRY_RC("Push T0 MM2S", XAie_DmaChannelPushBdToQueue(Dev, T0, 0U, DMA_MM2S, 1U));
	TRY_RC("Push T0 S2MM", XAie_DmaChannelPushBdToQueue(Dev, T0, 0U, DMA_S2MM, 9U));
	TRY_RC("Push T1 MM2S", XAie_DmaChannelPushBdToQueue(Dev, T1, 0U, DMA_MM2S, 1U));
	TRY_RC("Push T1 S2MM", XAie_DmaChannelPushBdToQueue(Dev, T1, 0U, DMA_S2MM, 9U));

	STEP("DmaChannelEnable");
	TRY_RC("ChEnable T0 MM2S", XAie_DmaChannelEnable(Dev, T0, 0U, DMA_MM2S));
	TRY_RC("ChEnable T0 S2MM", XAie_DmaChannelEnable(Dev, T0, 0U, DMA_S2MM));
	TRY_RC("ChEnable T1 MM2S", XAie_DmaChannelEnable(Dev, T1, 0U, DMA_MM2S));
	TRY_RC("ChEnable T1 S2MM", XAie_DmaChannelEnable(Dev, T1, 0U, DMA_S2MM));
	return XAIE_OK;
}

static void wait_s2mm_drain(XAie_DevInst *Dev, XAie_LocType Tile) {
	u8 Pending = 1U;
	u32 retry;

	for (retry = 0U; Pending != 0U && retry < PENDING_RETRY; retry++) {
		(void)XAie_DmaGetPendingBdCount(Dev, Tile, 0U, DMA_S2MM, &Pending);
	}
	if (Pending != 0U) {
		printf("[debug_dma_txn] S2MM still pending on (%u,%u); reading counters anyway\n",
		       Tile.Col, Tile.Row);
	}
}

static void stop_dma_channels(XAie_DevInst *Dev, XAie_LocType T0, XAie_LocType T1) {
	(void)XAie_DmaChannelDisable(Dev, T0, 0U, DMA_MM2S);
	(void)XAie_DmaChannelDisable(Dev, T0, 0U, DMA_S2MM);
	(void)XAie_DmaChannelDisable(Dev, T1, 0U, DMA_MM2S);
	(void)XAie_DmaChannelDisable(Dev, T1, 0U, DMA_S2MM);
}

/* Use case #1+#2: rectangle of tiles, enable MM2S ch0, one-shot loopback, disable. */
static int test_enable_disable(XAie_DevInst *Dev) {
	AieRC RC;
	XAie_LocType T0 = XAie_TileLoc(TILE_COL, TILE_ROW0);
	XAie_LocType T1 = XAie_TileLoc(TILE_COL, TILE_ROW1);
	XAie_SubPartition Sub = {{TILE_COL, 1U}, {TILE_ROW0, 2U}};
	u32 End0 = 0U, End1 = 0U;

	printf("[debug_dma_txn] --- test_enable_disable (group 1x2, MM2S ch0) ---\n");
	fflush(stdout);

	STEP("XAie_DmaTxnCountEnable");
	RC = XAie_DmaTxnCountEnable(Dev, &Sub, 1U, CH_GROUP);
	if (RC != XAIE_OK) {
		return fail("XAie_DmaTxnCountEnable", RC);
	}

	RC = start_oneshot_loopback(Dev, T0, T1);
	if (RC != XAIE_OK) {
		(void)XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);
		return fail("oneshot DMA setup", RC);
	}
	STEP("wait S2MM drain");
	wait_s2mm_drain(Dev, T0);

	probe_counter_binding(Dev, T0, "baseline after DMA");
	probe_counter_binding(Dev, T1, "baseline after DMA");

	STEP("XAie_DmaTxnCountGet");
	if (print_mm2s_counts(Dev, T0, CH_GROUP, 1U, &End0) != 0 ||
	    print_mm2s_counts(Dev, T1, CH_GROUP, 1U, &End1) != 0) {
		(void)XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);
		return -1;
	}

	stop_dma_channels(Dev, T0, T1);
	RC = XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);
	if (RC != XAIE_OK) {
		return fail("XAie_DmaTxnCountDisable", RC);
	}

	if (End0 == 0U || End1 == 0U) {
		printf("[debug_dma_txn] FAIL: EndCount stayed 0 (T0=%u T1=%u)\n", End0, End1);
		return -1;
	}
	if (End0 != End1) {
		printf("[debug_dma_txn] WARN: MM2S EndCount parity T0=%u T1=%u\n", End0, End1);
	}
	printf("[debug_dma_txn] test_enable_disable PASS\n");
	return 0;
}

/* Circular Next-BD: FINISHED_BD (End) should keep rising after one START_TASK. */
static int test_circular_nextbd(XAie_DevInst *Dev) {
	AieRC RC;
	XAie_LocType T0 = XAie_TileLoc(TILE_COL, TILE_ROW0);
	XAie_LocType T1 = XAie_TileLoc(TILE_COL, TILE_ROW1);
	XAie_SubPartition Sub = {{TILE_COL, 1U}, {TILE_ROW0, 2U}};
	u32 End0 = 0U, End1 = 0U;

	printf("[debug_dma_txn] --- test_circular_nextbd ---\n");
	fflush(stdout);

	STEP("XAie_DmaTxnCountEnable");
	RC = XAie_DmaTxnCountEnable(Dev, &Sub, 1U, CH_GROUP);
	if (RC != XAIE_OK) {
		return fail("XAie_DmaTxnCountEnable (circular)", RC);
	}

	RC = start_circular_loopback(Dev, T0, T1);
	if (RC != XAIE_OK) {
		(void)XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);
		return fail("circular DMA setup", RC);
	}

	STEP("XAie_DmaTxnCountGet");
	if (print_mm2s_counts(Dev, T0, CH_GROUP, 2U, &End0) != 0 ||
	    print_mm2s_counts(Dev, T1, CH_GROUP, 2U, &End1) != 0) {
		stop_dma_channels(Dev, T0, T1);
		(void)XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);
		return -1;
	}

	stop_dma_channels(Dev, T0, T1);
	(void)XAie_DmaTxnCountDisable(Dev, &Sub, 1U, CH_GROUP);

	if (End0 < 2U || End1 < 2U) {
		printf("[debug_dma_txn] FAIL: circular EndCount expected >= 2 (T0=%u T1=%u)\n",
		       End0, End1);
		return -1;
	}
	printf("[debug_dma_txn] test_circular_nextbd PASS (End0=%u End1=%u)\n",
	       End0, End1);
	return 0;
}

/* Use case #3: counters armed by XAie_PartitionInitialize_v2. */
static int test_partition_init_v2(XAie_DevInst *Dev) {
	AieRC RC;
	XAie_LocType T0 = XAie_TileLoc(TILE_COL, TILE_ROW0);
	XAie_LocType T1 = XAie_TileLoc(TILE_COL, TILE_ROW1);
	XAie_SubPartition Sub = {{TILE_COL, 1U}, {TILE_ROW0, 2U}};
	XAie_PartInitOpts_v2 Opts;
	u32 End0 = 0U, End1 = 0U;
	int ret;

	printf("[debug_dma_txn] --- test_partition_init_v2 ---\n");

	/*
	 * Mirror the aie-rt stest sequence: the partition is already initialized
	 * at this point, so tear it down before re-initializing through _v2.
	 * Re-initializing a live partition resets columns underneath the DMA
	 * state the loopback needs.
	 */
	STEP("XAie_PartitionTeardown");
	RC = XAie_PartitionTeardown(Dev);
	if (RC != XAIE_OK) {
		printf("[debug_dma_txn] SKIP v2: PartitionTeardown RC=0x%x\n", RC);
		return 0;
	}

	Opts.SubParts = &Sub;
	Opts.NumSubParts = 1U;
	Opts.InitOpts = XAIE_PART_INIT_OPT_DEFAULT;
	Opts.DmaTxnCountChGroup = CH_GROUP;

	RC = XAie_PartitionInitialize_v2(Dev, &Opts);
	if (RC != XAIE_OK) {
		printf("[debug_dma_txn] SKIP v2: PartitionInitialize_v2 RC=0x%x\n", RC);
		(void)XAie_PartitionInitialize(Dev, NULL);
		return 0;
	}

	probe_counter_binding(Dev, T0, "after _v2 arm");
	probe_counter_binding(Dev, T1, "after _v2 arm");

	RC = start_oneshot_loopback(Dev, T0, T1);
	if (RC != XAIE_OK) {
		(void)XAie_PartitionTeardown_v2(Dev, &Opts);
		return fail("v2 oneshot DMA setup", RC);
	}
	wait_s2mm_drain(Dev, T0);

	probe_counter_binding(Dev, T0, "after DMA");
	probe_counter_binding(Dev, T1, "after DMA");
	ret = 0;
	if (print_mm2s_counts(Dev, T0, CH_GROUP, 1U, &End0) != 0 ||
	    print_mm2s_counts(Dev, T1, CH_GROUP, 1U, &End1) != 0) {
		ret = -1;
	} else if (End0 == 0U || End1 == 0U) {
		printf("[debug_dma_txn] FAIL: v2 EndCount stayed 0 (T0=%u T1=%u)\n", End0, End1);
		ret = -1;
	}

	stop_dma_channels(Dev, T0, T1);
	(void)XAie_PartitionTeardown_v2(Dev, &Opts);
	/* Restore the partition state the rest of the app expects. */
	(void)XAie_PartitionInitialize(Dev, NULL);
	if (ret == 0) {
		printf("[debug_dma_txn] test_partition_init_v2 PASS\n");
	}
	return ret;
}

/* Use case #3 disable: DmaTxnCountChGroup=0 must not arm counters. */
static int test_partition_init_v2_off(XAie_DevInst *Dev) {
	AieRC RC;
	XAie_LocType T0 = XAie_TileLoc(TILE_COL, TILE_ROW0);
	XAie_LocType T1 = XAie_TileLoc(TILE_COL, TILE_ROW1);
	XAie_SubPartition Sub = {{TILE_COL, 1U}, {TILE_ROW0, 2U}};
	XAie_PartInitOpts_v2 Opts;
	int A0, A1;

	printf("[debug_dma_txn] --- test_partition_init_v2_off ---\n");

	STEP("XAie_PartitionTeardown");
	RC = XAie_PartitionTeardown(Dev);
	if (RC != XAIE_OK) {
		printf("[debug_dma_txn] SKIP v2_off: PartitionTeardown RC=0x%x\n", RC);
		return 0;
	}

	Opts.SubParts = &Sub;
	Opts.NumSubParts = 1U;
	Opts.InitOpts = XAIE_PART_INIT_OPT_DEFAULT;
	Opts.DmaTxnCountChGroup = 0U;

	RC = XAie_PartitionInitialize_v2(Dev, &Opts);
	if (RC != XAIE_OK) {
		printf("[debug_dma_txn] FAIL: PartitionInitialize_v2(ChGroup=0) RC=0x%x\n", RC);
		(void)XAie_PartitionInitialize(Dev, NULL);
		return -1;
	}

	A0 = probe_counter_binding(Dev, T0, "after _v2 ChGroup=0");
	A1 = probe_counter_binding(Dev, T1, "after _v2 ChGroup=0");
	(void)XAie_PartitionTeardown_v2(Dev, &Opts);
	(void)XAie_PartitionInitialize(Dev, NULL);

	if (A0 < 0 || A1 < 0) {
		return fail("v2_off counter probe", XAIE_ERR);
	}
	if (A0 != 0 || A1 != 0) {
		printf("[debug_dma_txn] FAIL: ChGroup=0 still armed counters (T0=%d T1=%d)\n",
		       A0, A1);
		return -1;
	}
	printf("[debug_dma_txn] test_partition_init_v2_off PASS\n");
	return 0;
}

int main(int argc, char *argv[]) {
	AieRC RC;
	int rc_en, rc_circ, rc_v2, rc_v2off;
	(void)argc;
	(void)argv;

	/*
	 * Caches are left enabled: this example only reaches the array through
	 * MMIO (XAie_DataMemBlockWrite and register reads), never through a
	 * shared DDR buffer, and disabling the D-cache makes the driver's
	 * partition-init busy loops take minutes instead of seconds.
	 */
	XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
			 XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START,
			 XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);
	XAie_InstDeclare(DevInst, &ConfigPtr);

#ifndef __AIESIM__
	/*
	 * IsProd defaults to enabled, which routes XAie_PartitionInitialize to
	 * _XAie_BaremetalIO_PrivilegeInitPart. That function's body is behind
	 * #ifdef XAIE_PROD and aiehlc.sh builds the driver without it, so it
	 * returns XAIE_OK without enabling the column clock buffers: the first
	 * AIE tile register access then stalls the AXI bus. Disabling it takes
	 * the generic privileged path, which does enable the clocks.
	 * Must precede XAie_CfgInitialize.
	 */
	STEP("XAie_SetXprodEnable(DISABLE)");
	RC = XAie_SetXprodEnable(&DevInst, XAIE_DISABLE);
	if (RC != XAIE_OK) {
		return fail("XAie_SetXprodEnable", RC);
	}
#endif

	STEP("XAie_CfgInitialize");
	RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
	if (RC != XAIE_OK) {
		return fail("XAie_CfgInitialize", RC);
	}

#ifdef __AIESIM__
	RC = XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
	if (RC != XAIE_OK) {
		return fail("XAie_SetIOBackend SIM", RC);
	}
#else
	RC = XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
	if (RC != XAIE_OK) {
		return fail("XAie_SetIOBackend", RC);
	}
#if AIE_GEN >= 2
	if (DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
#if AIE_GEN == 5
		RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
		RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
		if (RC != XAIE_OK) {
			return fail("XAie_UpdateNpiAddr", RC);
		}
	}
	STEP("XAie_PartitionInitialize");
	RC = XAie_PartitionInitialize(&DevInst, NULL);
	if (RC != XAIE_OK) {
		return fail("XAie_PartitionInitialize", RC);
	}
#endif
#endif
	STEP("device ready");

	rc_en = test_enable_disable(&DevInst);
	rc_circ = test_circular_nextbd(&DevInst);
	rc_v2 = test_partition_init_v2(&DevInst);
	rc_v2off = test_partition_init_v2_off(&DevInst);

	if (rc_en == 0 && rc_circ == 0 && rc_v2 == 0 && rc_v2off == 0) {
		printf("[debug_dma_txn] Test Passed\n");
		return 0;
	}
	printf("[debug_dma_txn] Test Failed (enable=%d circular=%d v2=%d v2off=%d)\n",
	       rc_en, rc_circ, rc_v2, rc_v2off);
	fflush(stdout);
	return 1;
}
