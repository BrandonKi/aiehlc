// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.verbatim "#include <stdint.h>"
  emitc.verbatim "#include <adf.h>"
  emitc.verbatim "#include <aie_api/aie.hpp>"
  emitc.verbatim "#include <aie_api/aie_adf.hpp>"
  emitc.verbatim "#define FOR_READ  1"
  emitc.verbatim "#define FOR_WRITE 0"
  emitc.verbatim "#define BUF_SZ_IN_0 1024"
  emitc.verbatim "#define BUF_SZ_IN_1 1024"
  emitc.verbatim "#define BUF_SZ_OUT_0 512"
  emitc.verbatim "#define BUF_SZ 1024"
  emitc.verbatim "inline int8_t* acquire_output_window(output_window_int8* win) {\0A  window_internal* w = (window_internal*)win;\0A  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\0A  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\0A  acquire_greater_equal(w->lockids[0], 1);\0A  return (int8_t*)w->ptr;\0A}"
  emitc.verbatim "inline void release_output_window(output_window_int8* win) {\0A  chess_memory_fence();\0A  window_internal* w = (window_internal*)win;\0A  release(w->lockids[1], 1);\0A  w->heads[w->current_bufid] = w->head;\0A  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\0A}"
  emitc.verbatim "inline int8_t* acquire_input_window(input_window_int8* win) {\0A  window_internal* w = (window_internal*)win;\0A  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\0A  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\0A  acquire_greater_equal(w->lockids[1], 1);\0A  return (int8_t*)w->ptr;\0A}"
  emitc.verbatim "inline void release_input_window(input_window_int8* win) {\0A  chess_memory_fence();\0A  window_internal* w = (window_internal*)win;\0A  release(w->lockids[0], 1);\0A  w->heads[w->current_bufid] = w->head;\0A  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\0A}"
  emitc.verbatim "#define LOCK_window_in_0_ACQ 48"
  emitc.verbatim "#define LOCK_window_in_0_REL 49"
  emitc.verbatim "v4int8 buf_in_ping_0[1024];"
  emitc.verbatim "v4int8 buf_in_pong_0[1024];"
  emitc.verbatim "// window_def window_in_0"
  emitc.verbatim "#define LOCK_window_in_1_ACQ 50"
  emitc.verbatim "#define LOCK_window_in_1_REL 51"
  emitc.verbatim "v4int8 buf_in_ping_1[1024];"
  emitc.verbatim "v4int8 buf_in_pong_1[1024];"
  emitc.verbatim "// window_def window_in_1"
  emitc.verbatim "#define LOCK_window_out_0_ACQ 52"
  emitc.verbatim "#define LOCK_window_out_0_REL 53"
  emitc.verbatim "v4int8 buf_out_ping_0[512];"
  emitc.verbatim "v4int8 buf_out_pong_0[512];"
  emitc.verbatim "// window_def window_out_0"
  emitc.verbatim "#include \22kernel_log.h\22"
  emitc.verbatim "#include \22matmul.cc\22"
  emitc.verbatim "// kernel_decl matmul"
  emitc.func @main() -> i32 {
    emitc.verbatim "volatile static int sync_buffer[8] = {0, -1};"
    emitc.verbatim "sync_buffer[0] = 0;"
    emitc.verbatim "klog_init();"
    emitc.verbatim "// alloc_sync_buffer"
    emitc.verbatim "// sync_buffer_write"
    emitc.verbatim "// log(...)"
    emitc.verbatim "window_internal window_window_in_0[1];"
    emitc.verbatim "window_init(window_window_in_0, 1, buf_in_ping_0, LOCK_window_in_0_ACQ, buf_in_pong_0, LOCK_window_in_0_REL, 1024, 1024);"
    emitc.verbatim "window_internal window_window_in_1[1];"
    emitc.verbatim "window_init(window_window_in_1, 1, buf_in_ping_1, LOCK_window_in_1_ACQ, buf_in_pong_1, LOCK_window_in_1_REL, 1024, 1024);"
    emitc.verbatim "window_internal window_window_out_0[1];"
    emitc.verbatim "window_init(window_window_out_0, 1, buf_out_ping_0, LOCK_window_out_0_ACQ, buf_out_pong_0, LOCK_window_out_0_REL, 512, 512);"
    emitc.verbatim "// kernel_invoke matmul"
    emitc.verbatim "matmul(get_input_async_window_int8(window_window_in_0), get_input_async_window_int8(window_window_in_1), get_output_async_window_int8(window_window_out_0));"
    emitc.verbatim "done();"
    %0 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.return %0 : i32
  }
}
