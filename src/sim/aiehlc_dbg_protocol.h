// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Layout must stay byte-identical to naiebaremetal src/ipc/ps_ipc_protocol.h;
// the same debug-UI client decodes both.

#ifndef AIEHLC_DBG_PROTOCOL_H
#define AIEHLC_DBG_PROTOCOL_H

#include <stdint.h>

typedef uint8_t aiehlc_dbg_cmd_t;

enum {
    AIEHLC_DBG_CMD_PING = 0x01u,

    AIEHLC_DBG_CMD_WRITE32 = 0x10u,
    AIEHLC_DBG_CMD_READ32 = 0x11u,

    AIEHLC_DBG_CMD_NPI_WRITE32 = 0x12u,
    AIEHLC_DBG_CMD_NPI_READ32 = 0x13u,
};

typedef uint8_t aiehlc_dbg_status_t;

enum {
    AIEHLC_DBG_STATUS_OK = 0x00u,
    AIEHLC_DBG_STATUS_ERR_PROTO = 0x01u,
    AIEHLC_DBG_STATUS_ERR_IO = 0x02u,
};

#pragma pack(push, 1)

typedef struct {
    aiehlc_dbg_cmd_t cmd;
    uint8_t _reserved[3];
    uint64_t arg1;
    uint32_t arg2;
} aiehlc_dbg_request_t;

typedef struct {
    aiehlc_dbg_status_t status;
    uint8_t _reserved[7];
    uint64_t value;
} aiehlc_dbg_response_t;

#pragma pack(pop)

#define AIEHLC_DBG_SOCKET_PATH_MAX 256

#define AIEHLC_DBG_ENV_DIR "AIEHLC_DBG_DIR"
// "1" enables writes; any other value makes WRITE32 / NPI_WRITE32 return ERR_PROTO.
#define AIEHLC_DBG_ENV_ALLOW_WRITE "AIEHLC_DBG_ALLOW_WRITE"
// Seconds of debug-socket idle before the simulator exits; "0" disables the hold.
#define AIEHLC_DBG_ENV_HOLD_SEC "AIEHLC_DBG_HOLD_SEC"
#define AIEHLC_DBG_ENV_POLL_NS "AIEHLC_DBG_POLL_NS"

#ifdef __cplusplus
#include <cerrno>
#include <cstddef>
#include <unistd.h>

namespace aiehlc_dbg {

inline int send_all(int fd, const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t sent = ::write(fd, p, remaining);
        if (sent < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (sent == 0)
            return -1;
        p += sent;
        remaining -= static_cast<size_t>(sent);
    }
    return 0;
}

inline int recv_all(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t got = ::read(fd, p, remaining);
        if (got < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (got == 0)
            return -1;
        p += got;
        remaining -= static_cast<size_t>(got);
    }
    return 0;
}

} // namespace aiehlc_dbg
#endif // __cplusplus

#endif // AIEHLC_DBG_PROTOCOL_H
