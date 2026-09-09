// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "aiehlc_dbg_server.h"
#include "aiehlc_dbg_protocol.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace {

aiehlc_dbg_callbacks_t g_cb = {nullptr, nullptr, nullptr, nullptr};
aiehlc_dbg_addr_info_t g_addr = {0, 0, 0, 0};
void (*g_wake)(void) = nullptr;

std::atomic<bool> g_active{false};
std::atomic<bool> g_stop{false};
std::atomic<bool> g_allow_write{false};

std::string g_socket_path;
std::string g_info_path;
int g_listen_fd = -1;

struct DbgWorkItem {
    aiehlc_dbg_request_t req;
    aiehlc_dbg_response_t resp;
    bool ready;
    std::mutex mtx;
    std::condition_variable cv;
    DbgWorkItem() : ready(false) {
        std::memset(&req, 0, sizeof(req));
        std::memset(&resp, 0, sizeof(resp));
    }
};

std::queue<std::shared_ptr<DbgWorkItem>> g_queue;
std::mutex g_queue_mtx;
std::atomic<uint64_t> g_service_count{0};

bool dbg_verbose() {
    static int v = -1;
    if (v < 0)
        v = std::getenv("AIEHLC_DBG_VERBOSE") ? 1 : 0;
    return v == 1;
}

std::shared_ptr<DbgWorkItem> enqueue(const aiehlc_dbg_request_t &req) {
    auto item = std::make_shared<DbgWorkItem>();
    item->req = req;
    {
        std::lock_guard<std::mutex> lk(g_queue_mtx);
        g_queue.push(item);
    }
    // Wake the SystemC drain thread now; without this the request waits for the
    // next simulated-time tick, which never arrives after the array clock stops.
    if (g_wake)
        g_wake();
    return item;
}

// Must run on the SystemC thread; the register callbacks drive the AXI fabric.
void dispatch(std::shared_ptr<DbgWorkItem> &item) {
    const aiehlc_dbg_request_t &req = item->req;
    aiehlc_dbg_response_t &resp = item->resp;
    std::memset(&resp, 0, sizeof(resp));
    resp.status = AIEHLC_DBG_STATUS_OK;

    switch (req.cmd) {
    case AIEHLC_DBG_CMD_PING:
        break;

    case AIEHLC_DBG_CMD_READ32:
        if (g_cb.read32)
            resp.value = g_cb.read32(static_cast<uint64_t>(req.arg1));
        else
            resp.status = AIEHLC_DBG_STATUS_ERR_IO;
        break;

    case AIEHLC_DBG_CMD_NPI_READ32:
        if (g_cb.npi_read32)
            resp.value = g_cb.npi_read32(static_cast<uint64_t>(req.arg1));
        else
            resp.status = AIEHLC_DBG_STATUS_ERR_IO;
        break;

    case AIEHLC_DBG_CMD_WRITE32:
        if (!g_allow_write.load()) {
            resp.status = AIEHLC_DBG_STATUS_ERR_PROTO;
        } else if (g_cb.write32) {
            g_cb.write32(static_cast<uint64_t>(req.arg1), static_cast<unsigned int>(req.arg2));
        } else {
            resp.status = AIEHLC_DBG_STATUS_ERR_IO;
        }
        break;

    case AIEHLC_DBG_CMD_NPI_WRITE32:
        if (!g_allow_write.load()) {
            resp.status = AIEHLC_DBG_STATUS_ERR_PROTO;
        } else if (g_cb.npi_write32) {
            g_cb.npi_write32(static_cast<uint64_t>(req.arg1), static_cast<unsigned int>(req.arg2));
        } else {
            resp.status = AIEHLC_DBG_STATUS_ERR_IO;
        }
        break;

    default:
        resp.status = AIEHLC_DBG_STATUS_ERR_PROTO;
        break;
    }

    {
        std::lock_guard<std::mutex> lk(item->mtx);
        item->ready = true;
    }
    item->cv.notify_one();
}

void client_thread(int client_fd) {
    using namespace aiehlc_dbg;
    while (!g_stop.load()) {
        aiehlc_dbg_request_t req;
        if (recv_all(client_fd, &req, sizeof(req)) != 0)
            break;
        auto item = enqueue(req);
        {
            std::unique_lock<std::mutex> lk(item->mtx);
            while (!item->ready) {
                if (g_stop.load())
                    break;
                item->cv.wait_for(lk, std::chrono::milliseconds(200));
            }
            if (!item->ready)
                break;
        }
        if (send_all(client_fd, &item->resp, sizeof(item->resp)) != 0)
            break;
    }
    ::close(client_fd);
}

void accept_thread() {
    while (!g_stop.load()) {
        int client_fd = ::accept(g_listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR || errno == ECONNABORTED)
                continue;
            if (errno == EMFILE || errno == ENFILE || errno == ENOBUFS || errno == ENOMEM) {
                if (dbg_verbose())
                    std::fprintf(stderr, "[aiehlc_dbg] accept(): %s; retrying\n", std::strerror(errno));
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            break;
        }
        std::thread(client_thread, client_fd).detach();
    }
}

void write_info_file() {
    if (g_info_path.empty())
        return;
    FILE *f = std::fopen(g_info_path.c_str(), "w");
    if (!f)
        return;
    std::fprintf(f,
                 "{\n"
                 "  \"socket\": \"%s\",\n"
                 "  \"pid\": %d,\n"
                 "  \"base_address\": %llu,\n"
                 "  \"column_shift\": %u,\n"
                 "  \"row_shift\": %u,\n"
                 "  \"aie_gen\": %d,\n"
                 "  \"writes_enabled\": %s\n"
                 "}\n",
                 g_socket_path.c_str(), static_cast<int>(getpid()),
                 static_cast<unsigned long long>(g_addr.base_address), g_addr.column_shift, g_addr.row_shift,
                 g_addr.aie_gen, g_allow_write.load() ? "true" : "false");
    std::fclose(f);
}

} // namespace

extern "C" void aiehlc_dbg_set_callbacks(const aiehlc_dbg_callbacks_t *cb) {
    if (cb)
        g_cb = *cb;
}

extern "C" void aiehlc_dbg_set_wake(void (*wake)(void)) { g_wake = wake; }

extern "C" int aiehlc_dbg_start(const aiehlc_dbg_addr_info_t *addr) {
    if (g_active.load())
        return 0;
    if (addr)
        g_addr = *addr;

    const char *dir = std::getenv(AIEHLC_DBG_ENV_DIR);
    if (!dir || !dir[0]) {
        if (dbg_verbose())
            std::fprintf(stderr, "[aiehlc_dbg] %s not set; debug socket disabled\n", AIEHLC_DBG_ENV_DIR);
        return 1;
    }

    const char *aw = std::getenv(AIEHLC_DBG_ENV_ALLOW_WRITE);
    g_allow_write.store(aw && aw[0] == '1');

    ::mkdir(dir, 0755);

    char path[AIEHLC_DBG_SOCKET_PATH_MAX];
    std::snprintf(path, sizeof(path), "%s/aiehlc_ps_%d.sock.dbg", dir, static_cast<int>(getpid()));
    g_socket_path = path;

    char info[AIEHLC_DBG_SOCKET_PATH_MAX];
    std::snprintf(info, sizeof(info), "%s/dbg_info.json", dir);
    g_info_path = info;

    ::unlink(g_socket_path.c_str());

    g_listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (g_listen_fd < 0) {
        std::perror("[aiehlc_dbg] socket()");
        return 1;
    }

    struct sockaddr_un sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sun_family = AF_UNIX;
    std::strncpy(sa.sun_path, g_socket_path.c_str(), sizeof(sa.sun_path) - 1);

    if (::bind(g_listen_fd, reinterpret_cast<struct sockaddr *>(&sa), sizeof(sa)) < 0) {
        std::perror("[aiehlc_dbg] bind()");
        ::close(g_listen_fd);
        g_listen_fd = -1;
        return 1;
    }
    if (::listen(g_listen_fd, 16) < 0) {
        std::perror("[aiehlc_dbg] listen()");
        ::close(g_listen_fd);
        g_listen_fd = -1;
        return 1;
    }

    g_active.store(true);
    write_info_file();

    if (dbg_verbose())
        std::fprintf(stderr, "[aiehlc_dbg] debug socket listening on %s (writes %s)\n", g_socket_path.c_str(),
                     g_allow_write.load() ? "enabled" : "disabled");

    std::thread(accept_thread).detach();
    return 0;
}

extern "C" int aiehlc_dbg_drain(void) {
    int serviced = 0;
    while (true) {
        std::shared_ptr<DbgWorkItem> item;
        {
            std::lock_guard<std::mutex> lk(g_queue_mtx);
            if (g_queue.empty())
                break;
            item = g_queue.front();
            g_queue.pop();
        }
        dispatch(item);
        ++serviced;
    }
    if (serviced)
        g_service_count.fetch_add(serviced);
    return serviced;
}

extern "C" uint64_t aiehlc_dbg_service_count(void) { return g_service_count.load(); }

extern "C" void aiehlc_dbg_stop(void) {
    if (!g_active.exchange(false))
        return;
    g_stop.store(true);
    if (g_listen_fd >= 0) {
        ::shutdown(g_listen_fd, SHUT_RDWR);
        ::close(g_listen_fd);
        g_listen_fd = -1;
    }
    if (!g_socket_path.empty())
        ::unlink(g_socket_path.c_str());
    if (!g_info_path.empty())
        ::unlink(g_info_path.c_str());
}

extern "C" int aiehlc_dbg_active(void) { return g_active.load() ? 1 : 0; }
