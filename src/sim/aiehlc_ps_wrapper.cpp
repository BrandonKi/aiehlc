/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

// Avoid issue with compilation error caused by boost::log's MPL enum arithmetic in Vitis 2026.2
#ifndef __LOGGING_H__
#define __LOGGING_H__
#include <systemc.h>
#define SC_LOG(lvl)                                                                                                    \
    while (false)                                                                                                      \
    std::cerr
#define SC_LOG_TRACE                                                                                                   \
    while (false)                                                                                                      \
    std::cerr
#define SC_LOG_DEBUG                                                                                                   \
    while (false)                                                                                                      \
    std::cerr
#endif

#include <adf/wrapper/me_ip_block.h>
#include <xtlm.h>
#include "ioutils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef __global__
#define __global__
#endif
#ifdef __cplusplus
extern "C" void host_canonicalized();
#else
void host_canonicalized();
#endif

#ifdef AIEHLC_HOST_SRC
#  include AIEHLC_HOST_SRC
#else
#  error "Set AIEHLC_HOST_SRC to the absolute path of host.cc on the compiler command line"
#endif

extern int         ps_main_complete;
extern int         graph_return_value;
extern std::string g_pkg_dir;

extern "C" {
    void         start_plios();
    void         Write32  (uint64_t addr, unsigned int  data);
    unsigned int Read32   (uint64_t addr);
    void         WriteCmd (unsigned char Command,
                           unsigned char ColId, unsigned char RowId,
                           unsigned int CmdWd0, unsigned int CmdWd1,
                           unsigned char* CmdStr);
}

extern int aiehlc_ps_main(int argc, char** argv);

#define BUSWIDTH 128

class PSIP_aiehlc : public IPBlock {
    SC_HAS_PROCESS(PSIP_aiehlc);

public:
    sc_event_queue                              toggle_AIE_array_clk;
    xtlm::xtlm_aximm_initiator_socket          PS_AxiMM_Rd;
    xtlm::xtlm_aximm_initiator_socket          PS_AxiMM_Wr;
    xtlm::xtlm_aximm_initiator_rd_socket_util* PS_AxiMM_Rd_Util;
    xtlm::xtlm_aximm_initiator_wr_socket_util* PS_AxiMM_Wr_Util;
    xtlm::xtlm_aximm_mem_manager*              mem_manager;

    static PSIP_aiehlc* createInstance(sc_module_name name);
    static PSIP_aiehlc* getInstance();

    void     write32  (uint64_t Addr, uint32_t Data);
    uint32_t read32   (uint64_t Addr);
    void     write128 (uint64_t Addr, uint32_t* Data);
    void     read128  (uint64_t Addr, uint32_t* Data);
    void     writeGM  (uint64_t addr, const void* data, uint64_t size);
    void     readGM   (uint64_t addr, void* data, uint64_t size);

private:
    explicit PSIP_aiehlc(sc_module_name nm);
    static PSIP_aiehlc* psObj;

    sc_event transRspAvail;

    void aximm_transaction(
        xtlm::xtlm_aximm_initiator_rd_socket_util& rd_util,
        xtlm::xtlm_aximm_initiator_wr_socket_util& wr_util,
        xtlm::xtlm_command command,
        unsigned long long address,
        unsigned char* pData,
        unsigned int trans_size_in_bytes);

    void set_payload_attr(xtlm::aximm_payload* trans, size_t transBytes);
    void main_action();
    void response_process();

    template <typename T>
    void CallPsMainFunction(T func) {
        using ReturnType = decltype(func());
        if constexpr (!std::is_void<ReturnType>::value) {
            graph_return_value = static_cast<int>(func());
        } else {
            func();
        }
    }
};

PSIP_aiehlc* PSIP_aiehlc::psObj = nullptr;

PSIP_aiehlc::PSIP_aiehlc(sc_module_name nm)
  : IPBlock(nm)
  , toggle_AIE_array_clk("toggle_AIE_clk")
  , PS_AxiMM_Rd("ps_axi_rd", BUSWIDTH)
  , PS_AxiMM_Wr("ps_axi_wr", BUSWIDTH)
{
    PS_AxiMM_Rd_Util = new xtlm::xtlm_aximm_initiator_rd_socket_util(
        "PS_AxiMM_Util_rd_socket", xtlm::aximm::TRANSACTION, BUSWIDTH);
    PS_AxiMM_Wr_Util = new xtlm::xtlm_aximm_initiator_wr_socket_util(
        "PS_AxiMM_Util_wr_socket", xtlm::aximm::TRANSACTION, BUSWIDTH);
    mem_manager = new xtlm::xtlm_aximm_mem_manager(this);

    PS_AxiMM_Rd_Util->rd_socket.bind(PS_AxiMM_Rd);
    PS_AxiMM_Wr_Util->wr_socket.bind(PS_AxiMM_Wr);

    SC_THREAD(main_action);

    SC_THREAD(response_process);
    sensitive << (PS_AxiMM_Wr_Util->resp_available);
    sensitive << (PS_AxiMM_Rd_Util->data_available);

    std::cout << "IP-INFO: [" << basename() << "] AIEHLC PS IP loaded." << std::endl;
}

PSIP_aiehlc* PSIP_aiehlc::createInstance(sc_module_name name) {
    if (!psObj) psObj = new PSIP_aiehlc(name);
    return psObj;
}
PSIP_aiehlc* PSIP_aiehlc::getInstance() { return psObj; }

void PSIP_aiehlc::set_payload_attr(xtlm::aximm_payload* trans, size_t transBytes) {
    trans->create_and_get_data_ptr(transBytes);
    trans->set_data_length(transBytes);
    trans->set_burst_length(1);
}

void PSIP_aiehlc::aximm_transaction(
    xtlm::xtlm_aximm_initiator_rd_socket_util& rd_util,
    xtlm::xtlm_aximm_initiator_wr_socket_util& wr_util,
    xtlm::xtlm_command command,
    unsigned long long address,
    unsigned char* pData,
    unsigned int trans_size_in_bytes)
{
    unsigned int n16 = (trans_size_in_bytes + 15) / 16;
    xtlm::aximm_payload* payload = mem_manager->get_payload();
    payload->acquire();
    payload->set_response_status(xtlm::XTLM_INCOMPLETE_RESPONSE);
    payload->set_command(command);
    payload->set_data_ptr(pData, trans_size_in_bytes);
    payload->set_address(address);
    payload->set_burst_type(1);
    payload->set_burst_length(n16);
    payload->set_burst_size(16);
    sc_time delay = SC_ZERO_TIME;
    if (command == xtlm::XTLM_READ_COMMAND) {
        if (!rd_util.is_slave_ready()) wait(rd_util.transaction_sampled);
        rd_util.send_transaction(*payload, delay);
        wait(rd_util.data_available);
        payload = rd_util.get_data();
    } else {
        if (!wr_util.is_slave_ready()) wait(wr_util.transaction_sampled);
        wr_util.send_transaction(*payload, delay);
        wait(wr_util.resp_available);
        payload = wr_util.get_resp();
    }
}

void PSIP_aiehlc::write32(uint64_t Addr, uint32_t Data) {
    xtlm::aximm_payload* trans = mem_manager->get_payload();
    trans->acquire();
    set_payload_attr(trans, sizeof(uint32_t));
    trans->set_command(xtlm::XTLM_WRITE_COMMAND);
    trans->set_address(Addr);
    memcpy(trans->get_data_ptr(), &Data, sizeof(uint32_t));
    sc_time delay = SC_ZERO_TIME;
    PS_AxiMM_Wr_Util->b_transport(*trans, delay);
    trans->release();
}

uint32_t PSIP_aiehlc::read32(uint64_t Addr) {
    xtlm::aximm_payload* trans = mem_manager->get_payload();
    trans->acquire();
    set_payload_attr(trans, sizeof(uint32_t));
    trans->set_command(xtlm::XTLM_READ_COMMAND);
    trans->set_address(Addr);
    sc_time delay = SC_ZERO_TIME;
    PS_AxiMM_Rd_Util->b_transport(*trans, delay);
    uint32_t data = *(uint32_t*)trans->get_data_ptr();
    trans->release();
    if (getenv("AIE_SYNC_READ")) wait(10, SC_NS);
    return data;
}

void PSIP_aiehlc::write128(uint64_t Addr, uint32_t* Data) {
    xtlm::aximm_payload* trans = mem_manager->get_payload();
    trans->acquire();
    set_payload_attr(trans, 4 * sizeof(uint32_t));
    trans->set_command(xtlm::XTLM_WRITE_COMMAND);
    trans->set_address(Addr);
    memcpy(trans->get_data_ptr(), Data, 4 * sizeof(uint32_t));
    sc_time delay = SC_ZERO_TIME;
    PS_AxiMM_Wr_Util->b_transport(*trans, delay);
    trans->release();
}

void PSIP_aiehlc::read128(uint64_t Addr, uint32_t* Data) {
    xtlm::aximm_payload* trans = mem_manager->get_payload();
    trans->acquire();
    set_payload_attr(trans, 4 * sizeof(uint32_t));
    trans->set_command(xtlm::XTLM_READ_COMMAND);
    trans->set_address(Addr);
    sc_time delay = SC_ZERO_TIME;
    PS_AxiMM_Rd_Util->b_transport(*trans, delay);
    memcpy(Data, trans->get_data_ptr(), 4 * sizeof(uint32_t));
    trans->release();
}

void PSIP_aiehlc::writeGM(uint64_t addr, const void* data, uint64_t size) {
    toggle_AIE_array_clk.notify(1, SC_NS);
    uint64_t remaining = size;
    uint64_t cur = addr;
    unsigned char* ptr = (unsigned char*)data;
    unsigned int n16 = (size + 15) / 16;
    xtlm::aximm_payload* payload = mem_manager->get_payload();
    payload->acquire();
    payload->set_response_status(xtlm::XTLM_INCOMPLETE_RESPONSE);
    payload->set_command(xtlm::XTLM_WRITE_COMMAND);
    payload->set_data_ptr((unsigned char*)data, size);
    payload->set_address(addr);
    payload->set_burst_type(1);
    payload->set_burst_length(n16);
    payload->set_burst_size(16);
    unsigned int ret = PS_AxiMM_Wr_Util->transport_dbg(*payload);
    if (ret != 0) { payload->release(); return; }
    while (remaining >= 4096) {
        aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                          xtlm::XTLM_WRITE_COMMAND, cur, ptr, 4096);
        cur += 4096; ptr += 4096; remaining -= 4096;
    }
    if (remaining > 0)
        aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                          xtlm::XTLM_WRITE_COMMAND, cur, ptr, (unsigned)remaining);
    toggle_AIE_array_clk.notify(SC_ZERO_TIME);
}

void PSIP_aiehlc::readGM(uint64_t addr, void* data, uint64_t size) {
    toggle_AIE_array_clk.notify(1, SC_NS);
    uint64_t remaining = size;
    uint64_t cur = addr;
    unsigned char* ptr = (unsigned char*)data;
    unsigned int n16 = (size + 15) / 16;
    xtlm::aximm_payload* payload = mem_manager->get_payload();
    payload->acquire();
    payload->set_response_status(xtlm::XTLM_INCOMPLETE_RESPONSE);
    payload->set_command(xtlm::XTLM_READ_COMMAND);
    payload->set_data_ptr((unsigned char*)data, size);
    payload->set_address(addr);
    payload->set_burst_type(1);
    payload->set_burst_length(n16);
    payload->set_burst_size(16);
    unsigned int ret = PS_AxiMM_Rd_Util->transport_dbg(*payload);
    if (ret != 0) { payload->release(); return; }
    while (remaining >= 4096) {
        aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                          xtlm::XTLM_READ_COMMAND, cur, ptr, 4096);
        cur += 4096; ptr += 4096; remaining -= 4096;
    }
    if (remaining > 0)
        aximm_transaction(*PS_AxiMM_Rd_Util, *PS_AxiMM_Wr_Util,
                          xtlm::XTLM_READ_COMMAND, cur, ptr, (unsigned)remaining);
    toggle_AIE_array_clk.notify(SC_ZERO_TIME);
}

static void         PSWrite32    (uint64_t a, unsigned int d)         { PSIP_aiehlc::getInstance()->write32(a, d); }
static unsigned int PSRead32     (uint64_t a)                         { return PSIP_aiehlc::getInstance()->read32(a); }
static void         PSWriteGM    (uint64_t a, const void* d, uint64_t s) { PSIP_aiehlc::getInstance()->writeGM(a, d, s); }
static void         PSReadGM     (uint64_t a, void* d, uint64_t s)    { PSIP_aiehlc::getInstance()->readGM(a, d, s); }
static void         PSNpiWrite32 (uint64_t a, unsigned int d)         { Write32(a, d); }
static unsigned int PSNpiRead32  (uint64_t a)                         { return Read32(a); }
static void         PSWriteCmd   (unsigned char c, unsigned char col, unsigned char row,
                                   unsigned int w0, unsigned int w1, unsigned char* s) { WriteCmd(c,col,row,w0,w1,s); }
static void         PSStartPLIO  ()                                   { start_plios(); }

void PSIP_aiehlc::response_process() {
    while (true) {
        wait();
        if (PS_AxiMM_Rd_Util->is_data_available())
            transRspAvail.notify(SC_ZERO_TIME);
        if (PS_AxiMM_Wr_Util->is_resp_available())
            transRspAvail.notify(SC_ZERO_TIME);
    }
}

void PSIP_aiehlc::main_action() {
    std::cout << "IP-INFO: [" << basename() << "] AIEHLC PS IP started." << std::endl;

    setWrite32Ptr   (PSWrite32);
    setRead32Ptr    (PSRead32);
    setWriteGMPtr   (PSWriteGM);
    setReadGMPtr    (PSReadGM);
    setWriteCmdPtr  (PSWriteCmd);
    setNpiWrite32Ptr(PSNpiWrite32);
    setNpiRead32Ptr (PSNpiRead32);
    setPLIOStartPtr (PSStartPLIO);

    if (const char* wd = getenv("AIE_WORK_DIR")) g_pkg_dir = wd;

    CallPsMainFunction([](){ return aiehlc_ps_main(0, nullptr); });

    ps_main_complete = 1;
    std::cout << "IP-INFO: [" << basename() << "] AIEHLC PS main completed." << std::endl;
}

extern "C" {

void     ess_Write128(uint64_t Addr, uint32_t* Data) { PSIP_aiehlc::getInstance()->write128(Addr, Data); }
void     ess_Read128 (uint64_t Addr, uint32_t* Data) { PSIP_aiehlc::getInstance()->read128 (Addr, Data); }

IPBlock* create_ip   (sc_module_name name) { return PSIP_aiehlc::createInstance(name); }
void     destroy_ip  (IPBlock* ip)         { delete ip; }

} // extern "C"
