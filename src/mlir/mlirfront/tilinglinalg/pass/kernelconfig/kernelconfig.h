/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//====================================================================
// kernelconfig.h  —  BCF/PRX generation for tilinglinalg kernel compilation
//====================================================================
#ifndef TILINGLINALG_KERNELCONFIG_H
#define TILINGLINALG_KERNELCONFIG_H

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// ──────────────────────────────────────────────────────────────
// TilingBcf  —  generates BCF (Bridge Configuration File) for xchesscc linker
// ──────────────────────────────────────────────────────────────
// BCF format pins buffer symbols to specific DM addresses so the xchesscc
// linker places them where the host-side DMA BDs expect.
//
// Output example:
//   _entry_point _main_init
//   _symbol _main _after  _main_init
//   _symbol _main_init 0
//   _symbol buf_out_ping_0 0x78000
//   _symbol buf_out_pong_0 0x78020
//   _stack DM_stack 0x70000 0x2800
//   _reserved DMb 0x40000 0x10000

class TilingBcf {
  public:
    TilingBcf() : stackAddr_(0x70000), stackSize_(0x2800) {}

    void addSymbol(const std::string &name, uint32_t addr) { symbols_.push_back({name, addr}); }

    void setStack(uint32_t addr, uint32_t size) {
        stackAddr_ = addr;
        stackSize_ = size;
    }

    void addReservedDMB(uint32_t addr, uint32_t len) { reservedDMB_.push_back({addr, len}); }

    std::string generate() const {
        // Sort symbols by address
        auto sorted = symbols_;
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        std::ostringstream ostr;
        ostr << "_entry_point _main_init\n";
        ostr << "_symbol _main _after  _main_init\n";
        ostr << "_symbol _main_init 0\n";
        for (const auto &sym : sorted) {
            ostr << "_symbol " << sym.first << " 0x" << std::hex << sym.second << "\n";
        }
        ostr << "_stack DM_stack 0x" << std::hex << stackAddr_ << " 0x" << std::hex << stackSize_ << "\n";
        for (const auto &r : reservedDMB_) {
            ostr << "_reserved DMb 0x" << std::hex << r.first << " 0x" << std::hex << r.second << "\n";
        }
        return ostr.str();
    }

    bool exportToFile(const std::string &path) const {
        std::ofstream ofs(path);
        if (!ofs.is_open())
            return false;
        ofs << generate();
        ofs.close();
        return true;
    }

  private:
    std::vector<std::pair<std::string, uint32_t>> symbols_;
    uint32_t stackAddr_;
    uint32_t stackSize_;
    std::vector<std::pair<uint32_t, uint32_t>> reservedDMB_;
};

// ──────────────────────────────────────────────────────────────
// TilingPrx  —  generates PRX (Project XML) for xchesscc/xchessmk
// ──────────────────────────────────────────────────────────────
// PRX format is XML that tells xchessmk where to find the BCF, kernel sources,
// and what compiler options to use.
//
// Output example:
//   <project name="kernel" processor="me">
//     <file type="lbc" name="kernel.ll" path="./build/"/>
//     <issinit/>
//     <option id="cpp.define" value="__AIENGINE__ __AIEARCH__=22" inherit="1"/>
//     <option id="llvm.xargs" value="-fno-jump-tables ..." inherit="1"/>
//     <option id="bridge.cfg" value="aieml.bcf" path="./"/>
//     <option id="project.name" value="kernel"/>
//     <option id="project.type" value="exe"/>
//   </project>

class TilingPrx {
  public:
    TilingPrx(const std::string &projectName = "kernel", int aieArch = 22)
        : projectName_(projectName), aieArch_(aieArch), bcfPath_("aieml.bcf"), kernelLLPath_("./build/") {}

    void setBcfPath(const std::string &bcfPath) { bcfPath_ = bcfPath; }
    void setKernelLLPath(const std::string &llPath) { kernelLLPath_ = llPath; }
    void setProjectName(const std::string &name) { projectName_ = name; }

    std::string generate() const {
        std::ostringstream ostr;
        ostr << "<project name=\"" << projectName_ << "\" processor=\"me\">\n";
        ostr << "    <file type=\"lbc\" name=\"kernel.ll\" path=\"" << kernelLLPath_ << "\"/>\n";
        ostr << "    <issinit/>\n";
        ostr << "    <option id=\"cpp.define\" value=\"__AIENGINE__ __AIEARCH__=" << std::dec << aieArch_
             << "\" inherit=\"1\"/>\n";
        ostr << "    <option id=\"llvm.xargs\" value=\"-fno-jump-tables "
             << "-fno-discard-value-names "
             << "-mllvm -chess-collapse-struct-types-during-linking=0\" inherit=\"1\"/>\n";
        ostr << "    <option id=\"bridge.cfg\" value=\"" << bcfPath_ << "\" path=\"./\"/>\n";
        ostr << "    <option id=\"project.name\" value=\"" << projectName_ << "\"/>\n";
        ostr << "    <option id=\"project.type\" value=\"exe\"/>\n";
        ostr << "</project>\n";
        return ostr.str();
    }

    bool exportToFile(const std::string &path) const {
        std::ofstream ofs(path);
        if (!ofs.is_open())
            return false;
        ofs << generate();
        ofs.close();
        return true;
    }

  private:
    std::string projectName_;
    int aieArch_;
    std::string bcfPath_;
    std::string kernelLLPath_;
};

#endif // TILINGLINALG_KERNELCONFIG_H
