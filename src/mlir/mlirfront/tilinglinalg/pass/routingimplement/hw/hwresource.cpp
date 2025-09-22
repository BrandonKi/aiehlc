/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/*=============================================================
 *  hw_resource.cpp    (private implementation)
 *===========================================================*/
#include "hw/hwresource.h"
#include <unordered_map>
#include <stdexcept>

namespace detail
{
/*-------------------- generation-specific data --------------*/
template<int GEN> struct GenTraits;

/*  Gen-1  : default colShift 23, rowShift 18  ----------------*/
template<> struct GenTraits<1>
{
    static const std::string& defaultVariant()
    {
        static const std::string def = "XCVE2302";
        return def;
    }

    static const std::map<TileType, std::vector<PortTemplate>>& defaultPortTemplates() {
        static const std::map<TileType, std::vector<PortTemplate>> templates = {
            { TileType::Core, {
                 { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::Shim, {
                 { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::Mem, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::NocShim, {
                 { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            }
        };
        return templates;
    }
    static const std::unordered_map<std::string, AIEDeviceLayout>& table()
    {
        static const std::unordered_map<std::string, AIEDeviceLayout> db = {
            { "XCVE2302",
              { 8, 8, 0x2000'0000'000, 23, 18,
                { {0,0,TileType::Shim},{1,2,TileType::Mem},{3,7,TileType::Core} },
                  {},
                  defaultPortTemplates() } },

            { "XCVE2802",
              { 12, 8, 0x2000'0000'000, 23, 18,
                { {0,1,TileType::Shim},
                  {2,2,TileType::Mem},
                  {3,7,TileType::Core} } ,
                  {},
                  defaultPortTemplates()} }
        };
        return db;
    }
};

/*  Gen-2  : default colShift 25, rowShift 20  ----------------*/
template<> struct GenTraits<2>
{
    static const std::string& defaultVariant()
    {
        static const std::string def = "XCVEk280";
        return def;
    }

    static const std::map<TileType, std::vector<PortTemplate>>& defaultPortTemplates() {
        static const std::map<TileType, std::vector<PortTemplate>> templates = {
            { TileType::Core, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::Shim, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::Mem, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            },
            { TileType::NocShim, {
                { PortDirection::North, PortRole::Master, 4 },
                { PortDirection::South, PortRole::Master, 4 },
                { PortDirection::East, PortRole::Master, 4 },
                { PortDirection::West, PortRole::Master, 4 },
                { PortDirection::DMA, PortRole::Master, 4 },
                { PortDirection::North, PortRole::Slave, 4 },
                { PortDirection::South, PortRole::Slave, 4 },
                { PortDirection::East, PortRole::Slave, 4 },
                { PortDirection::West, PortRole::Slave, 4 },
                { PortDirection::DMA, PortRole::Slave, 4 }
              }
            }
        };
        return templates;
    }
    static const std::unordered_map<std::string, AIEDeviceLayout>& table()
    {
        static const std::unordered_map<std::string, AIEDeviceLayout> db = {
            { "XCVEk280",
              { 11, 38, 0x2000'0000000, 25, 20,
                { 
                  {0,0,TileType::Shim},
                  {1,2,TileType::Mem},
                  {3,11,TileType::Core},
                } ,
                {2, 3, 6, 7, 14, 15, 22, 23, 30, 31, 34, 35},
                defaultPortTemplates()
              } 
            }
        };
        return db;
    }
};

/*-------------------- concrete hidden resource --------------*/
template<int GEN>
class GenResource final : public IHwResource
{
public:
    explicit GenResource(const std::string& var =
                         GenTraits<GEN>::defaultVariant())
    {
        const auto& db = GenTraits<GEN>::table();
        auto it = db.find(var);
        if (it == db.end()) throw std::runtime_error("Unknown variant: "+var);
        layout_  = it->second;
        variant_ = var;
    }

    /* interface impl. ------------------*/
    uint32_t getRows()    const override { return layout_.rows;    }
    uint32_t getColumns() const override { return layout_.columns; }

    uint64_t getBaseAddr() const override { return layout_.baseAddr; }
    void     setBaseAddr(uint64_t a) override { layout_.baseAddr = a; }

    uint64_t tileAddr(uint32_t r,uint32_t c) const override
    { return layout_.tilePhysAddr(r,c); }

    TileType tileType(uint32_t r,uint32_t c) const override
    { return layout_.tileType(r,c); }

    //

    uint32_t absTileRow(TileType type, uint32_t relativeRow) const override
    { return layout_.absTileRow(type,relativeRow);}

    const std::vector<PortTemplate>& getPortsForTileType(TileType type) const override {
        return layout_.getPortsForTileType(type);
    }

    const std::unordered_set<uint32_t>& getShimNoc() const override{
        return layout_.getShimNoc();
    }

    std::string name() const override
    { return "Gen"+std::to_string(GEN)+":"+variant_; }

private:
    AIEDeviceLayout layout_{};
    std::string     variant_;
    std::unordered_map<int,int> shimNoc;
};
} // namespace detail

/*-------------------- public factory ------------------------*/
std::unique_ptr<IHwResource>
makeResource(const std::string& gen, const std::string& var)
{
    using namespace detail;
    if (gen == "Gen1")
        return var.empty()
               ? std::make_unique<GenResource<1>>()
               : std::make_unique<GenResource<1>>(var);

    if (gen == "Gen2")
        return var.empty()
               ? std::make_unique<GenResource<2>>()
               : std::make_unique<GenResource<2>>(var);

    throw std::runtime_error("Unsupported generation: " + gen);
}