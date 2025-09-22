/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "routing/routingtopology.h"

RoutingTopology::RoutingTopology(std::string gen, std::string name)
    : gen_(gen), name_(name),
      rm_(std::make_shared<ResourceMgr>(makeResource(gen, name)))
{}

// ── createDataIO ────────────────────────────────────────────────────────
std::shared_ptr<DataIO>
RoutingTopology::_createDataIO(std::string dioName, std::optional<Point> shim, DMADIRECTION direction, int channel)
{
    if (!shim) {
        throw std::runtime_error("No free shim tile for `" + std::string(dioName) + "`");
        return nullptr;
    }
    auto dio      = rm_->createDataIO(IOType::Input,shim->r, shim->c, direction, channel, dioName);
    dataios_.emplace(dio->id(), dio);
    return dio;
}

std::shared_ptr<DataIO>
RoutingTopology::createDataIO(std::string dioName, std::optional<TypeBasedTileLoc> loc,DMADIRECTION direct)
{
    //std::optional<FoundDmaSlot> ResourceMgr::freeShimNoc(std::optional<TypeBasedTileLoc> ioPaireddstTileloc,
    //                                                 DMADIRECTION direct,
    //                                                 int requesterIoId)

    /*
    struct FoundDmaSlot {
    DMADIRECTION direct;
    Point loc;
    int channel;
    */
    auto dioId = rm_->allocdioid();
    std::optional<FoundDmaSlot> foundDmaSlot = rm_->freeShimNoc(loc, direct, dioId);// optional<TileCoord>
    auto shimpoint = std::make_optional<Point>(foundDmaSlot->loc);
    return _createDataIO(dioName, shimpoint, foundDmaSlot->direct, foundDmaSlot->channel);;
}

std::shared_ptr<DataIO>
RoutingTopology::createDataIO(std::string dioName)
{
    auto shim = rm_->freeShimNoc();         // optional<TileCoord>
    return _createDataIO(dioName,shim);
}

std::vector<Point> RoutingTopology::ReserveTiles(int nums,int dioID) {
    std::vector<Point> allocatedTiles;
    auto result = rm_->reserveTiles(dioID, 8, ReservationStrategy::COLUMN_FIRST, allocatedTiles);
    return allocatedTiles;
}
bool RoutingTopology::occupyLink(Point a, Point b, const int ioId,int& portNum, PortDirection& pda, PortDirection& pdb){
    return rm_->occupyLink(a,  b, ioId,portNum, pda, pdb);
}
// ── createPath ──────────────────────────────────────────────────────────
std::optional<std::shared_ptr<const RoutingPath>> 
RoutingTopology::createPath(int dioID, std::vector<Point> dsttiles) {
    std::vector<Point> wall ={};
    if (dataios_.find(dioID) == dataios_.end()) {
        return std::nullopt;
    }
    auto dio = dataios_[dioID];
    std::shared_ptr<RoutingPath> rpath = std::make_shared<RoutingPath>(rm_, dio, wall);
    bool ok = rpath->addIOTree({dsttiles});
    paths_.push_back(rpath);
    return rpath;
    /*
    if (!dio) return nullptr;

    auto region = rm_->allocateTileRegion(cols, rows);    // optional<vector<TileCoord>>
    if (!region) return nullptr;

    auto path = std::make_shared<RoutingPath>(dio->shimTile, *region);
    if (!path->connect()) {                               // hypothetical
        rm_->releaseTileRegion(*region);
        return nullptr;
    }

    paths_.push_back(path);
    return path;          // now freely shareable
    */
}
