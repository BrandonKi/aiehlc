/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "hw/hwresource.h"
#include "hw/ResourceManager.h"
#include <cassert>
#include <iostream>

int main()
{
    auto res = makeResource("Gen2");          // default variant

    assert(res->getRows()    == 11);
    assert(res->getColumns() == 38);

    // quick address sanity-check (row=1, col=0 → only row shift)
    uint64_t expected = 0x20000000000 + (1 << 20);
    std::cout << res->tileAddr(1, 0)  << std::endl;
    assert(res->tileAddr(1, 0) == expected);

    // resource manager test
    bool ret = ResourceMgr::init(std::move(res));
    auto rmgr = ResourceMgr::instance();
    std::optional<Point> dst(Point{3, 20});
    auto free_shim = rmgr->freeShimNoc(dst);
    if (free_shim) {
        std::cout << "free col is " << free_shim->c <<std::endl;
    } else {
        std::cout << "no free shim" << std::endl;
    }
    //test with tile loc with the tile type info
    for (int i = 0; i < 3; i++) {
        Point d2 = Point{0 + i, 20};
        std::optional<TypeBasedTileLoc> loc(TypeBasedTileLoc{TileType::Core, d2});
        std::cout << "tile type is  TileType::Core , tile relative row is " << d2.r <<std::endl;
        auto f2 = rmgr->freeShimNoc(loc);
        if (f2) {
            std::cout << "From the said TypeBasedTileLoc get free col is " << f2->c <<std::endl;
        } else {
            std::cout << "no free shim" << std::endl;
        }
    }

    Point shimPoint = *free_shim, memtile1 = {1, free_shim->c};
    int portNum;
    ret = rmgr->linkAvailable(shimPoint, memtile1, portNum);
    std::cout << " shim (0," << free_shim->c << ")" << " to (1, " << memtile1.c << " ) link available is " << ret;
    std::cout << " portnum is " << portNum << std::endl; 
    if (ret) {
        std::cout << "no available link " <<std::endl;
    }
    auto dio = rmgr->createDataIO(IOType::Input,free_shim->r,free_shim->c);
    PortDirection pda=PortDirection::North, pdb=PortDirection::South;
    ret = rmgr->occupyLink(shimPoint, memtile1, dio->id(),portNum, pda,pdb);
    std::cout << " occupy " << ret << std::endl;
    ret = rmgr->linkAvailable(shimPoint, memtile1, portNum);
    std::cout << " shim (0," << free_shim->c << ")" << " to (1, " << memtile1.c << " ) link available is " << ret;
    std::cout << " portnum is " << portNum << std::endl; 
    assert(portNum != 0);
    
    std::vector<Point> allocatedTiles;
    for (int i = 0; i < 3; i ++) {
        auto result = rmgr->reserveTiles(dio->id(), 28, ReservationStrategy::ROW_FIRST, allocatedTiles);
        std::cout << "Reserved 3 tiles for DataIO1 (column-first): " << (result ? "Success" : "Failed") << std::endl;
        
        if (result) {
            std::cout << "Allocated tiles: " << i << " rounds : ";
            for (const auto& p : allocatedTiles) {
                std::cout << "(" << p.r << ", " << p.c << ") ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << (int)PortDirection::North << "is " << PortDirectiontoString(PortDirection::North) <<std::endl;
    std::cout << "All HW-resource sanity checks passed.\n";
    return 0;
}
