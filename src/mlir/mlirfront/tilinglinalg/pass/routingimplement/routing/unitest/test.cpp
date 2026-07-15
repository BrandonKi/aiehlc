/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "routing/routingpath.h"
#include <cassert>
#include <iostream>
//--------------------------------------------------------------------
// Optional demo: compile with  -DROUTING_DEMO_MAIN
//--------------------------------------------------------------------
int main() {
    auto res = makeResource("Gen2");          // default variant
    Point dst = {4, 5};
    // resource manager test
    bool ret = ResourceMgr::init(std::move(res));
    auto rmgr = ResourceMgr::instance();
    auto free_shim = rmgr->freeShimNoc(dst);
    auto dio = rmgr->createDataIO(IOType::Input,free_shim->r, free_shim->c);
    //ret = rmgr->occupyLink(shimPoint, memtile1, dio->id(), portNum);

    // 8×8 grid; vertical wall in column 3
    std::vector<Point> wall ={};//{Point{2,6}};// for (int r = 0; r < 8; ++r) wall.push_back({r,3});
    RoutingPath router(rmgr, dio, wall);
 
    std::vector<Point> allocatedTiles;
    auto result = rmgr->reserveTiles(dio->id(), 8, ReservationStrategy::COLUMN_FIRST, allocatedTiles);

    MultiPath tree;
    bool ok = router.addIOTree({allocatedTiles}, tree);
    std::cout << "Tree success = " << ok << "\n";
    router.dumpGrid();

    // print each branch
    for (size_t i = 0; i < tree.branches.size(); ++i) {
        std::cout << "Branch to (" << tree.dsts[i].r
                  << "," << tree.dsts[i].c << "): ";
        for (auto p : tree.branches[i])
            std::cout << "(" << p.r << "," << p.c << ") ";
        std::cout << "\n";
    }
    return 0;
}