/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_TILE__
#define __ROUTING_TILE__
enum RoutingLoc{
    uint8_t START,
    uint8_t MIDDLE,
    uint8_t END,
    uint8_t STARTANDMIDDLE,
    uint8_t MIDDLEANDEND
};
enum TileType {
    uint8_t SHIM,
    uint8_t MEM,
    uint8_t CORE
};
class RoutingTile {
public:
    RoutingTile();
private:
    TileType tileType;
    RoutingLoc locType;
    uint16_t row;
    uint16_t col;
};
#endif //__ROUTING_TILE__