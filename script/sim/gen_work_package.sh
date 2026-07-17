#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT

set -e

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK_DIR="${SIM_DIR}/Work"
PS_SO="${SIM_DIR}/build/aiehlc_ps.so"
AIE_ARCH="aie-ml"
SHIM_COL=3
KERNEL_COL=4
KERNEL_ROW=1
STUB_ELF="${SIM_DIR}/build/stub_kernel_build/stub_kernel"
STUB_ALL=0
STUB_TILES=""
NOC_ALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --work-dir)       WORK_DIR="$2";       shift 2 ;;
        --ps-so)          PS_SO="$2";           shift 2 ;;
        --aie-arch)       AIE_ARCH="$2";        shift 2 ;;
        --shim-col)       SHIM_COL="$2";        shift 2 ;;
        --kernel-col)     KERNEL_COL="$2";      shift 2 ;;
        --kernel-row)  KERNEL_ROW="$2";   shift 2 ;;
        --stub-elf)       STUB_ELF="$2";        shift 2 ;;
        --stub-all)       STUB_ALL=1;           shift ;;
        --stub-tiles)     STUB_TILES="$2";      shift 2 ;;
        --noc-all)        NOC_ALL=1;            shift ;;
        *) echo "[gen_work_package] Unknown option: $1"; exit 1 ;;
    esac
done

case "$AIE_ARCH" in
    aie)
        DEVICE_JSON_DIR="data/devices"
        DEVICE_JSON_FILE="VC1902.json"
        PHY_DEVICE="xcvc1902-vsva2197-2MP-e-S"
        AIE_FREQ=1250000000.0
        NOC_NAME_PREFIX="AIE_NOC"
        TILE_ROW_START=1
        ;;
    aie-ml)
        DEVICE_JSON_DIR="data/aie_ml/devices"
        DEVICE_JSON_FILE="VC2802.json"
        PHY_DEVICE="xcve2802-nsvh1369-1LP-e-S"
        AIE_FREQ=1000000000.0
        NOC_NAME_PREFIX="AIE_ML_NOC"
        TILE_ROW_START=3
        ;;
    aie2ps)
        DEVICE_JSON_DIR="data/aie2ps/devices"
        DEVICE_JSON_FILE="XC2VE3858.json"
        PHY_DEVICE="xc2ve3858-ssva2112-1LP-e-S"
        AIE_FREQ=1000000000.0
        NOC_NAME_PREFIX="AIE2PS_NOC"
        TILE_ROW_START=3
        ;;
    *)
        echo "[gen_work_package] Unknown AIE_ARCH: $AIE_ARCH"; exit 1 ;;
esac

KERNEL_ABS_ROW=$(( TILE_ROW_START + KERNEL_ROW - 1 ))

TILE_NAME="${KERNEL_COL}_${KERNEL_ROW}"

echo "[gen_work_package] Work/ → ${WORK_DIR}"
echo "  AIE arch:      ${AIE_ARCH}"
echo "  Shim col:      ${SHIM_COL} (NOC/DMA)"
echo "  Kernel tile:   col=${KERNEL_COL} row=${KERNEL_ROW} → abs_row=${KERNEL_ABS_ROW} dir=${TILE_NAME}"
echo "  Stub ELF:      ${STUB_ELF}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}/config"
mkdir -p "${WORK_DIR}/arch"
mkdir -p "${WORK_DIR}/ps/c_rts/systemC/generated-objects"
mkdir -p "${WORK_DIR}/reports"

echo -n "hw" > "${WORK_DIR}/.target"

PS_SO_ABS="$(cd "$(dirname "$PS_SO")" && pwd)/$(basename "$PS_SO")"
PS_SO_WORK_REL="ps/c_rts/systemC/generated-objects/aiehlc_ps.so"
if [ -f "${PS_SO_ABS}" ]; then
    ln -sf "${PS_SO_ABS}" "${WORK_DIR}/${PS_SO_WORK_REL}"
else
    echo "[gen_work_package] WARNING: PS .so not found: ${PS_SO_ABS}"
fi

cat > "${WORK_DIR}/config/scsim_config.json" <<EOF
{
    "SimulationConfig": {
        "device_json": {
            "directory": "${DEVICE_JSON_DIR}",
            "file": "${DEVICE_JSON_FILE}"
        },
        "phy_device_file": "${PHY_DEVICE}",
        "aiearch": "${AIE_ARCH}",
        "aie_freq": ${AIE_FREQ},
        "use_real_noc": 1,
        "evaluate_fifo_depth": 0,
        "shim_sol": "arch/aieshim_solution.aiesol",
        "xpe_report": "reports/aiehlc.xpe",
        "pl_ip_block": [
            {
                "name": "ps_aiehlc_main",
                "ip": "ps",
                "lib_path": "${PS_SO_WORK_REL}",
                "pl_freq": 250000000.0,
                "axi_mm": [
                    {
                        "port_name": "ps_axi",
                        "direction": "ps_to_gm",
                        "bus_width": 0
                    }
                ],
                "event_bus": []
            }
        ]
    }
}
EOF

{
    PLACEMENTS=""
    PORT_IDX=0
    if [ "$NOC_ALL" -eq 1 ]; then
        # Determine column count from arch
        case "$AIE_ARCH" in
            aie)    NOC_NUM_COLS=50 ;;
            *)      NOC_NUM_COLS=38 ;;
        esac
        echo "  NOC: declaring all shim NOC ports (${NOC_NUM_COLS} cols) …"
        for col in $(seq 0 2 $(( NOC_NUM_COLS - 1 ))); do
            PHYS_IDX=$(( col / 2 ))
            PORT_NAME=$(printf "M%02dAXI" "$PORT_IDX" | sed 's/AXI/_AXI/')
            SEP=","
            [ "$PORT_IDX" -eq 0 ] && SEP=""
            PLACEMENTS+="${SEP}
    {
      \"LogicalInstance\": {\"InstanceName\": \"ai_engine_0\", \"PortName\": \"${PORT_NAME}\"},
      \"PhysicalInstance\": [{\"name\": \"${NOC_NAME_PREFIX}_X${PHYS_IDX}Y0_${NOC_NAME_PREFIX}_M_AXI\",
                               \"column\": ${col}, \"channel\": 0}],
      \"IsSoft\": true
    }"
            PORT_IDX=$(( PORT_IDX + 1 ))
        done
    else
        _noc_all_cols=($SHIM_COL)
        if [ "$AIE_ARCH" != "aie-ml" ] && [ -n "$STUB_TILES" ]; then
            declare -A _noc_cols_seen
            _noc_cols_seen[$SHIM_COL]=1
            IFS=',' read -ra _stub_list <<< "$STUB_TILES"
            for _ts in "${_stub_list[@]}"; do
                _c="${_ts%%:*}"
                if [ -z "${_noc_cols_seen[$_c]:-}" ]; then
                    _noc_cols_seen[$_c]=1
                    _noc_all_cols+=($_c)
                fi
            done
            unset _noc_cols_seen
        fi
        PLACEMENTS=""
        _port_idx=0
        for _col in "${_noc_all_cols[@]}"; do
            NOC_PHYS_IDX=$(( _col / 2 ))
            PORT_NAME=$(printf "M%02d_AXI" "$_port_idx")
            SEP=""
            [ $_port_idx -gt 0 ] && SEP=","
            PLACEMENTS+="${SEP}
    {
      \"LogicalInstance\": {\"InstanceName\": \"ai_engine_0\", \"PortName\": \"${PORT_NAME}\"},
      \"PhysicalInstance\": [{\"name\": \"${NOC_NAME_PREFIX}_X${NOC_PHYS_IDX}Y0_${NOC_NAME_PREFIX}_M_AXI\",
                               \"column\": ${_col}, \"channel\": 0}],
      \"IsSoft\": true
    }"
            _port_idx=$(( _port_idx + 1 ))
        done
    fi

    cat > "${WORK_DIR}/arch/aieshim_solution.aiesol" <<EOF
{
  "Placement": [${PLACEMENTS}
  ]
}
EOF
}

cat > "${WORK_DIR}/arch/aie_partition.json" <<'EOF'
{
  "AIE": {
    "ai_engine_0": {
      "startColumn": 0,
      "numColumns": 38,
      "partitions": [
        {
          "startColumn": 0,
          "numColumns": 38,
          "uuid": "0xaiehlc000000000000000000000001",
          "aie_pl_intf_id": "0x00000001",
          "namespace": ""
        }
      ]
    }
  }
}
EOF

cat > "${WORK_DIR}/arch/cfgraph.xml" <<'EOF'
<cf:model cf:cpu="cortex_a72" cf:name="root" cf:partition="0" cf:prefix="_p0_"
          xd:type="design"
          xmlns:cf="http_//www_xilinx_com/connections"
          xmlns:xd="http_//www_xilinx_com/xd">
  <cf:block cf:name="aieip">
    <cf:port cf:direction="out" cf:name="gmio_axi" cf:portType="aximm"/>
  </cf:block>
  <cf:comp cf:name="ai_engine_0" xd:componentRef="ai_engine"/>
  <cf:instance cf:blockName="aieip" cf:compName="ai_engine_0" cf:name="ai_engine_0">
    <cf:portMap cf:blockPort="gmio_axi" cf:compPort="M00_AXI"
                cf:paddedWidth="0" xd:isRegistered="false"
                xd:readBandwidth="1" xd:writeBandwidth="1"/>
  </cf:instance>
</cf:model>
EOF

STUB_BUILD_DIR="$(dirname "$STUB_ELF")"
mkdir -p "${WORK_DIR}/aie"

STUB_ELF_ABS="$(cd "$(dirname "$STUB_ELF")" 2>/dev/null && pwd)/$(basename "$STUB_ELF")"

install_stub() {
    local tname="$1"
    local tdir="${WORK_DIR}/aie/${tname}/Release"
    mkdir -p "$tdir"
    if [ -f "${STUB_ELF_ABS}" ]; then
        ln -sf "${STUB_ELF_ABS}" "${tdir}/${tname}"
        for ext in '#' '##' sdr srv; do
            local src="${STUB_BUILD_DIR}/stub_kernel.${ext}"
            [ -f "$src" ] && ln -sf "$(realpath "$src")" "${tdir}/${tname}.${ext}" || true
        done
    fi
}

if [ -n "$STUB_TILES" ] && [ "$STUB_ALL" -eq 0 ]; then
    echo "  Stubbing tiles: ${STUB_TILES}"
    ACTIVE_CORES_LIST=""
    ACTIVE_MEM_ENTRIES=""
    _mem_cols_done=",$SHIM_COL," 
    if [ "$TILE_ROW_START" -gt 1 ]; then
        for mr in $(seq 1 $(( TILE_ROW_START - 1 ))); do
            ACTIVE_MEM_ENTRIES+="\"${SHIM_COL}_${mr}\", "
        done
    fi
    IFS=',' read -ra TILE_LIST <<< "$STUB_TILES"
    for tile_spec in "${TILE_LIST[@]}"; do
        col="${tile_spec%%:*}"
        row="${tile_spec##*:}"
        tname="${col}_${row}"
        install_stub "$tname"
        ACTIVE_CORES_LIST+="    {\"${tname}\": \"${WORK_DIR}/aie/${tname}\"},\n"
        if [[ "$_mem_cols_done" != *",$col,"* ]]; then
            _mem_cols_done+="$col,"
            if [ "$TILE_ROW_START" -gt 1 ]; then
                for mr in $(seq 1 $(( TILE_ROW_START - 1 ))); do
                    ACTIVE_MEM_ENTRIES+="\"${col}_${mr}\", "
                done
            fi
        fi
    done
    ACTIVE_CORES_LIST="${ACTIVE_CORES_LIST%,\\n}"
    ACTIVE_MEM_ENTRIES="${ACTIVE_MEM_ENTRIES%, }"
    printf '{\n  "ActiveCores": [\n%b\n  ],\n  "ActiveMemory": [%s]\n}\n' \
        "$ACTIVE_CORES_LIST" "$ACTIVE_MEM_ENTRIES" \
        > "${WORK_DIR}/aie/active_cores.json"

elif [ "$STUB_ALL" -eq 1 ]; then
    case "$AIE_ARCH" in
        aie)    NUM_ROWS=8;  NUM_COLS=50 ;;
        *)      NUM_ROWS=8;  NUM_COLS=38 ;;
    esac

    if [ ! -f "${STUB_ELF}" ]; then
        echo "[gen_work_package] ERROR: --stub-all requires stub ELF: ${STUB_ELF}"
        echo "  Run: make -C script/sim AIEHLC_HOST_SRC=... AIE_ARCH=..."
        exit 1
    fi

    echo "  Stubbing all ${NUM_COLS}x${NUM_ROWS} core tiles..."
    ACTIVE_CORES_LIST=""
    for col in $(seq 0 $(( NUM_COLS - 1 ))); do
        for row in $(seq 1 "${NUM_ROWS}"); do
            tname="${col}_${row}"
            install_stub "$tname"
            ACTIVE_CORES_LIST+="    {\"${tname}\": \"${WORK_DIR}/aie/${tname}\"},\n"
        done
    done
    ACTIVE_CORES_LIST="${ACTIVE_CORES_LIST%,\\n}"  # strip trailing comma

    ACTIVE_MEM_LIST=""
    for col in $(seq 0 $(( NUM_COLS - 1 ))); do
        for mr in $(seq 1 $(( TILE_ROW_START - 1 ))); do
            ACTIVE_MEM_LIST+="\"${col}_${mr}\", "
        done
    done
    ACTIVE_MEM_LIST="${ACTIVE_MEM_LIST%, }"

    printf '{\n  "ActiveCores": [\n%b\n  ],\n  "ActiveMemory": [%s]\n}\n' \
        "$ACTIVE_CORES_LIST" "$ACTIVE_MEM_LIST" \
        > "${WORK_DIR}/aie/active_cores.json"

    echo "  All tiles activated — load any kernel at runtime via XAie_LoadElfMem"

else
    install_stub "${TILE_NAME}"
    if [ -f "${STUB_ELF}" ]; then
        echo "  Tile ${TILE_NAME}: stub ELF installed"
    else
        echo "[gen_work_package] WARNING: stub ELF not found: ${STUB_ELF}"
        echo "  Kernel tile will be activated via XAie_LoadElfMem only (cycle-approx mode)"
    fi

    ACTIVE_MEM_ENTRIES=""
    if [ "$TILE_ROW_START" -gt 1 ]; then
        for mr in $(seq 1 $(( TILE_ROW_START - 1 ))); do
            ACTIVE_MEM_ENTRIES+="\"${SHIM_COL}_${mr}\", \"${KERNEL_COL}_${mr}\", "
        done
        ACTIVE_MEM_ENTRIES="${ACTIVE_MEM_ENTRIES%, }"
    fi

    cat > "${WORK_DIR}/aie/active_cores.json" <<EOF
{
  "ActiveCores": [
    {"${TILE_NAME}": "${WORK_DIR}/aie/${TILE_NAME}"}
  ],
  "ActiveMemory": [${ACTIVE_MEM_ENTRIES}]
}
EOF
fi

echo "[gen_work_package] Done → ${WORK_DIR}"
