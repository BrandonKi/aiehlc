CLEAN_BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
( \
    clear 2>/dev/null; \
    rm -rf ${CLEAN_BUILD_DIR}/aout/ && \
    rm -f ${CLEAN_BUILD_DIR}/build/main.elf && \
    rm -rf ${CLEAN_BUILD_DIR}/script/sim/build/kernel_wrapper/ \
           ${CLEAN_BUILD_DIR}/script/sim/build/kernel2_wrapper/ \
           ${CLEAN_BUILD_DIR}/script/sim/build/aiehlc_ps.so \
           ${CLEAN_BUILD_DIR}/script/sim/build/kernel_elf_init.{cc,o} \
           ${CLEAN_BUILD_DIR}/script/sim/build/aie_rt_objs/ && \
    make -C ${CLEAN_BUILD_DIR}/build -j$(nproc) && \
    source ${CLEAN_BUILD_DIR}/script/setup.sh --enable-llvmaie --bsp-use-git-repo=https://gitenterprise.xilinx.com/ai-engine/aie-rt && \
    # make -C ${CLEAN_BUILD_DIR}/build install && \

    # python3 ${CLEAN_BUILD_DIR}/script/aiehlc_triton.py \
    #     ${CLEAN_BUILD_DIR}/example/tileprogram/design/triton/triton_matmul_1x1.py \
    #     --output-dir ${CLEAN_BUILD_DIR}/worklocal && \

    # python3 ${CLEAN_BUILD_DIR}/script/aiehlc_triton.py \
    #     ${CLEAN_BUILD_DIR}/example/tileprogram/design/triton/triton_matmul.py \
    #     --output-dir ${CLEAN_BUILD_DIR}/worklocal && \

    # source aiehlc.sh --use-llvm-aie --platform sim --aie-version 2 --runtime-source-file ./tutorial/example.cpp && \
    # source aiehlc.sh --use-llvm-aie --platform baremetal --aie-version 2 --runtime-source-file ./tutorial/example.cpp && \
    # source aiehlc.sh --use-llvm-aie --platform baremetal --aie-version 2 --runtime-source-file ./example/nokernelload/small.cpp && \
    # source ${CLEAN_BUILD_DIR}/script/aiehlc.sh --platform baremetal --aie-version 2 --runtime-source-file ${CLEAN_BUILD_DIR}/example/tileprogram/design/ccode/proposal3.cc && \
    # source aiehlc.sh --use-llvm-aie --platform baremetal --aie-version 2 --runtime-source-file ./example/perf/perf.cpp && \
    # source ${CLEAN_BUILD_DIR}/script/aiehlc.sh --aie-version 2 --runtime-source-file ${CLEAN_BUILD_DIR}/example/multi/memtile_test.cpp && \
    # cp ./thirdparty/alib/aie-rt/driver/src/libxaiengine.so.3 /home/$USER/vek280/new_libxaiengine.so.3 && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:4 --runtime-source-file ./example/tileprogram/ccode/simple.cc && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:4 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles "0:3,0:4,0:5,0:6,1:3,1:4,1:5,1:6,2:3,2:4,2:5,2:6,3:3,3:4,3:5,3:6" --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc && \
    source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:2 --runtime-source-file ./tutorial/example.cpp && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 0:3 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc && \
    # source script/aiehlc.sh --platform baremetal --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc && \
    # 2x2 mesh, partition {3,4,0,5}: compute tiles at abs rows 3-4, cols 3-4
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 3:3,4:3,3:4,4:4 --runtime-source-file ./example/tileprogram/ccode/simplematmul2_sim.cc && \
    # source script/aiehlc.sh --platform sim --aie-version 2 --sim-tiles 4:4 --runtime-source-file ./example/perf/aieml_perf.cc && \
    # source script/aiehlc.sh --platform baremetal --aie-version 2 --runtime-source-file ./example/perf/aieml_perf.cc && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:4 --runtime-source-file ./example/perf/aieml_perf.cc && \
    # source script/aiehlc.sh --platform baremetal --aie-version 5 --runtime-source-file ./example/perf/aieml_perf.cc && \
    # source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:2 --runtime-source-file ./tutorial/example.cpp && \
    # source script/aiehlc.sh --platform baremetal --aie-version 5 --runtime-source-file ./tutorial/example.cpp && \
    # source script/aiehlc.sh --platform sim --aie-version 2 --sim-tiles 4:4 --runtime-source-file ./example/multi/multi_kernel.cc
    # source script/aiehlc.sh --platform baremetal --aie-version 2 --runtime-source-file ./example/multi/multi_kernel.cc && \
    cp ${CLEAN_BUILD_DIR}/aout/main.elf /home/$USER/vek280/mk.elf
) 2>&1 | tee build_log.txt

# python3 ./script/test/appvek385.py -y  -nonreboot > ./applogvek 2>&1


