// Override runtime debug level (from #pragma aie_debug_level)
int g_runtime_debug_level = 97;

#include "aie_runtime.h"
#include "aie_runtime_debug.h"
void host_canonicalized(XAie_DevInst* v1, void* v2, void* v3, void* v4) {
  int32_t v5 = 16;
  XAie_DevInst* dev = v1;
  void* v6 = __runtime_buffer_offset(v3, 0);
  XAie_LocType v7 = XAie_TileLoc(0, 0);
  XAie_LocType v8 = XAie_TileLoc(0, 3);
  /* Allocated BD ID 0 for tile (0,3) */
  XAie_LocType v9 = XAie_TileLoc(0, 4);
  /* Allocated BD ID 0 for tile (0,4) */
  /* Allocated BD ID 0 for tile (0,0) */
  void* v10 = __runtime_buffer_offset(v3, 512);
  XAie_LocType v11 = XAie_TileLoc(1, 0);
  XAie_LocType v12 = XAie_TileLoc(1, 3);
  void* v13 = __runtime_buffer_offset(v10, 512);
  /* Allocated BD ID 0 for tile (1,3) */
  XAie_LocType v14 = XAie_TileLoc(1, 4);
  void* v15 = __runtime_buffer_offset(v10, 512);
  /* Allocated BD ID 0 for tile (1,4) */
  /* Allocated BD ID 0 for tile (1,0) */
  void* v16 = __runtime_buffer_offset(v2, 0);
  /* Allocated BD ID 1 for tile (0,3) */
  /* Allocated BD ID 1 for tile (1,3) */
  /* Allocated BD ID 1 for tile (0,0) */
  void* v17 = __runtime_buffer_offset(v4, 0);
  void* v18 = __runtime_buffer_offset(v17, 0);
  /* Allocated BD ID 2 for tile (0,3) */
  void* v19 = __runtime_buffer_offset(v17, 256);
  /* Allocated BD ID 2 for tile (1,3) */
  /* Allocated BD ID 1 for tile (1,0) */
  void* v20 = __runtime_buffer_offset(v2, 512);
  void* v21 = __runtime_buffer_offset(v20, 512);
  /* Allocated BD ID 1 for tile (0,4) */
  void* v22 = __runtime_buffer_offset(v20, 512);
  /* Allocated BD ID 1 for tile (1,4) */
  /* Allocated BD ID 2 for tile (1,0) */
  void* v23 = __runtime_buffer_offset(v4, 512);
  void* v24 = __runtime_buffer_offset(v23, 0);
  /* Allocated BD ID 2 for tile (0,4) */
  void* v25 = __runtime_buffer_offset(v23, 256);
  /* Allocated BD ID 2 for tile (1,4) */
  /* Allocated BD ID 3 for tile (1,0) */
  /* Load Kernel Group: 4 tile(s) */
  kernel_group v26 = __Runtime_load_kernel_group_4t(v1, v8, v9, v12, v14, 4);
  /* Launch Kernel Group */
  event v27 = __Runtime_launch_kernel_group(v1, v26);
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */
  void* v28 = __runtime_buffer_arg(v6);
  XAie_DmaDesc v29 = __Runtime_dma_bd_config(v1, v7, v28, 0, 512, -1, 0, 0, 0, 0, 0, 0, -1);
  /* Create IO: channel_id=0, bd_id=0, tile=(0,0), direction=MM2S */
  io v30 = __Runtime_dma_createio_4(v7, v29, 0, 0, DMA_MM2S);
  /* DMA BD Config: bd_id=1, len=512, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v31 = __runtime_buffer_arg((void*)33280);
  XAie_DmaDesc v32 = __Runtime_dma_bd_config(v1, v8, v31, 1, 512, 0, 0, 0, 2, -1, 3, 1, -1);
  /* Lock init: tile(0,3) lock=2 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v33 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v34 = __Runtime_dma_bd_config(v1, v8, v33, 0, 512, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */
  io v35 = __Runtime_dma_createio_4(v8, v34, 1, 0, DMA_S2MM);
  /* DMA BD Config: bd_id=1, len=512, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v36 = __runtime_buffer_arg((void*)33280);
  XAie_DmaDesc v37 = __Runtime_dma_bd_config(v1, v9, v36, 1, 512, 0, 0, 0, 2, -1, 3, 1, -1);
  /* Lock init: tile(0,4) lock=2 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v38 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v39 = __Runtime_dma_bd_config(v1, v9, v38, 0, 512, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */
  io v40 = __Runtime_dma_createio_4(v9, v39, 1, 0, DMA_S2MM);
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */
  void* v41 = __runtime_buffer_arg(v10);
  XAie_DmaDesc v42 = __Runtime_dma_bd_config(v1, v11, v41, 0, 512, -1, 0, 0, 0, 0, 0, 0, -1);
  /* Create IO: channel_id=0, bd_id=0, tile=(1,0), direction=MM2S */
  io v43 = __Runtime_dma_createio_4(v11, v42, 0, 0, DMA_MM2S);
  /* DMA BD Config: bd_id=1, len=512, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v44 = __runtime_buffer_arg((void*)33280);
  XAie_DmaDesc v45 = __Runtime_dma_bd_config(v1, v12, v44, 1, 512, 0, 0, 0, 2, -1, 3, 1, -1);
  /* Lock init: tile(1,3) lock=2 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v46 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v47 = __Runtime_dma_bd_config(v1, v12, v46, 0, 512, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */
  io v48 = __Runtime_dma_createio_4(v12, v47, 1, 0, DMA_S2MM);
  /* DMA BD Config: bd_id=1, len=512, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v49 = __runtime_buffer_arg((void*)33280);
  XAie_DmaDesc v50 = __Runtime_dma_bd_config(v1, v14, v49, 1, 512, 0, 0, 0, 2, -1, 3, 1, -1);
  /* Lock init: tile(1,4) lock=2 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));
  /* DMA BD Config: bd_id=0, len=512, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */
  void* v51 = __runtime_buffer_arg((void*)32768);
  XAie_DmaDesc v52 = __Runtime_dma_bd_config(v1, v14, v51, 0, 512, 1, 0, 0, 2, -1, 3, 1, -1);
  /* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */
  io v53 = __Runtime_dma_createio_4(v14, v52, 1, 0, DMA_S2MM);
  /* DMA BD Config: bd_id=1, len=512, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */
  void* v54 = __runtime_buffer_arg(v16);
  XAie_DmaDesc v55 = __Runtime_dma_bd_config(v1, v7, v54, 1, 512, -1, 0, 0, 0, 0, 0, 0, -1);
  /* Create IO: channel_id=1, bd_id=1, tile=(0,0), direction=MM2S */
  io v56 = __Runtime_dma_createio_4(v7, v55, 1, 1, DMA_MM2S);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v57 = __runtime_buffer_arg((void*)34304);
  XAie_DmaDesc v58 = __Runtime_dma_bd_config(v1, v8, v57, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* Lock init: tile(0,3) lock=0 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v59 = __runtime_buffer_arg((void*)33792);
  XAie_DmaDesc v60 = __Runtime_dma_bd_config(v1, v8, v59, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */
  io v61 = __Runtime_dma_createio_4(v8, v60, 0, 2, DMA_S2MM);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v62 = __runtime_buffer_arg((void*)34304);
  XAie_DmaDesc v63 = __Runtime_dma_bd_config(v1, v12, v62, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* Lock init: tile(1,3) lock=0 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v64 = __runtime_buffer_arg((void*)33792);
  XAie_DmaDesc v65 = __Runtime_dma_bd_config(v1, v12, v64, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */
  io v66 = __Runtime_dma_createio_4(v12, v65, 0, 2, DMA_S2MM);
  /* DMA BD Config: bd_id=3, len=256, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */
  void* v67 = __runtime_buffer_arg(v17);
  int64_t v68 = (int64_t) v5;
  void* v69 = __runtime_buffer_offset(v67, v68);
  XAie_DmaDesc v70 = __Runtime_dma_bd_config_multidim(v1, v11, v69, 3, 256, -1, 0, 2, -1, 0, -1, 0, -1, 2, 4, 4, 32, 16, 0, 0, 0, 0);
  /* DMA BD Config: bd_id=2, len=256, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */
  void* v71 = __runtime_buffer_arg(v17);
  XAie_DmaDesc v72 = __Runtime_dma_bd_config_multidim(v1, v11, v71, 2, 256, -1, 0, 1, -1, 0, -1, 0, -1, 2, 4, 4, 32, 16, 0, 0, 0, 0);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,0), direction=S2MM */
  /* Enable out-of-order BD on tile(1,0) ch=0 dir=S2MM */
  __Runtime_dma_channel_enable_ooo(v1, v11, 0, DMA_S2MM);
  io v73 = __Runtime_dma_createio_4(v11, v72, 0, 2, DMA_S2MM);
  /* DMA BD Config: bd_id=5, len=256, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v74 = __runtime_buffer_arg((void*)35072);
  XAie_DmaDesc v75 = __Runtime_dma_bd_config(v1, v8, v74, 5, 256, 4, 1, 1, 5, -1, 4, 1, 2);
  /* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));
  /* DMA BD Config: bd_id=4, len=256, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=2 */
  void* v76 = __runtime_buffer_arg((void*)34816);
  XAie_DmaDesc v77 = __Runtime_dma_bd_config(v1, v8, v76, 4, 256, 5, 1, 1, 5, -1, 4, 1, 2);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */
  io v78 = __Runtime_dma_createio_4(v8, v77, 0, 4, DMA_MM2S);
  /* DMA BD Config: bd_id=5, len=256, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v79 = __runtime_buffer_arg((void*)35072);
  XAie_DmaDesc v80 = __Runtime_dma_bd_config(v1, v12, v79, 5, 256, 4, 1, 2, 5, -1, 4, 1, 3);
  /* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));
  /* DMA BD Config: bd_id=4, len=256, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */
  void* v81 = __runtime_buffer_arg((void*)34816);
  XAie_DmaDesc v82 = __Runtime_dma_bd_config(v1, v12, v81, 4, 256, 5, 1, 2, 5, -1, 4, 1, 3);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */
  io v83 = __Runtime_dma_createio_4(v12, v82, 0, 4, DMA_MM2S);
  /* DMA BD Config: bd_id=4, len=512, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */
  void* v84 = __runtime_buffer_arg(v20);
  XAie_DmaDesc v85 = __Runtime_dma_bd_config(v1, v11, v84, 4, 512, -1, 0, 0, 0, 0, 0, 0, -1);
  /* Create IO: channel_id=1, bd_id=4, tile=(1,0), direction=MM2S */
  io v86 = __Runtime_dma_createio_4(v11, v85, 1, 4, DMA_MM2S);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v87 = __runtime_buffer_arg((void*)34304);
  XAie_DmaDesc v88 = __Runtime_dma_bd_config(v1, v9, v87, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* Lock init: tile(0,4) lock=0 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v89 = __runtime_buffer_arg((void*)33792);
  XAie_DmaDesc v90 = __Runtime_dma_bd_config(v1, v9, v89, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */
  io v91 = __Runtime_dma_createio_4(v9, v90, 0, 2, DMA_S2MM);
  /* DMA BD Config: bd_id=3, len=512, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v92 = __runtime_buffer_arg((void*)34304);
  XAie_DmaDesc v93 = __Runtime_dma_bd_config(v1, v14, v92, 3, 512, 2, 0, 0, 0, -1, 1, 1, -1);
  /* Lock init: tile(1,4) lock=0 init_value=2 */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));
  /* DMA BD Config: bd_id=2, len=512, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */
  void* v94 = __runtime_buffer_arg((void*)33792);
  XAie_DmaDesc v95 = __Runtime_dma_bd_config(v1, v14, v94, 2, 512, 3, 0, 0, 0, -1, 1, 1, -1);
  /* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */
  io v96 = __Runtime_dma_createio_4(v14, v95, 0, 2, DMA_S2MM);
  /* DMA BD Config: bd_id=7, len=256, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */
  void* v97 = __runtime_buffer_arg(v23);
  int64_t v98 = (int64_t) v5;
  void* v99 = __runtime_buffer_offset(v97, v98);
  XAie_DmaDesc v100 = __Runtime_dma_bd_config_multidim(v1, v11, v99, 7, 256, -1, 0, 4, -1, 0, -1, 0, -1, 2, 4, 4, 32, 16, 0, 0, 0, 0);
  /* DMA BD Config: bd_id=6, len=256, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */
  void* v101 = __runtime_buffer_arg(v23);
  XAie_DmaDesc v102 = __Runtime_dma_bd_config_multidim(v1, v11, v101, 6, 256, -1, 0, 3, -1, 0, -1, 0, -1, 2, 4, 4, 32, 16, 0, 0, 0, 0);
  /* Create IO: channel_id=1, bd_id=6, tile=(1,0), direction=S2MM */
  /* Enable out-of-order BD on tile(1,0) ch=1 dir=S2MM */
  __Runtime_dma_channel_enable_ooo(v1, v11, 1, DMA_S2MM);
  io v103 = __Runtime_dma_createio_4(v11, v102, 1, 6, DMA_S2MM);
  /* DMA BD Config: bd_id=5, len=256, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */
  void* v104 = __runtime_buffer_arg((void*)35072);
  XAie_DmaDesc v105 = __Runtime_dma_bd_config(v1, v9, v104, 5, 256, 4, 1, 3, 5, -1, 4, 1, 6);
  /* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));
  /* DMA BD Config: bd_id=4, len=256, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */
  void* v106 = __runtime_buffer_arg((void*)34816);
  XAie_DmaDesc v107 = __Runtime_dma_bd_config(v1, v9, v106, 4, 256, 5, 1, 3, 5, -1, 4, 1, 6);
  /* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */
  io v108 = __Runtime_dma_createio_4(v9, v107, 0, 4, DMA_MM2S);
  /* DMA BD Config: bd_id=5, len=256, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v109 = __runtime_buffer_arg((void*)35072);
  XAie_DmaDesc v110 = __Runtime_dma_bd_config(v1, v14, v109, 5, 256, 4, 1, 4, 5, -1, 4, 1, 7);
  /* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */
  XAie_LockSetValue(dev, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));
  /* DMA BD Config: bd_id=4, len=256, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=7 */
  void* v111 = __runtime_buffer_arg((void*)34816);
  XAie_DmaDesc v112 = __Runtime_dma_bd_config(v1, v14, v111, 4, 256, 5, 1, 4, 5, -1, 4, 1, 7);
  /* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */
  io v113 = __Runtime_dma_createio_4(v14, v112, 0, 4, DMA_MM2S);
  ioevent v114 = __Runtime_startio(v1, v30, 0, 1);
  ioevent v115 = __Runtime_startio(v1, v43, 0, 1);
  ioevent v116 = __Runtime_startio(v1, v56, 1, 1);
  ioevent v117 = __Runtime_startio(v1, v73, 1, 2);
  ioevent v118 = __Runtime_startio(v1, v86, 2, 1);
  ioevent v119 = __Runtime_startio(v1, v103, 3, 2);
  ioevent v120 = __Runtime_startio(v1, v35, 0, 1);
  ioevent v121 = __Runtime_startio(v1, v40, 0, 1);
  ioevent v122 = __Runtime_startio(v1, v48, 0, 1);
  ioevent v123 = __Runtime_startio(v1, v53, 0, 1);
  ioevent v124 = __Runtime_startio(v1, v61, 1, 1);
  ioevent v125 = __Runtime_startio(v1, v66, 1, 1);
  ioevent v126 = __Runtime_startio(v1, v78, 2, 1);
  ioevent v127 = __Runtime_startio(v1, v83, 2, 1);
  ioevent v128 = __Runtime_startio(v1, v91, 1, 1);
  ioevent v129 = __Runtime_startio(v1, v96, 1, 1);
  ioevent v130 = __Runtime_startio(v1, v108, 2, 1);
  ioevent v131 = __Runtime_startio(v1, v113, 2, 1);
  /* Wait for 3 event(s) */
  __Runtime_wait(v1, v27);
  __Runtime_wait(v1, v117);
  __Runtime_wait(v1, v119);
  /* AieRt debug snapshot */
  {
    uint8_t _dbg_io_cols[] = {0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
    uint8_t _dbg_io_rows[] = {0, 3, 4, 0, 3, 4, 0, 3, 3, 0, 3, 3, 0, 4, 4, 0, 4, 4};
    uint8_t _dbg_io_chs[] = {0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0};
    uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 4, 4, 4, 2, 2, 6, 4, 4};
    int _dbg_io_dirs[] = {DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S};
    uint8_t _dbg_t_cols[] = {0, 0, 1, 1};
    uint8_t _dbg_t_rows[] = {3, 4, 3, 4};
    AieRt_DebugSnapshotFromCoords(dev,
        _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 18,
        _dbg_t_cols, _dbg_t_rows, 4);
  }
  return;
}

__global__ void dskernel_receiver(size_t v1) {
  // the real kernel will be emitted separately

  return;
}



// ===== User source (preserved from original file) =====
#define AIEHLC_TILING_STUBS_DEFINED
namespace aie {
enum class Pattern  { Broadcast = 0, Scatter = 1, Multicast = 2, Gather = 3 };
enum class Layout   { Row = 0, Col = 1, Grid = 2 };
enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };
enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };
enum class TileMode { Partition = 0, Overlap = 1 };
struct tile_dim {
  int size   = 0;
  int stride = 0;
  int groups = 0;
};
struct SpatialPolicy {
  Pattern pattern      = Pattern::Broadcast;
  Layout  distribution = Layout::Row;
  Flow    merge_order  = Flow::Default;
  int     pp_depth     = 2;
  int     max_buffer_bytes = 4096;
  LayoutTransform layout_transform = LayoutTransform::None;
  TileMode mode = TileMode::Partition;
  bool     require_full_coverage = true;
};
struct GemmSpace {
  SpatialPolicy policy;
  tile_dim m;
  tile_dim n;
  tile_dim k;
  tile_dim d1;
  tile_dim d2;
};
struct Conv2dSpace {
  SpatialPolicy policy;
  tile_dim ih;
  tile_dim iw;
  tile_dim ic;
  tile_dim oc;
  tile_dim kh;
  tile_dim kw;
  int stride = 1;
  int pad = 0;
  tile_dim m;
};
struct DmaTransform {
  struct Dim { int stride; int wrap; };
  Dim dims[4] = {};
  int num_dims = 0;
  int iter_step = 0;
  int iter_wrap = 0;
  int mode = 0;
  int halo_slice = 0;
  int halo_step = 0;
  int split_dim = 0;
  int raw_h = 0;
  int raw_wc = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int input_c = 0;
  int stride = 0;
  int ow = 0;
  int oh_per_row = 0;
  static constexpr DmaTransform flat() { return {}; }
  static constexpr DmaTransform im2col(int H, int W, int C,
      int KH, int KW, int S, int P) {
    DmaTransform d;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform dilated_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int D) {
    DmaTransform d;
    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;
    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;
    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform pool(int H, int W, int C,
      int KH, int KW, int S, int P) {
    return im2col(H, W, C, KH, KW, S, P);
  }
  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int G) {
    DmaTransform d;
    int CPG = C / G;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform transpose(int rows, int cols) {
    DmaTransform d;
    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};
    d.num_dims = 2;
    return d;
  }
  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {
    DmaTransform d;
    d.dims[0] = {H * W, C};
    d.dims[1] = {1, W};
    d.num_dims = 2;
    d.iter_step = W;
    d.iter_wrap = H;
    return d;
  }
  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {
    DmaTransform d;
    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};
    d.num_dims = 2;
    d.iter_step = 1; d.iter_wrap = C;
    return d;
  }
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    DmaTransform d;
    int OH = (H + 2*P - KH) / S + 1;
    int oh_per_row = OH / R;
    d.mode = 1;
    d.halo_slice = (oh_per_row - 1) * S + KH;
    d.halo_step  = oh_per_row * S;
    d.split_dim  = 0;
    d.raw_h  = H;
    d.raw_wc = W * C;
    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;
    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;
    return d;
  }
};
struct ConvTiling {
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);
  }
};
template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { using type = T; };
template<typename T> constexpr int get_num_rounds(T) { return 0; }
template<typename T> constexpr int get_buffer_size(T) { return 0; }
constexpr int get_tile_rows() { return 0; }
constexpr int get_tile_cols() { return 0; }
constexpr int get_k_dim() { return 0; }
constexpr int get_tile_m() { return 0; }
constexpr int get_tile_n() { return 0; }
constexpr int get_effective_k() { return 0; }
constexpr int get_k_rounds() { return 0; }
constexpr int get_spatial_m_rounds() { return 0; }
constexpr int get_spatial_n_rounds() { return 0; }
template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }
constexpr int get_kernel_h() { return 0; }
constexpr int get_kernel_w() { return 0; }
constexpr int get_input_c() { return 0; }
constexpr int get_stride() { return 0; }
constexpr int get_ow() { return 0; }
constexpr int get_oh_per_row() { return 0; }
constexpr int get_halo_slice() { return 0; }
}
struct aiePartition {
    int startCol, endCol, startRow, endRow;
};
struct aieMesh {
    int rows, cols;
    aiePartition partition;
    int meshId;
};
struct aieArray {
    int nextMeshId = 0;
    XAie_DevInst* _dev = nullptr;
    aieMesh partition(aiePartition p, int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol + 1);
        return aieMesh{rows, cols, p, meshId};
    }
    void* alloc(size_t size) { return __Runtime_alloc_buffer(_dev, size); }
    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }
    void synchronizecpu(void* ptr, size_t size) { __Runtime_sync_for_cpu(_dev, ptr, size); }
};
struct aieDim {
    int rows, cols;
    aiePartition partition;
    bool hasPartition;
    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}
    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), hasPartition(true) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
extern unsigned char _binary_kernel_matmul_start[];
extern unsigned char _binary_kernel_matmul_end[];
extern unsigned int _binary_kernel_matmul_size;

inline void __aie_launch(const char* kernel, aieMesh mesh, void* _t0, size_t _s0, void* _t1, size_t _s1, void* _t2, size_t _s2, ...) {
    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);
    __Runtime_set_kernel_elf(_binary_kernel_matmul_start);
    XAie_MemSyncForDevVAddr(dev, _t0, (uint64_t)_s0);
    XAie_MemSyncForDevVAddr(dev, _t1, (uint64_t)_s1);
    XAie_MemSyncForDevVAddr(dev, _t2, (uint64_t)_s2);
    host_canonicalized(dev, _t0, _t1, _t2);
}
inline void __aie_launch(const char* kernel, aieDim mesh, void* _t0, size_t _s0, void* _t1, size_t _s1, void* _t2, size_t _s2, ...) {
    (void)kernel;
    XAie_DevInst* dev;
    if (mesh.hasPartition) {
        dev = __Runtime_explicit_init_partition(mesh.partition.startCol, mesh.partition.endCol - mesh.partition.startCol + 1);
    } else {
        dev = __Runtime_explicit_init();
    }
    __Runtime_set_kernel_elf(_binary_kernel_matmul_start);
    XAie_MemSyncForDevVAddr(dev, _t0, (uint64_t)_s0);
    XAie_MemSyncForDevVAddr(dev, _t1, (uint64_t)_s1);
    XAie_MemSyncForDevVAddr(dev, _t2, (uint64_t)_s2);
    host_canonicalized(dev, _t0, _t1, _t2);
    __Runtime_explicit_teardown(dev);
}
// Stub type declarations for Clang parsing (function body skipped via #ifdef KERNEL_COMPILE)
#ifndef AIEHLC_STUBS_DEFINED
#define AIEHLC_STUBS_DEFINED
template<typename T> struct input_window {};
template<typename T> struct output_window {};
typedef int int32;
typedef input_window<int32> input_window_int32;
typedef output_window<int32> output_window_int32;
typedef signed char int8;
typedef input_window<int8> input_window_int8;
typedef output_window<int8> output_window_int8;
typedef short int16;
typedef input_window<int16> input_window_int16;
typedef output_window<int16> output_window_int16;
typedef unsigned char uint8_t;
typedef unsigned long uintptr_t;
typedef int int8_t __attribute__((mode(QI)));
typedef int int32_t __attribute__((mode(SI)));
typedef int v4int8 __attribute__((vector_size(4)));
typedef int v4int32 __attribute__((vector_size(16)));
inline unsigned get_coreid() { return 0; }
inline void klog(const char*, int) {}
template<typename T> inline void* acquire_input_window(T*) { return (void*)0; }
template<typename T> inline void* acquire_output_window(T*) { return (void*)0; }
template<typename T> inline void release_input_window(T*) {}
template<typename T> inline void release_output_window(T*) {}
#define BUF_SZ 16
#endif

// CUDA-style AIE API stubs for Clang parsing
#ifndef AIEHLC_TILING_STUBS_DEFINED
#define AIEHLC_TILING_STUBS_DEFINED
namespace aie {
enum class Pattern  { Broadcast = 0, Scatter = 1, Multicast = 2, Gather = 3 };
enum class Layout   { Row = 0, Col = 1, Grid = 2 };
enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };
enum class LayoutTransform { None = 0, DmaShuffle = 1, CoreShuffle = 2 };
enum class TileMode { Partition = 0, Overlap = 1 };
struct tile_dim {
  int size   = 0;
  int stride = 0;
  int groups = 0;
};
struct SpatialPolicy {
  Pattern pattern      = Pattern::Broadcast;
  Layout  distribution = Layout::Row;
  Flow    merge_order  = Flow::Default;
  int     pp_depth     = 2;
  int     max_buffer_bytes = 4096;
  LayoutTransform layout_transform = LayoutTransform::None;
  TileMode mode = TileMode::Partition;
  bool     require_full_coverage = true;
};
struct GemmSpace {
  SpatialPolicy policy;
  tile_dim m;
  tile_dim n;
  tile_dim k;
  tile_dim d1;
  tile_dim d2;
};
struct Conv2dSpace {
  SpatialPolicy policy;
  tile_dim ih;
  tile_dim iw;
  tile_dim ic;
  tile_dim oc;
  tile_dim kh;
  tile_dim kw;
  int stride = 1;
  int pad = 0;
  tile_dim m;
};
struct DmaTransform {
  struct Dim { int stride; int wrap; };
  Dim dims[4] = {};
  int num_dims = 0;
  int iter_step = 0;
  int iter_wrap = 0;
  int mode = 0;
  int halo_slice = 0;
  int halo_step = 0;
  int split_dim = 0;
  int raw_h = 0;
  int raw_wc = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int input_c = 0;
  int stride = 0;
  int ow = 0;
  int oh_per_row = 0;
  static constexpr DmaTransform flat() { return {}; }
  static constexpr DmaTransform im2col(int H, int W, int C,
      int KH, int KW, int S, int P) {
    DmaTransform d;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * C}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform dilated_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int D) {
    DmaTransform d;
    int OW = (W + 2*P - D*(KW-1) - 1) / S + 1;
    int OH = (H + 2*P - D*(KH-1) - 1) / S + 1;
    d.dims[0] = {D * C, KW}; d.dims[1] = {W * C * D, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform pool(int H, int W, int C,
      int KH, int KW, int S, int P) {
    return im2col(H, W, C, KH, KW, S, P);
  }
  static constexpr DmaTransform depthwise_im2col(int H, int W, int C,
      int KH, int KW, int S, int P, int G) {
    DmaTransform d;
    int CPG = C / G;
    int OW = (W + 2*P - KW) / S + 1;
    int OH = (H + 2*P - KH) / S + 1;
    d.dims[0] = {1, KW * CPG}; d.dims[1] = {W * C, KH}; d.dims[2] = {S * C, OW};
    d.num_dims = 3;
    d.iter_step = W * C * S; d.iter_wrap = OH;
    return d;
  }
  static constexpr DmaTransform transpose(int rows, int cols) {
    DmaTransform d;
    d.dims[0] = {cols, rows}; d.dims[1] = {1, cols};
    d.num_dims = 2;
    return d;
  }
  static constexpr DmaTransform chw_to_hwc(int C, int H, int W) {
    DmaTransform d;
    d.dims[0] = {H * W, C};
    d.dims[1] = {1, W};
    d.num_dims = 2;
    d.iter_step = W;
    d.iter_wrap = H;
    return d;
  }
  static constexpr DmaTransform hwc_to_chw(int H, int W, int C) {
    DmaTransform d;
    d.dims[0] = {C, W}; d.dims[1] = {W * C, H};
    d.num_dims = 2;
    d.iter_step = 1; d.iter_wrap = C;
    return d;
  }
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    DmaTransform d;
    int OH = (H + 2*P - KH) / S + 1;
    int oh_per_row = OH / R;
    d.mode = 1;
    d.halo_slice = (oh_per_row - 1) * S + KH;
    d.halo_step  = oh_per_row * S;
    d.split_dim  = 0;
    d.raw_h  = H;
    d.raw_wc = W * C;
    d.kernel_h = KH; d.kernel_w = KW; d.input_c = C; d.stride = S;
    d.ow = (W + 2*P - KW) / S + 1; d.oh_per_row = oh_per_row;
    return d;
  }
};
struct ConvTiling {
  static constexpr DmaTransform spatial(int H, int W, int C,
      int KH, int KW, int S, int P, int R) {
    return DmaTransform::spatial(H, W, C, KH, KW, S, P, R);
  }
};
template<typename T, auto Space, DmaTransform D = DmaTransform::flat()> struct port { using type = T; };
template<typename T> constexpr int get_num_rounds(T) { return 0; }
template<typename T> constexpr int get_buffer_size(T) { return 0; }
constexpr int get_tile_rows() { return 0; }
constexpr int get_tile_cols() { return 0; }
constexpr int get_data_row() { return 0; }
constexpr int get_data_col() { return 0; }
constexpr int get_k_dim() { return 0; }
constexpr int get_tile_m() { return 0; }
constexpr int get_tile_n() { return 0; }
constexpr int get_effective_k() { return 0; }
constexpr int get_k_rounds() { return 0; }
constexpr int get_spatial_m_rounds() { return 0; }
constexpr int get_spatial_n_rounds() { return 0; }
template<typename T> constexpr int get_spatial_multiple_rounds(T) { return 0; }
constexpr int get_kernel_h() { return 0; }
constexpr int get_kernel_w() { return 0; }
constexpr int get_input_c() { return 0; }
constexpr int get_stride() { return 0; }
constexpr int get_ow() { return 0; }
constexpr int get_oh_per_row() { return 0; }
constexpr int get_halo_slice() { return 0; }
}
struct aiePartition {
    int startCol, endCol, startRow, endRow;
};
struct XAie_DevInst;
extern "C" XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int numCols);
extern "C" XAie_DevInst *__Runtime_get_partition_dev(int meshId);
extern "C" void *__Runtime_alloc_buffer(XAie_DevInst *dev, __SIZE_TYPE__ size_bytes);
extern "C" void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);
extern "C" void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, __SIZE_TYPE__ size);
extern "C" void __Runtime_teardown_all();
struct aieMesh {
    int rows, cols;
    aiePartition partition;
    int meshId;
};
struct aieArray {
    int nextMeshId = 0;
    XAie_DevInst* _dev = nullptr;
    aieMesh partition(aiePartition p, int rows, int cols) {
        int meshId = nextMeshId++;
        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol + 1);
        return aieMesh{rows, cols, p, meshId};
    }
    void* alloc(__SIZE_TYPE__ size) { return __Runtime_alloc_buffer(_dev, size); }
    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }
    void synchronizecpu(void* ptr, __SIZE_TYPE__ size) { __Runtime_sync_for_cpu(_dev, ptr, size); }
};
struct aieDim {
    int rows, cols;
    aiePartition partition;
    bool hasPartition;
    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}
    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), hasPartition(true) {}
};
inline void aieSetDevice(int) {}
inline void aieDeviceSynchronize() {}
extern void host_canonicalized(...);
template<typename... Args>
inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {
    (void)kernel; (void)mesh; (void)sizeof...(args);
}
template<typename... Args>
inline void __aie_launch(const char* kernel, aieDim mesh, Args... args) {
    (void)kernel; (void)mesh; (void)sizeof...(args);
}
#endif

/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (sim version)
 *
 * 2x2 mesh matching hardware (VEK385 / XC2VE3858: 4 compute rows × 36 cols).
 * Reduced to 64x64x64 (vs 128x128x128) for faster simulation.
 * M,K,N multiples of (HW_ROWS * d1.size) = 2*16 = 32.
 */
// 2×2 mesh matching hardware (VEK385 / XC2VE3858: 4 compute rows × 36 cols).
// Reduced to 32x32x32 (vs 128x128x128) for faster simulation.
// M,K,N multiples of (HW_ROWS * d1.size) = 2*16 = 32.
#define HW_ROWS 2
#define HW_COLS 2
#define M 32
#define K 32
#define N 32

#include "simplematmul.h"
#pragma aie_debug_level(1 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER)
// Composition-based spatial spaces: a generic SpatialPolicy composed with a
// PER-PORT 2D iteration space. Each port describes its OWN matrix via d1/d2:
//   win_a A=[M,K] -> d1 = M-tile,  d2 = K-chunk
//   win_b B=[N,K] -> d1 = N-tile,  d2 = K-chunk
//   win_c C=[M,N] -> d1 = M-tile,  d2 = N-tile
constexpr aie::GemmSpace RowBA = {.policy = {.pattern = aie::Pattern::Broadcast,
                                             .distribution = aie::Layout::Row,
                                             .pp_depth = 2,
                                             .max_buffer_bytes = 4096,
                                             .mode = aie::TileMode::Partition},
                                  .d1 = {.size = 16, .stride = 16},  // A: M-tile (Partition: stride == size)
                                  .d2 = {.size = 64, .stride = 64}}; // A: K chunk (4 k-rounds for K=256)
constexpr aie::GemmSpace ColBB = {.policy = {.pattern = aie::Pattern::Broadcast,
                                             .distribution = aie::Layout::Col,
                                             .pp_depth = 2,
                                             .max_buffer_bytes = 4096,
                                             .mode = aie::TileMode::Partition},
                                  .d1 = {.size = 16, .stride = 16},  // B: N-tile
                                  .d2 = {.size = 64, .stride = 64}}; // B: K chunk
constexpr aie::GemmSpace LtoR_Merge = {.policy = {.pattern = aie::Pattern::Gather,
                                                  .distribution = aie::Layout::Row,
                                                  .merge_order = aie::Flow::LeftToRight,
                                                  .pp_depth = 2,
                                                  .max_buffer_bytes = 4096,
                                                  .mode = aie::TileMode::Partition},
                                       .d1 = {.size = 16, .stride = 16},  // C: M-tile
                                       .d2 = {.size = 16, .stride = 16}}; // C: N-tile
#define DEBUG_OUTPUT_ORDER 1

// Global variables for kernel: matmul
extern unsigned char _binary_kernel_matmul_start[];
extern unsigned char _binary_kernel_matmul_end[];
extern unsigned int _binary_kernel_matmul_size;




// Global variables for kernel: mul2
extern unsigned char _binary_kernel_mul2_start[];
extern unsigned char _binary_kernel_mul2_end[];
extern unsigned int _binary_kernel_mul2_size;



// HOST — argc/argv needed for aiehlc_ps_main(int, char**) in sim wrapper.
int main(int, char**) {
    printf("=== Matrix Multiply (SIM) on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8\n", M, N, M, K, K, N);
    // --- Device + mesh ---
    aieSetDevice(0);
    aieArray device;
    // 2x2 mesh: cols 3-4, rows 0-5. Matches HW config (VEK385 has 4 compute rows).
    aieMesh mesh = device.partition({3, 4, 0, 5}, HW_ROWS, HW_COLS);
    // aieDim mesh(HW_ROWS, HW_COLS);
    //  --- Allocate DMA-capable host memory (cache stays enabled) ---
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    // --- Initialize input matrices ---
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;
    // --- Launch kernel on tile mesh ---
    __aie_launch("matmul", mesh, A, (size_t)1024, B, (size_t)1024, C, (size_t)1024, M, N, K);
    // stlkernel<<mesh>>>(A, B, C);
    //  device.synchronizecpu(C, M * N * sizeof(int8_t) * 4);
    int result = verify_matmul(A, B, C);
    /// __aie_launch("mul2", mesh, C, B, A, M, N, K);
    // int result2 = verify_matmul(C, B, A);
    //   --- Wait for all partitions and teardown ---
    // --- Verify output ---
    // --- Cleanup ---
    device.free(A);
    device.free(B);
    device.free(C);
    return result;
}

