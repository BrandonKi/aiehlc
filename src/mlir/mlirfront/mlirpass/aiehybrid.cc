/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aiehybrid.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

extern bool use_llvm_aie;

void HybridPass::convertDialect(mlir::Operation* op)
{
	/*
	MLIRContext &context = getContext();
	ConvertionTarget target(context);
	target.addLegalDialect<>();
	target.addIllegalDialect<>();
	MyTypeConverter typeConverter;
	mlir::RewritePattern pattern(&context);
	pattern.add<MyOperation>(&context, typeConvertert);
	mlir::applyPartialConversion(module, target, std::move(pattern));
	*/
	return;
}

void HybridPass::runOnOperation() {
	std::cout << "HybridPass: " << std::endl;

	std::vector<mlir::Operation*> ops;

	mlir::Operation* op = getOperation();
	op->walk([&](mlir::Operation* op) {
		if (llvm::isa<mlir::ModuleOp>(op)) {
			llvm::outs() << "op is ModuleOp return \n";
			return;
		}
		
		llvm::outs() << "op is: ";
		op->print(llvm::outs());
		llvm::outs() << "\n";

		if (op->getName().getStringRef() =="Aie.create_kernelobject_op") {
			ops.push_back(op);
		}
	});

	for(auto *op : ops) {
		std::string kname;
		std::string fname;
		std::vector<Buffer> kfuncparams;

        uint32_t max_window_size = 0;

        Bcf bcf;
		Prx prx("Project", "me");

		mlir::aie::CreateKernelObjectOp kop = llvm::dyn_cast<mlir::aie::CreateKernelObjectOp >(*op);
		kname = llvm::dyn_cast<mlir::aie::KernelFuncNameType>(kop.getKname().getType()).getName().str();
		fname = llvm::dyn_cast<mlir::aie::KernelFileNameType>(kop.getFname().getType()).getName().str();
		std::string in_param_type = kop.getInParamType().str();
		std::string out_param_type = kop.getOutParamType().str();

		for (mlir::Value value : kop.getArguments()) {
			mlir::aie::CreateWindowOp window = value.getDefiningOp<mlir::aie::CreateWindowOp>();
			if (!window) {
				llvm::errs() << "Error: Expected a CreateWindowOp for value in " << "" << "\n";
				continue; // Skip if the cast failed
			}
			auto pingaddr =  window.getPingaddr();
			auto pongaddr = window.getPongaddr();
			auto direct = window.getDirection();
			auto wname = window.getName().str();
            auto acquireLockId = window.getPinglockid(); // MLIR uses pinglockid for acquire
            auto releaseLockId = window.getPonglockid(); // MLIR uses ponglockid for release
            std::cout << wname << std::endl;
            std::ostringstream ostr;

            // Check if this is ping-pong mode (pongaddr != 0) or legacy single buffer mode
            bool isPingPongMode = (pongaddr != 0);

            if (isPingPongMode) {
                // Ping-pong mode: use _ping and _pong suffixes
                ostr << wname << "_ping";
                auto wping = ostr.str();
                bcf.addsymbols(ostr.str(), 0x70000 + window.getPingaddr());
                ostr.str("");
                ostr.clear();
                ostr << wname << "_pong";
                auto wpong = ostr.str();
                bcf.addsymbols(ostr.str(), 0x70000 + window.getPongaddr());

                kfuncparams.push_back(Buffer(direct, wping, wpong, pingaddr, pongaddr, acquireLockId, releaseLockId));
                max_window_size = std::max(max_window_size, (uint32_t)(pongaddr - pingaddr));
            } else {
                // Legacy single buffer mode: just use the window name (no suffix)
                bcf.addsymbols(wname, 0x70000 + window.getPingaddr());
                // Use pingaddr for both ping and pong names (pong won't be used)
                kfuncparams.push_back(Buffer(direct, wname, wname, pingaddr, 0, acquireLockId, releaseLockId));
                if (window.getSize() > 0)
                    max_window_size = std::max(max_window_size, window.getSize());
            }
        }

        //FIXME count the variable to get the real stack size and give a real stack address
		bcf.setstack(0x7e000, 0x1024);
		//FIXME add the real reserved address
		bcf.addreservedDMB(0x40000, 0x10000);
		prx.add_kernel_info( "lbc", "kernel.ll", "../../build/" + kname + "/obj/");
		// prx.add_kernel_info( "lbc", "kernel.ll", "../../build/obj");
		
		prx.kernel_name = kname;
		bcf.kernel_name = kname;

		if(use_llvm_aie) {
			LdScript ldscript(bcf);
			// FIXME use the real stack address and size
			ldscript.setstack(0x7e000, 0x400);
			ldscript.exportfile();
		} else {
			bcf.exportfile();

			// TODO use the correct __AIEARCH__ value
			prx.setOption("cpp.define", "__AIENGINE__ __AIEARCH__=20", "1","");
			prx.setOption("llvm.xargs", "-fno-jump-tables -fno-discard-value-names -mllvm -chess-collapse-struct-types-during-linking=0", "1","");
			prx.setOption("llvm.lang", "Follow file extension", "", "");
			prx.setOption("bridge.cfg", bcf.getfilename(), "", "./");
			prx.setOption("cpp.include","&lt;XILINX_VITIS_AIETOOLS&gt;../TheHouseOfCommons ./ /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/aie_api /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1/aietools//include/drivers/aiengine/ /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/aie_api /proj/xbuilds/2023.1_daily_latest/installs/lin64/Vitis/2023.1//aietools//include/drivers/aiengine/","1","");
			prx.setOption("project.dir", "&lt;CONFIG&gt;./", "", "");
			prx.setOption("project.name", "kernel", "", "");
			prx.setOption("project.type", "exe", "", "");
			prx.exportfile();
		}
		Wrapper wrap(kname, fname);
        wrap.setbufsize(max_window_size);
        wrap.addkernelfuncparams(kfuncparams);
		wrap.set_kernel_in_param_type(in_param_type);
		wrap.set_kernel_out_param_type(out_param_type);

        // Check for streaming attributes
        if (auto streamingEnabled = kop->getAttrOfType<mlir::BoolAttr>("streaming_enabled")) {
            if (streamingEnabled.getValue()) {
                llvm::outs() << "Enabling streaming mode for wrapper generation\n";
                wrap.enableStreaming();
            }
        }

        wrap.exportfile();
		llvm::outs() << "Exported files for " << kname << "\n";
	}
}

