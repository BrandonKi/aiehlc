/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "AieFrontEnd.h"
#include "AieLinkDialect.h"
#include <getopt.h>
#include "tilinglinalg.h"

void dumpmlir();
void run_tests(std::string path){
  std::vector<std::string> files = {
    "loadkernel_test.mlir",
    "kernelObj_test.mlir",
    "window_test.mlir",
    "tilekernelObj_test.mlir"
  };

  std::cout << "**********Testing Printer Parser***********\n";
  for(auto str: files){
    TestPrinterParser(path + str);
  }
}

void traverse(std::string file_path){
  TraverseMLIR(file_path);
}

void generate_wrapper(std::string file_path){
  generateWrapperCode(file_path);
}

void aielinkdialect_test() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::aiel::AieLinkDialect>();
  ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
  OpBuilder builder(&context);
  builder.setInsertionPointToEnd(module.getBody());

  /*
	auto dataType = mlir::IntegerType::get(&context, 32);
	auto _row = mlir::IntegerAttr::get(dataType, row);
	auto _col = mlir::IntegerAttr::get(dataType, col);
        Value val_row = builder.create<ConstantInteger32Op>(UnknownLoc::get(&context), _row)
		.getResult();
        Value val_col = builder.create<ConstantInteger32Op>(UnknownLoc::get(&context), _col)
		.getResult();
	mlir::StringAttr kernelsrcAttr = builder.getStringAttr(kernel_src.c_str());

	mlir::Type type_val = mlir::aie::TileKernelType::get(&context, row, col, kernelsrcAttr);

        builder.create<CreateTileKernelObjectOp>(builder.getUnknownLoc(), type_val, val_row, val_col, 
		kernelsrcAttr);
        module.dump();
  */
}

void createlinalgfunc() {
  tilinglinalg tl;
  tl.creatematmul(256,64,128);
  return;
}

int main(int argc, char** argv) {
    int opt;
    while((opt = getopt(argc, argv, "ep:t:w:a:m:")) != -1){
      if(opt == -1) {
        printf("usage -m -a -w -t -ep\n");
        break;
      }
      switch(optopt){
        case 'm':{
          printf("mlir linalg test\n");
          createlinalgfunc();
          break;
        }
        case 'l':{
          aielinkdialect_test();
          break;
        }
        case 'e':{
          std::string input = "loadkernel(aa, {37, 21})";
		      generateMLIR(input);
          std::cout << "********* PRINTING THE MODULE DUMP FOR EMIT FUNCTIONS WRITTERN ************\n";
          CreateTileKernelObjectOperation(3, 4, "matrix_mult");
          CreateWindowOperation(1, -1, 0, 1, 14, 34, "mywindow");
          //CreateKernelObjectOperation(1, 1);
          CreateLoadKernel(3, 4, "mm");
          break;
        }
        case 't':{
          traverse(std::string(optarg));
          break;
        }
        case 'p':{
          run_tests(std::string(optarg));
          break;
        }
        case 'w':{
          generate_wrapper(std::string(optarg));
          break;
        }
        case '?':
          printf("Unknown option: %c\n", optopt);
          break;
        case ':':
          printf("Missing arg for %c\n", optopt);
          break;
				case 'a':{
					printf("pass\n");
					AieFrontEnd af;
					af.RunPass(std::string(optarg));
					break;
								 }
        default: {
          std::cerr << "reached default for option argument\n";
          exit(1);
        }
      }
    }
    return 0;
}

