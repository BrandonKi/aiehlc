/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include "../../include/gcommon.h"
#include "clang/AST/APValue.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

//#include "AieFrontEnd.h"
#include "../mlir/mlirfront/AieFrontEnd.h"
#include <boost/algorithm/string.hpp>

#include <ios>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <cstdlib>
#include <regex>
#include <unordered_map>

#include "routingimplement/include/hw/hwresource.h"
#include "tilinglinalg_pipeline.h"

class AieFrontEnd;

static bool use_llvm_aie = false;
static bool isTilingLinalgMode = false;
static int tilingMeshRows = 0, tilingMeshCols = 0;

struct ParsedTensorInfo {
    std::string varName;
    std::vector<int64_t> shape;
    int elementBitWidth;
    bool isInput;
    std::string spatialTag; // "row_broadcast_in", "col_broadcast_in", etc. or "" for default
    std::string policyName; // "RowBC", "ColBC", "LtoR_Merge", etc. (from aie::port<T, Policy>)
    // Resolved SpatialPolicy struct fields (from Clang AST constexpr evaluation)
    int pattern = 0;      // 0=Broadcast, 1=Scatter, 2=Multicast, 3=Gather
    int distribution = 0; // 0=Row, 1=Col, 2=Grid
    int mergeOrder = 0;   // 0=Default, 1=LeftToRight, 2=RightToLeft
    int pingPong = 2;
    int maxBufferBytes = 4096;   // max per-buffer size (PP_MAX_BYTES equivalent)
    // Two-level tiling hints (0 = auto-derive from memory budget)
    int tileM = 0;               // explicit sub-tile rows per core
    int tileN = 0;               // explicit sub-tile cols per core
    int tileK = 0;               // explicit K chunk size for temporal tiling
    bool policyResolved = false; // true once AST extraction succeeds
};
static std::vector<ParsedTensorInfo> parsedTensors;

// DerivedTilingParams is defined in tilinglinalg_pipeline.h
static DerivedTilingParams derivedTilingParams;

static int64_t macroDimM = 0, macroDimN = 0, macroDimK = 0; // GEMM dimensions from launch args or macros

static int parsedDebugLevel = -1; // -1 = not set by user, >=0 = #pragma aie_debug_level value
static std::string userSourceDir; // directory containing the original user source file
static PartitionDesc parsedPartition; // default: invalid (all -1), set by #pragma aie_partition
static std::vector<MeshKernelDesc> parsedMeshKernels; // multi-kernel mode: one per <<<mesh>>> launch
static std::vector<std::string> kernel_name_list;
static std::unordered_map<std::string, const clang::FunctionDecl*> globalKernelFuncs;
static std::string userKernelBody;     // raw source text of __global__ function body (last kernel, backward compat)
static std::string userKernelFuncName; // kernel function name from __global__ (last kernel, backward compat)
static std::unordered_map<std::string, std::string> globalKernelBodies; // per-kernel: name -> cleaned body text
static std::vector<std::string> userMacroDefines; // #define lines from user source

using namespace clang;
using namespace clang::tooling;

// Extract partition bounds from the 3rd argument of aieDim constructor.
// Handles InitListExpr {2,7,0,10}, CXXConstructExpr (copy/aggregate), and DeclRefExpr (variable).
static void extractPartitionFromExpr(const Expr *partExpr, ASTContext *Context) {
    partExpr = partExpr->IgnoreParenImpCasts();
    // Try InitListExpr: aiePartition{2,7,0,10} or {2,7,0,10}
    if (const auto *IL = dyn_cast<InitListExpr>(partExpr)) {
        if (IL->getNumInits() >= 4) {
            clang::Expr::EvalResult r0, r1, r2, r3;
            if (IL->getInit(0)->EvaluateAsInt(r0, *Context) && IL->getInit(1)->EvaluateAsInt(r1, *Context) &&
                IL->getInit(2)->EvaluateAsInt(r2, *Context) && IL->getInit(3)->EvaluateAsInt(r3, *Context)) {
                parsedPartition.startCol = r0.Val.getInt().getExtValue();
                parsedPartition.endCol = r1.Val.getInt().getExtValue();
                parsedPartition.startRow = r2.Val.getInt().getExtValue();
                parsedPartition.endRow = r3.Val.getInt().getExtValue();
                llvm::outs() << "[TilingLinalg] Partition: [" << parsedPartition.startCol << ","
                             << parsedPartition.endCol << "," << parsedPartition.startRow << ","
                             << parsedPartition.endRow << "]\n";
            }
        }
        return;
    }
    // Try CXXConstructExpr: aggregate or copy constructor
    if (const auto *CE = dyn_cast<CXXConstructExpr>(partExpr)) {
        if (CE->getNumArgs() >= 4) {
            clang::Expr::EvalResult r0, r1, r2, r3;
            if (CE->getArg(0)->EvaluateAsInt(r0, *Context) && CE->getArg(1)->EvaluateAsInt(r1, *Context) &&
                CE->getArg(2)->EvaluateAsInt(r2, *Context) && CE->getArg(3)->EvaluateAsInt(r3, *Context)) {
                parsedPartition.startCol = r0.Val.getInt().getExtValue();
                parsedPartition.endCol = r1.Val.getInt().getExtValue();
                parsedPartition.startRow = r2.Val.getInt().getExtValue();
                parsedPartition.endRow = r3.Val.getInt().getExtValue();
                llvm::outs() << "[TilingLinalg] Partition: [" << parsedPartition.startCol << ","
                             << parsedPartition.endCol << "," << parsedPartition.startRow << ","
                             << parsedPartition.endRow << "]\n";
            }
        } else if (CE->getNumArgs() == 1) {
            // Copy/move ctor — unwrap inner
            extractPartitionFromExpr(CE->getArg(0), Context);
        }
        return;
    }
    // Try DeclRefExpr: variable reference like `part`
    if (const auto *DR = dyn_cast<DeclRefExpr>(partExpr)) {
        if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
            if (VD->hasInit())
                extractPartitionFromExpr(VD->getInit(), Context);
        }
    }
}

class GlobalFunctionVisitor : public RecursiveASTVisitor<GlobalFunctionVisitor> {
private:
	std::string GetFuncText(FunctionDecl *f) {
		clang::SourceRange functionRange = f->getSourceRange();
		clang::SourceLocation startLocation = functionRange.getBegin();
		clang::SourceLocation endLocation = Rewrite->getSourceMgr().getExpansionLoc(functionRange.getEnd());
		clang::SourceLocation endLocationWithOffset = clang::Lexer::getLocForEndOfToken(endLocation, 0, Rewrite->getSourceMgr(), Rewrite->getLangOpts());
		bool InvalidTemp = false;
		std::string fullkernel;
		clang::StringRef text = Rewrite->getSourceMgr().getBufferData(Rewrite->getSourceMgr().getFileID(startLocation), &InvalidTemp);
		if (!InvalidTemp) {
			const char* begin = text.begin() + Rewrite->getSourceMgr().getFileOffset(startLocation);
			const char* end = text.begin() + Rewrite->getSourceMgr().getFileOffset(endLocationWithOffset);
			std::string functionString(begin, end);
			fullkernel = functionString;
		} else {
			std::cerr << "Error: Unable to access the buffer for the original function text.\n";
			return "";
		}
		std::string str(fullkernel.data(), fullkernel.size());
		return str;
	}
	void ReplaceKernel(FunctionDecl *f) {
		return;
	}
public:
    explicit GlobalFunctionVisitor(Rewriter* Rewrite, ASTContext *Context, AieFrontEnd* Aiefrontend)
        : Context(Context), Rewrite(Rewrite) , Aiefrontend(Aiefrontend){
			 std::ofstream kl(AOUT+std::string("kernel_list"), std::ios::out);
		}

     void ExportFunction(FunctionDecl *FD, std::string kname) {
	     /*
        if (FD->hasBody()) {
            llvm::outs() << FD->getNameAsString() << " Function Body:\n";
            FD->getBody()->printPretty(llvm::outs(), nullptr, PrintingPolicy(Context->getLangOpts()));
            llvm::outs() << "\n\n";
        }*/
			llvm::outs() << "Exporting File: " << AOUT+std::string("kernel_list") << "\n\n";
			 std::ofstream kl(AOUT+std::string("kernel_list"), std::ios::app | std::ios::out);
			 kl << kname << std::endl;
			 kl.close();

			 std::error_code error_code;
			 //change the ext from .c into .cc to fix the aiecompiler compile issue
			llvm::outs() << "Exporting File: " << AOUT+kname + ".cc" << "\n";
			 std::ofstream fd(AOUT+kname + ".cc");
			 auto str = GetFuncText(FD);
			 //std::string toRemove = "__attribute__((annotate(\"__global__\")))";
			//fixme, shoud only remove mem_address and global annotate attribute, correct way is to do
      // one more pass to remove the related attribute
			 std::string toRemove = "__attribute__";
			 size_t pos, prepos=std::string::npos;
			 while((pos = str.find(toRemove)) != std::string::npos){
         //avoid infinite loop
				 if (prepos != std::string::npos && prepos == pos) break;
				 prepos = pos;
         //the remove logic
				 auto nstr = str.substr(pos + toRemove.size());
				 auto npos = nstr.find("(");
				 if (npos == std::string::npos) break;
				 auto cstr = nstr.c_str();
				 int lb = 1;
				 while(lb && npos < nstr.size()) {
					 auto c = cstr[++npos];
					 lb += (c == '(' ? 1 :(c == ')' ? -1 : 0) );
					 //std::cout << "npos is " << npos << " "<<lb << " is lb cstr is " << &cstr[npos]<< std::endl;
				 }
				 str.erase(pos, toRemove.size() + npos + 1);
				 //avoid infinite loop, reset prepos
				 prepos = std::string::npos;
			 }

             // Remove #ifdef KERNEL_COMPILE / #endif guard pairs added during
             // preprocessing.  Only remove #endif lines that were inserted to
             // close the KERNEL_COMPILE guard — leave user #endif directives
             // (e.g. #ifdef DEBUG_OUTPUT_ORDER) intact.
             std::string ifdefToRemove = "#ifdef KERNEL_COMPILE";
             while ((pos = str.find(ifdefToRemove)) != std::string::npos) {
                 // Erase the #ifdef KERNEL_COMPILE line
                 auto endOfLine = str.find("\n", pos);
                 if (endOfLine != std::string::npos) {
                     str.erase(pos, endOfLine - pos + 1);
                 } else {
                     str.erase(pos, ifdefToRemove.size());
                 }
                 // Find and erase the matching #endif (last one in the string,
                 // since the KERNEL_COMPILE guard wraps the entire body).
                 std::string endifToRemove = "#endif";
                 auto lastEndif = str.rfind(endifToRemove);
                 if (lastEndif != std::string::npos) {
                     auto endOfEndifLine = str.find("\n", lastEndif);
                     if (endOfEndifLine != std::string::npos) {
                         str.erase(lastEndif, endOfEndifLine - lastEndif + 1);
                     } else {
                         str.erase(lastEndif, endifToRemove.size());
                     }
                 }
             }

             // Strip spatial type wrappers from kernel source text
             // "aie::row_broadcast_in<input_window_int8 *>" → "input_window_int8 *"
             for (const auto &wrapper : {"row_broadcast_in", "col_broadcast_in", "tiled_in", "row_major_out",
                                         "col_major_out", "row_reduce_out"}) {
                 std::string prefix = "aie::" + std::string(wrapper) + "<";
                 size_t spos;
                 while ((spos = str.find(prefix)) != std::string::npos) {
                     size_t start = spos;
                     size_t angleStart = spos + prefix.size();
                     int depth = 1;
                     size_t end = angleStart;
                     while (end < str.size() && depth > 0) {
                         if (str[end] == '<')
                             depth++;
                         else if (str[end] == '>')
                             depth--;
                         end++;
                     }
                     // Replace "aie::wrapper<INNER>" with "INNER"
                     std::string inner = str.substr(angleStart, end - angleStart - 1);
                     str.replace(start, end - start, inner);
                 }
             }

             // Strip aie::port<T, Policy> → T (extract first template arg only)
             {
                 std::string portPrefix = "aie::port<";
                 size_t spos;
                 while ((spos = str.find(portPrefix)) != std::string::npos) {
                     size_t start = spos;
                     size_t angleStart = spos + portPrefix.size();
                     // Find the top-level comma (separating T from Policy) or closing >
                     int depth = 1;
                     size_t commaPos = std::string::npos;
                     size_t end = angleStart;
                     while (end < str.size() && depth > 0) {
                         if (str[end] == '<')
                             depth++;
                         else if (str[end] == '>')
                             depth--;
                         else if (str[end] == ',' && depth == 1 && commaPos == std::string::npos)
                             commaPos = end;
                         if (depth > 0)
                             end++;
                     }
                     end++; // skip closing >
                     // Extract first template arg (the inner type T)
                     size_t innerEnd = (commaPos != std::string::npos) ? commaPos : (end - 1);
                     std::string inner = str.substr(angleStart, innerEnd - angleStart);
                     // Trim whitespace
                     while (!inner.empty() && inner.front() == ' ')
                         inner.erase(0, 1);
                     while (!inner.empty() && inner.back() == ' ')
                         inner.pop_back();
                     str.replace(start, end - start, inner);
                 }
             }

             // Capture cleaned kernel body for tiling mode computekernel.cc emission
             userKernelFuncName = kname;
             userKernelBody = str;
             globalKernelBodies[kname] = str; // per-kernel map for multi-kernel support
             llvm::outs() << "[TilingLinalg] Captured __global__ kernel body for: " << kname << "\n";

             std::string header = "/******************************************************************************"
                                  "\n* Auto-generated by aiehlc.\n"
                                  "******************************************************************************/\n";
             // llvm aie can not handle these headers yet, results in compile errors
			 if(use_llvm_aie) {
				header += 
				 "\n#include <adf.h>\n\n"
				//  "\n#include <aie_api/aie.hpp>\n"
				//  "\n#include <aie_api/aie_adf.hpp>\n"
				//  "\n#include <aie_api/utils.hpp>\n\n";
				;
			 }
			 else {
				header += 
				 "\n#include <adf.h>"
				 "\n#include <aie_api/aie.hpp>"
				 "\n#include <aie_api/aie_adf.hpp>"
				 "\n#include <aie_api/utils.hpp>\n\n";
			 }
             // Add user macro definitions before kernel body
             std::string macroBlock;
             if (!userMacroDefines.empty()) {
                 macroBlock = "\n// User macro definitions from source file\n";
                 for (const auto &macro : userMacroDefines) {
                     macroBlock += macro + "\n";
                 }
                 macroBlock += "\n";
             }
             str = header + macroBlock + str;
             fd << str << std::endl;
    }

		 void printSourceRange(const clang::SourceRange &range, const clang::SourceManager &sourceManager) {
			 llvm::outs() << "SourceRange: ";

			 // Get the start and end locations
			 clang::SourceLocation startLoc = range.getBegin();
			 clang::SourceLocation endLoc = range.getEnd();

			 // Print the file name
			 auto& sc = Rewrite->getSourceMgr();
			 llvm::outs() << sourceManager.getFilename(startLoc).str() << ":";
			 llvm::outs() << sourceManager.getFilename(endLoc).str() << ":";

			 ///*
			 // Print the start line and column
			 llvm::outs() << sourceManager.getSpellingLineNumber(startLoc) << ":" << sourceManager.getSpellingColumnNumber(startLoc) << "-";

			 // Print the end line and column
			 llvm::outs() << sourceManager.getSpellingColumnNumber(endLoc) << ":" << sourceManager.getSpellingColumnNumber(endLoc) << "\n";
			 //*/
		 }
		 bool VisitFunctionDecl(FunctionDecl *f) {
			 if (f->hasBody() && f->hasAttr<AnnotateAttr>()) {
                 // Check for streaming annotation
                 bool isStreamingKernel = false;
                 for (auto attr : f->attrs()) {
                     if (auto anno = clang::dyn_cast<clang::AnnotateAttr>(attr)) {
                         if (anno->getAnnotation() == "streaming") {
                             isStreamingKernel = true;
                             llvm::outs() << "Detected streaming kernel annotation\n";
                             break;
                         }
                     }
                 }

                for (auto attr:f->attrs()) {
					if (auto anno = clang::dyn_cast<clang::AnnotateAttr>(attr)){
						if (anno->getAnnotation() == "__global__" || anno->getAnnotation() == "__kernel__") {
							std::string kernelName = f->getNameInfo().getName().getAsString();

							// Store function decl for later tensor parameter extraction
							globalKernelFuncs[kernelName] = f;

							// std::cout << "find global " << f->getNameInfo().getName().getAsString() << std::endl;
							ParmVarDecl *param = f->getParamDecl(0);
							// llvm::outs() << "param->hasAttrs() is " << param->hasAttrs() << "\n";
							//llvm::outs << "range is " << range <<"\n";
							ExportFunction(f, kernelName);
							clang::SourceRange functionRange = f->getSourceRange();
							clang::SourceLocation startLocation = functionRange.getBegin();

							clang::SourceLocation lineStart = Rewrite->getSourceMgr().translateLineCol(
							Rewrite->getSourceMgr().getFileID(startLocation), 
							Rewrite->getSourceMgr().getSpellingLineNumber(startLocation), 1);
							Rewrite->RemoveText(f->getSourceRange());
                            // do other rewrite logic only after the RemoveText work done
                            // Pass streaming flag to frontend
                            Aiefrontend->createKernelDefinitionOp(f, Rewrite, lineStart, isStreamingKernel);
                            if (anno->getAnnotation() == "__global__") {
								std::string globalVars =
									"\n// Global variables for kernel: " + kernelName + "\n" +
									"extern unsigned char _binary_kernel_" + kernelName + "_start[];\n" +
									"extern unsigned char _binary_kernel_" + kernelName + "_end[];\n" +
									"extern unsigned int _binary_kernel_" + kernelName + "_size;\n\n";
								Rewrite->InsertText(lineStart, globalVars, true, true);
							} else if(anno->getAnnotation() == "__kernel__") {
                                // _start is declared in the generated template function; no additional globals needed here.
                            }
						} else {
							std::cerr << "not global func" << std::endl;
						}
						auto range = f->getSourceRange();
						auto& sc = Rewrite->getSourceMgr();
						// std::cout << "range.isValid() " << range.isValid() <<std::endl;
						// printSourceRange(range,  sc);
					}
					for (unsigned int i = 0; i < f->getNumParams(); ++i) {
						ParmVarDecl *param = f->getParamDecl(i);
						// llvm::outs() << "  Param " << (i + 1) << ": type: "
						// 	<< param->getType().getAsString() << "  name:"
						// 	<< param->getNameAsString() << "\n";
					}
				}
			}
			Stmt *body = f->getBody();
			TraverseStmt(body);

			 //PrintCode();
			 // This involves manipulating the AST node of the function
			 return true;
		 }
		 std::string GetCallExprString(const CallExpr *  CE) {
			 std::ostringstream ostr;
			 const FunctionDecl *Callee = CE->getDirectCallee();
			 if (Callee) {
				 ostr << Callee->getNameAsString() << "(";
				 
				 //const Expr *thirdArgExpr = CE->getArg(2)->IgnoreImpCasts();
				 //std::string functionName = "";
				 //if (const clang::StringLiteral *strLit = clang::dyn_cast<clang::StringLiteral>(thirdArgExpr)) {
					// functionName = strLit->getString().str();
					 // Use 'functionName' as needed...
				//	 std::cout<< "load kernel functionName " << functionName << std::endl; 
				 //}
				 auto num = CE->getNumArgs();			
				 for (unsigned i = 0; i < num; i++) {
					 const clang::Expr* arg = CE->getArg(i);
					 clang::QualType argType = arg->getType();
					//  std::cout << "x- Parameter " << i << " type: ";
					//  std::cout << argType.getAsString();
					//  std::cout << std::endl;
					 if (auto *DRE = dyn_cast<DeclRefExpr>(arg->IgnoreImpCasts())) {
						 if (const VarDecl *VD = dyn_cast<const VarDecl>(DRE->getDecl())) {
							 ostr << VD->getNameAsString() ;
							 ostr << ((i<num-1) ? ",":" ");
						 }
					 } else if (auto *CallAsArg = dyn_cast<CallExpr>(arg)) {
							 ostr << GetCallExprString(CallAsArg);
							 ostr << ((i<num-1) ?",":" ");
					 } else if(const clang::PointerType *ptrType = argType->getAs<clang::PointerType>()) {
						 // Check if the pointee type is char (indicating char* or const char*)
						 clang::QualType pointeeType = ptrType->getPointeeType();
						 if (pointeeType->isCharType()) {
							 std::string argStr;
							 llvm::raw_string_ostream os(argStr);
							 arg->printPretty(os, nullptr, PrintingPolicy(LangOptions()));
							//  llvm::outs() << "Pretty printed argument " << i << ": " << os.str() << "\n";
							 ostr<< os.str() <<((i<num-1) ? ",":" ");
							 /*
							 if (const DeclRefExpr *declRef = dyn_cast<const DeclRefExpr>(arg->IgnoreParenImpCasts())) {
								 const ValueDecl *valueDecl = declRef->getDecl();

								 // Get the name of the function or variable
								 std::string paramName = valueDecl->getNameAsString();
								 llvm::outs() << "Pointer parameter name: " << paramName << "\n";
							 }
							 // Evaluate the expression
							 //clang::Expr::EvalResult evalResult;
							 //if (arg->EvaluateAsRValue(evalResult, *Context)) {
							 ///	clang::APValue value = evalResult.Val;
								//auto str = clang::dyn_cast<clang::StringLiteral>(CE->getArg(i)->IgnoreImpCasts());
							  //ostr << str->getString().str() << ((i<num-1) ? ",":" ");
							 //char* charArray = const_cast<char*>(value.getCharPtr().getPointer());
							 //	std::cout << " it is string " << str->getString().str() << " " << value.getCharPtr().getPointer() << std::endl;*/
							 }
						} else if (const CStyleCastExpr *castExpr = dyn_cast<const CStyleCastExpr>(arg)) {
								 // Get the expression being cast
								 const Expr *subExpr = castExpr->getSubExpr();

								 // Check if it's a reference to a variable
								 if (const DeclRefExpr *declRef = dyn_cast<const DeclRefExpr>(subExpr)) {
									 // Get the variable declaration
									 const ValueDecl *valueDecl = declRef->getDecl();

									 // Retrieve and print the name of the variable
									//  llvm::outs() << "Variable name: " << valueDecl->getName() << "\n";
									 ostr<< valueDecl->getName().str() <<((i < num -1) ?",":" ");
								 }
						} else if (argType->isIntegerType()) {
						 clang::Expr::EvalResult xResult;
						 if (arg->EvaluateAsInt(xResult, Callee->getASTContext())) {
							 int v = xResult.Val.getInt().getExtValue();
							 ostr << v ;
							 ostr<< ((i < num -1) ?",":" ");
						 }
					 }else {
						std::string argStr;
						llvm::raw_string_ostream os(argStr);
						arg->printPretty(os, nullptr, PrintingPolicy(LangOptions()));
						ostr << os.str() << ((i < num - 1) ? "," : " ");
					 }
				 }
			 }

			 ostr<< ")";
			 return ostr.str();
		}
		 //*/

		 bool VisitCallExpr(const CallExpr * 	CE) {
			const FunctionDecl *Callee = CE->getDirectCallee();

			// Detect __aie_launch("kernelName", mesh, A, B, C, M, N, K) for tiling mode
			if (Callee && Callee->getNameAsString() == "__aie_launch") {
				isTilingLinalgMode = true;

                // Extract kernel name to check for duplicate AST visits
                std::string currentLaunchKernel;
                if (CE->getNumArgs() >= 1) {
                    if (const auto *SL = dyn_cast<clang::StringLiteral>(CE->getArg(0)->IgnoreParenImpCasts()))
                        currentLaunchKernel = SL->getString().str();
                }

                // Guard: skip if this exact kernel was already parsed (AST visitor may visit twice)
                // Important: do NOT clear globals before this check — a same-name kernel called
                // twice (e.g. matmul<<<mesh>>>(A,B,C); matmul<<<mesh>>>(C,B,A);) must not
                // clobber the state captured from the first visit.
                for (const auto &mk : parsedMeshKernels) {
                    if (mk.kernelName == currentLaunchKernel)
                        return true;
                }

                // For multi-kernel: each launch gets its own parsedTensors / derivedTilingParams.
                // Clear per-kernel state so this kernel starts fresh.
                // This must happen AFTER the duplicate check above.
                bool isFirstKernel = parsedMeshKernels.empty();
                parsedTensors.clear();
                derivedTilingParams = DerivedTilingParams();

                // Arg 0: kernel name (string literal after preprocessing)
				if (CE->getNumArgs() >= 2) {
                    if (!currentLaunchKernel.empty()) {
                        llvm::outs() << "[TilingLinalg] Detected kernel launch: " << currentLaunchKernel << "\n";
                    }
                    // Arg 1: aieDim or aieMesh variable — extract rows/cols
                    const Expr *meshArg = CE->getArg(1)->IgnoreParenImpCasts();

                    // Local vars for this kernel's mesh dims and partition
                    int localMeshRows = 0, localMeshCols = 0;
                    PartitionDesc localPartition;
                    // Helper: extract mesh dims from a CXXConstructExpr (aieDim(rows, cols [, part]))
                    auto extractDimsFromConstruct = [&](const CXXConstructExpr *Construct) {
                        if (Construct->getNumArgs() >= 2) {
                            clang::Expr::EvalResult r0, r1;
                            if (Construct->getArg(0)->EvaluateAsInt(r0, *Context) &&
								Construct->getArg(1)->EvaluateAsInt(r1, *Context)) {
                                localMeshRows = r0.Val.getInt().getExtValue();
                                localMeshCols = r1.Val.getInt().getExtValue();
                            }
                            if (Construct->getNumArgs() >= 3) {
                                extractPartitionFromExpr(Construct->getArg(2), Context);
                                localPartition = parsedPartition;
                            }
                        }
                    };

                    // Helper: extract mesh dims from an aieMesh VarDecl initialized via device.partition(part, rows,
                    // cols)
                    auto extractDimsFromAieMeshVar = [&](const VarDecl *VD) -> bool {
                        if (!VD->hasInit())
                            return false;
                        const Expr *init = VD->getInit()->IgnoreParenImpCasts();
                        // Handle ExprWithCleanups wrapper
                        if (const auto *EWC = dyn_cast<ExprWithCleanups>(init))
                            init = EWC->getSubExpr()->IgnoreParenImpCasts();
                        // Handle MaterializeTemporaryExpr wrapper
                        if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(init))
                            init = MTE->getSubExpr()->IgnoreParenImpCasts();
                        // device.partition({...}, rows, cols) is a CXXMemberCallExpr
                        if (const auto *MCE = dyn_cast<CXXMemberCallExpr>(init)) {
                            if (const auto *MD = MCE->getMethodDecl()) {
                                if (MD->getNameAsString() == "partition" && MCE->getNumArgs() >= 3) {
                                    // Arg 0: aiePartition, Arg 1: rows, Arg 2: cols
                                    extractPartitionFromExpr(MCE->getArg(0), Context);
                                    localPartition = parsedPartition;
                                    clang::Expr::EvalResult r0, r1;
                                    if (MCE->getArg(1)->EvaluateAsInt(r0, *Context) &&
                                        MCE->getArg(2)->EvaluateAsInt(r1, *Context)) {
                                        localMeshRows = r0.Val.getInt().getExtValue();
                                        localMeshCols = r1.Val.getInt().getExtValue();
                                    }
                                    return true;
                                }
                            }
                        }
                        // Fallback: CXXConstructExpr (aieDim or aggregate init)
                        if (const auto *Construct = dyn_cast<CXXConstructExpr>(init)) {
                            extractDimsFromConstruct(Construct);
                            return localMeshRows > 0;
                        }
                        return false;
                    };

                    if (const auto *Construct = dyn_cast<CXXConstructExpr>(meshArg)) {
                        if (Construct->getNumArgs() >= 2) {
                            // Direct construction: __aie_launch("k", aieDim(4,4), ...)
                            extractDimsFromConstruct(Construct);
                        } else if (Construct->getNumArgs() == 1) {
                            // Copy/move constructor: __aie_launch("k", mesh, ...) where mesh is by-value
                            const Expr *inner = Construct->getArg(0)->IgnoreParenImpCasts();
                            if (const auto *DR = dyn_cast<DeclRefExpr>(inner)) {
                                if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
                                    if (!extractDimsFromAieMeshVar(VD)) {
                                        // Try aieDim ctor inside VarDecl init
                                        if (VD->hasInit()) {
                                            if (const auto *IC =
                                                    dyn_cast<CXXConstructExpr>(VD->getInit()->IgnoreParenImpCasts())) {
                                                extractDimsFromConstruct(IC);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if (const auto *DR = dyn_cast<DeclRefExpr>(meshArg)) {
                        // mesh variable already declared — get rows/cols from initializer
                        if (const auto *VD = dyn_cast<VarDecl>(DR->getDecl())) {
                            if (!extractDimsFromAieMeshVar(VD)) {
                                // Fallback: try CXXConstructExpr (aieDim)
                                if (VD->hasInit()) {
                                    if (const auto *Construct =
                                            dyn_cast<CXXConstructExpr>(VD->getInit()->IgnoreParenImpCasts())) {
                                        extractDimsFromConstruct(Construct);
                                    }
                                }
                            }
                        }
                    }

                    if (localMeshRows > 0 && localMeshCols > 0) {
                        llvm::outs() << "[TilingLinalg] Mesh: " << localMeshRows << " x " << localMeshCols << "\n";
                        if (localPartition.isValid()) {
                            llvm::outs() << "[TilingLinalg] Partition: [" << localPartition.startCol << ","
                                         << localPartition.endCol << "," << localPartition.startRow << ","
                                         << localPartition.endRow << "]\n";
                        }
                    }

                    // Update globals for backward compat (first kernel only)
                    if (isFirstKernel) {
                        tilingMeshRows = localMeshRows;
                        tilingMeshCols = localMeshCols;
                        parsedPartition = localPartition;
                    }
                    // Extract tensor params from the __global__ kernel function signature
					// Args 2+ of __aie_launch correspond to the kernel function parameters
					std::string launchKernelName;
					if (const auto *SL2 = dyn_cast<clang::StringLiteral>(CE->getArg(0)->IgnoreParenImpCasts())) {
						launchKernelName = SL2->getString().str();
					}

					if (!launchKernelName.empty()) {
						auto it = globalKernelFuncs.find(launchKernelName);
						if (it != globalKernelFuncs.end()) {
							const FunctionDecl *kernelFD = it->second;
							unsigned numKernelParams = kernelFD->getNumParams();

                            // Semantic validation: check launch arg count vs kernel param count
                            // Extra launch args beyond kernel params are dimension scalars (M, N, K)
                            unsigned numLaunchArgs = CE->getNumArgs() - 2; // exclude name and mesh
                            if (numLaunchArgs < numKernelParams) {
                                llvm::errs() << "Error: kernel '" << launchKernelName
											 << "' declares " << numKernelParams << " parameters but launch provides "
											 << numLaunchArgs << " arguments.\n";
								llvm::errs() << "  __global__ void " << launchKernelName << "(...) has "
											 << numKernelParams << " params\n";
								llvm::errs() << "  " << launchKernelName << "<<<mesh>>>(...) provides "
											 << numLaunchArgs << " args\n";
								// Don't abort — continue with available info
                            } else if (numLaunchArgs > numKernelParams) {
                                llvm::outs() << "[TilingLinalg] " << (numLaunchArgs - numKernelParams)
                                             << " extra launch args detected (dimension scalars)\n";
                            }

                            // Helper lambda: check if a QualType is a spatial wrapper or pointer type
                            // (i.e. a tensor parameter, not a scalar dimension).
                            // Spatial wrappers like aie::row_broadcast_in<T*> are struct types
                            // (not pointer types), but they wrap a pointer inside.
                            auto isTensorParam = [](clang::QualType qt) -> bool {
                                if (qt->isPointerType())
                                    return true;
                                // Check for aie:: spatial wrapper struct (legacy aliases + new port<>)
                                std::string typeStr = qt.getUnqualifiedType().getAsString();
                                for (const char *tag :
                                     {"aie::row_broadcast_in", "aie::col_broadcast_in", "aie::tiled_in",
                                      "aie::row_major_out", "aie::col_major_out", "aie::row_reduce_out", "aie::port"}) {
                                    if (typeStr.find(tag) != std::string::npos)
                                        return true;
                                }
                                return false;
                            };

                            // First pass: collect scalar dimension values from kernel params
                            // Launch args start at index 2 (0=name, 1=mesh, 2..=kernel params)
                            std::vector<int64_t> scalarDimValues;
							for (unsigned i = 0; i < numKernelParams; ++i) {
								const ParmVarDecl *kp = kernelFD->getParamDecl(i);
								clang::QualType ptype = kp->getType();
                                if (!isTensorParam(ptype)) {
                                    // Scalar param — try to evaluate from launch arg
									unsigned launchArgIdx = i + 2;
									if (launchArgIdx < CE->getNumArgs()) {
										clang::Expr::EvalResult evalRes;
										if (CE->getArg(launchArgIdx)->EvaluateAsInt(evalRes, *Context)) {
											scalarDimValues.push_back(evalRes.Val.getInt().getExtValue());
										}
									}
                                }
                            }

                            // Second: collect extra launch args beyond kernel params as dimension scalars
                            // e.g. matmul<<<mesh>>>(A, B, C, M, N, K) — M,N,K are at indices numKernelParams+2..
                            for (unsigned i = numKernelParams; i < numLaunchArgs; ++i) {
                                unsigned launchArgIdx = i + 2;
                                if (launchArgIdx < CE->getNumArgs()) {
                                    clang::Expr::EvalResult evalRes;
                                    if (CE->getArg(launchArgIdx)->EvaluateAsInt(evalRes, *Context)) {
                                        int64_t val = evalRes.Val.getInt().getExtValue();
                                        scalarDimValues.push_back(val);
                                        llvm::outs() << "[TilingLinalg] Extra launch arg[" << (i - numKernelParams)
                                                     << "] = " << val << "\n";
                                    }
                                }
                            }

                            // Determine default shape from scalar dims (e.g. M, N, K -> MxN for each tensor)
							// For GEMM: A[M*K], B[K*N], C[M*N] with scalars [M, N, K]
							// General case: use first two scalars as shape for all tensors
							int64_t defaultDim0 = 16, defaultDim1 = 16;
							if (scalarDimValues.size() >= 2) {
								defaultDim0 = scalarDimValues[0];
								defaultDim1 = scalarDimValues[1];
							} else if (scalarDimValues.size() == 1) {
								defaultDim0 = defaultDim1 = scalarDimValues[0];
							}

                            // Map scalar dims to GEMM M, N, K when 3 values available
                            // (from either kernel params or extra launch args)
                            macroDimM = 0;
                            macroDimN = 0;
                            macroDimK = 0;
                            if (scalarDimValues.size() >= 3) {
                                macroDimM = scalarDimValues[0];
                                macroDimN = scalarDimValues[1];
                                macroDimK = scalarDimValues[2];
                                llvm::outs() << "[TilingLinalg] GEMM dims from launch args: M=" << macroDimM
                                             << " N=" << macroDimN << " K=" << macroDimK << "\n";
                                defaultDim0 = macroDimM;
                                defaultDim1 = macroDimK;
                            }
                            // Fallback: when no scalar dims found, try to extract M, N, K from
                            // user #define macros (e.g. #define M 32, #define K 32, #define N 32).
                            // These are the standard GEMM dimension macros used in the kernel source.
                            if (macroDimM == 0 && scalarDimValues.empty()) {
                                for (const auto &macroLine : userMacroDefines) {
                                    // Parse "#define <NAME> <integer>"
                                    std::istringstream mss(macroLine);
                                    std::string tok_define, tok_name, tok_value;
                                    mss >> tok_define >> tok_name >> tok_value;
                                    if (tok_define != "#define" || tok_name.empty() || tok_value.empty())
                                        continue;
                                    // Only accept pure integer values (no expressions)
                                    bool allDigits = true;
                                    for (char c : tok_value) {
                                        if (!std::isdigit(c)) {
                                            allDigits = false;
                                            break;
                                        }
                                    }
                                    if (!allDigits)
                                        continue;
                                    int64_t val = std::stoll(tok_value);
                                    if (tok_name == "M")
                                        macroDimM = val;
                                    else if (tok_name == "N")
                                        macroDimN = val;
                                    else if (tok_name == "K")
                                        macroDimK = val;
                                }
                                if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    llvm::outs() << "[TilingLinalg] GEMM dims from macros: M=" << macroDimM
                                                 << " N=" << macroDimN << " K=" << macroDimK << "\n";
                                    // Update defaults (used when no per-tensor shape can be determined)
                                    defaultDim0 = macroDimM;
                                    defaultDim1 = macroDimK;
                                } else if (macroDimM > 0) {
                                    // Partial: at least M is defined
                                    defaultDim0 = macroDimM;
                                    defaultDim1 = macroDimN > 0 ? macroDimN : macroDimM;
                                }
                            }

                            // Second pass: build ParsedTensorInfo for each tensor parameter
                            // Handles both bare pointer types (input_window_int8*) and
                            // spatial wrapper types (aie::row_broadcast_in<input_window_int8 *>)
                            for (unsigned i = 0; i < numKernelParams; ++i) {
								const ParmVarDecl *kp = kernelFD->getParamDecl(i);
								clang::QualType ptype = kp->getType();
                                if (!isTensorParam(ptype))
                                    continue;

                                // Get the full type string for spatial tag extraction
                                std::string fullTypeStr = ptype.getUnqualifiedType().getAsString();

                                // Check for spatial type wrapper: aie::tag_name<inner_type>
                                // Also handles aie::port<inner_type, PolicyName>
                                std::string spatialTag;
                                std::string innerTypeStr;
                                std::string policyName;
                                static const char *spatialTags[] = {
                                    "row_broadcast_in", "col_broadcast_in", "tiled_in", "row_major_out",
                                    "col_major_out",    "row_reduce_out",   "port"};
                                for (const char *tag : spatialTags) {
                                    std::string prefix = std::string("aie::") + tag + "<";
                                    size_t pos = fullTypeStr.find(prefix);
                                    if (pos != std::string::npos) {
                                        // Extract everything between < and matching >
                                        size_t innerStart = pos + prefix.size();
                                        int depth = 1;
                                        size_t innerEnd = innerStart;
                                        while (innerEnd < fullTypeStr.size() && depth > 0) {
                                            if (fullTypeStr[innerEnd] == '<')
                                                depth++;
                                            else if (fullTypeStr[innerEnd] == '>')
                                                depth--;
                                            if (depth > 0)
                                                innerEnd++;
                                        }
                                        std::string templateArgs =
                                            fullTypeStr.substr(innerStart, innerEnd - innerStart);

                                        if (std::string(tag) == "port") {
                                            // aie::port<InnerType, PolicyName>
                                            // Split on top-level comma to get inner type and policy name
                                            int commaDepth = 0;
                                            size_t commaPos = std::string::npos;
                                            for (size_t ci = 0; ci < templateArgs.size(); ++ci) {
                                                if (templateArgs[ci] == '<')
                                                    commaDepth++;
                                                else if (templateArgs[ci] == '>')
                                                    commaDepth--;
                                                else if (templateArgs[ci] == ',' && commaDepth == 0) {
                                                    commaPos = ci;
                                                    break;
                                                }
                                            }
                                            if (commaPos != std::string::npos) {
                                                innerTypeStr = templateArgs.substr(0, commaPos);
                                                policyName = templateArgs.substr(commaPos + 1);
                                                // Trim whitespace
                                                while (!policyName.empty() && policyName.front() == ' ')
                                                    policyName.erase(0, 1);
                                                while (!policyName.empty() && policyName.back() == ' ')
                                                    policyName.pop_back();
                                                // Strip "aie::" namespace prefix if present
                                                if (policyName.substr(0, 5) == "aie::")
                                                    policyName = policyName.substr(5);
                                            } else {
                                                // aie::port<InnerType> with default policy
                                                innerTypeStr = templateArgs;
                                                policyName = "RowBC"; // default policy
                                            }
                                            // Don't set spatialTag for port — policyName is used instead
                                            llvm::outs() << "[TilingLinalg] Policy port: " << policyName
                                                         << " -> inner type: " << innerTypeStr << "\n";
                                        } else {
                                            spatialTag = tag;
                                            innerTypeStr = templateArgs;
                                            llvm::outs() << "[TilingLinalg] Spatial tag: " << spatialTag
                                                         << " -> inner type: " << innerTypeStr << "\n";
                                        }

                                        // Trim whitespace from innerTypeStr
                                        while (!innerTypeStr.empty() && innerTypeStr.front() == ' ')
                                            innerTypeStr.erase(0, 1);
                                        while (!innerTypeStr.empty() && innerTypeStr.back() == ' ')
                                            innerTypeStr.pop_back();
                                        // Remove trailing * if present (the pointer is inside the wrapper)
                                        while (!innerTypeStr.empty() && innerTypeStr.back() == '*') {
                                            innerTypeStr.pop_back();
                                            while (!innerTypeStr.empty() && innerTypeStr.back() == ' ')
                                                innerTypeStr.pop_back();
                                        }
                                        break;
                                    }
                                }

                                // For bare pointer types (no spatial tag or policy), use the pointee type string
                                if (spatialTag.empty() && policyName.empty() && ptype->isPointerType()) {
                                    clang::QualType pointee = ptype->getPointeeType();
                                    innerTypeStr = pointee.getUnqualifiedType().getAsString();
                                }

                                // Determine input/output and bit width from inner type string
                                bool isWindowParam = false;
                                bool isInputWindow = false;
                                int bitWidth = 32; // default

                                if (innerTypeStr.find("input_window") != std::string::npos) {
                                    isWindowParam = true;
                                    isInputWindow = true;
                                } else if (innerTypeStr.find("output_window") != std::string::npos) {
                                    isWindowParam = true;
                                    isInputWindow = false;
                                }

                                if (isWindowParam) {
                                    if (innerTypeStr.find("int8") != std::string::npos ||
                                        innerTypeStr.find("signed char") != std::string::npos ||
                                        innerTypeStr.find("char") != std::string::npos)
                                        bitWidth = 8;
                                    else if (innerTypeStr.find("int16") != std::string::npos ||
                                             innerTypeStr.find("short") != std::string::npos)
                                        bitWidth = 16;
                                } else if (ptype->isPointerType()) {
                                    // Plain pointer type (e.g. const int32_t*)
                                    clang::QualType pointee = ptype->getPointeeType();
                                    if (pointee->isSpecificBuiltinType(clang::BuiltinType::Int))
                                        bitWidth = 32;
                                    else if (pointee->isSpecificBuiltinType(clang::BuiltinType::Short))
                                        bitWidth = 16;
                                    else if (pointee->isSpecificBuiltinType(clang::BuiltinType::SChar) ||
                                             pointee->isSpecificBuiltinType(clang::BuiltinType::Char_S) ||
                                             pointee->isSpecificBuiltinType(clang::BuiltinType::UChar))
                                        bitWidth = 8;
                                    else if (const auto *BT = pointee->getAs<clang::BuiltinType>()) {
                                        bitWidth = Context->getTypeSize(BT);
                                    }
                                }

                                // Determine input/output from spatial tag, policy name, or type info
                                bool isInput;
                                // Determine input/output: always prefer inner type info
                                if (isWindowParam) {
                                    isInput = isInputWindow;
                                } else if (!spatialTag.empty()) {
                                    isInput = (spatialTag.find("_in") != std::string::npos);
                                } else if (ptype->isPointerType()) {
                                    isInput = ptype->getPointeeType().isConstQualified();
                                } else {
                                    isInput = true; // conservative default
                                }

                                ParsedTensorInfo pti;
                                pti.varName = kp->getNameAsString();
                                // Assign per-tensor GEMM shapes from M/N/K macros.
                                // Shape is initially set to defaults; re-assigned after AST
                                // resolution using resolved policy fields (pattern+distribution).
                                if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    bool shapeAssigned = false;
                                    if (!spatialTag.empty()) {
                                        if (spatialTag == "row_broadcast_in") {
                                            pti.shape = {macroDimM, macroDimK};
                                            shapeAssigned = true;
                                        } else if (spatialTag == "col_broadcast_in") {
                                            pti.shape = {macroDimK, macroDimN};
                                            shapeAssigned = true;
                                        } else if (spatialTag == "row_major_out" || spatialTag == "col_major_out") {
                                            pti.shape = {macroDimM, macroDimN};
                                            shapeAssigned = true;
                                        }
                                    }
                                    if (!shapeAssigned)
                                        pti.shape = {defaultDim0, defaultDim1};
                                } else {
                                    pti.shape = {defaultDim0, defaultDim1};
                                }
                                pti.elementBitWidth = bitWidth;
                                pti.isInput = isInput;
                                pti.spatialTag = spatialTag;
                                pti.policyName = policyName;

                                // --- AST-based SpatialPolicy struct extraction ---
                                // Extract the constexpr SpatialPolicy struct fields from the
                                // template argument of aie::port<T, Policy>.
                                //
                                // After template instantiation, ptype is a RecordType (the
                                // instantiated aie::port<T,P> struct). We reach the template
                                // args via ClassTemplateSpecializationDecl.
                                //
                                // The policy argument kind depends on the LLVM version:
                                //   - StructuralValue: the APValue is stored directly
                                //   - Declaration: a reference to the constexpr VarDecl;
                                //     we evaluate it to get the APValue
                                if (!policyName.empty()) {
                                    const clang::Type *rawType = ptype.getTypePtr();
                                    rawType = rawType->getUnqualifiedDesugaredType();

                                    const clang::CXXRecordDecl *recDecl = rawType->getAsCXXRecordDecl();
                                    if (auto *ctsd =
                                            dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(recDecl)) {
                                        const auto &targs = ctsd->getTemplateArgs();
                                        if (targs.size() >= 2) {
                                            const auto &policyArg = targs[1];
                                            const APValue *apval = nullptr;

                                            if (policyArg.getKind() == clang::TemplateArgument::StructuralValue) {
                                                apval = &policyArg.getAsStructuralValue();
                                            } else if (policyArg.getKind() == clang::TemplateArgument::Declaration) {
                                                // C++20 struct NTTP: Clang stores the evaluated value as a
                                                // TemplateParamObjectDecl (a unique'd APValue holder).
                                                ValueDecl *decl = policyArg.getAsDecl();
                                                if (auto *tpo = dyn_cast<clang::TemplateParamObjectDecl>(decl)) {
                                                    apval = &tpo->getValue();
                                                } else if (auto *vd = dyn_cast<clang::VarDecl>(decl)) {
                                                    apval = vd->getEvaluatedValue();
                                                    if (!apval)
                                                        vd->evaluateValue(), apval = vd->getEvaluatedValue();
                                                }
                                            } else if (policyArg.getKind() == clang::TemplateArgument::Expression) {
                                                // Fallback: evaluate the expression directly
                                                clang::Expr::EvalResult evalRes;
                                                if (policyArg.getAsExpr()->EvaluateAsConstantExpr(evalRes, *Context)) {
                                                    apval = &evalRes.Val;
                                                }
                                            }

                                            if (apval && apval->isStruct() && apval->getStructNumFields() >= 4) {
                                                pti.pattern = (int)apval->getStructField(0).getInt().getExtValue();
                                                pti.distribution = (int)apval->getStructField(1).getInt().getExtValue();
                                                pti.mergeOrder = (int)apval->getStructField(2).getInt().getExtValue();
                                                pti.pingPong = (int)apval->getStructField(3).getInt().getExtValue();
                                                if (apval->getStructNumFields() >= 5)
                                                    pti.maxBufferBytes =
                                                        (int)apval->getStructField(4).getInt().getExtValue();
                                                // Two-level tiling hints (fields 5,6,7)
                                                if (apval->getStructNumFields() >= 6)
                                                    pti.tileM = (int)apval->getStructField(5).getInt().getExtValue();
                                                if (apval->getStructNumFields() >= 7)
                                                    pti.tileN = (int)apval->getStructField(6).getInt().getExtValue();
                                                if (apval->getStructNumFields() >= 8)
                                                    pti.tileK = (int)apval->getStructField(7).getInt().getExtValue();
                                                pti.policyResolved = true;
                                                llvm::outs()
                                                    << "[TilingLinalg] Policy resolved: pattern=" << pti.pattern
                                                    << " distribution=" << pti.distribution
                                                    << " mergeOrder=" << pti.mergeOrder << " ppDepth=" << pti.pingPong
                                                    << " maxBufferBytes=" << pti.maxBufferBytes;
                                                if (pti.tileM > 0 || pti.tileN > 0 || pti.tileK > 0)
                                                    llvm::outs() << " tile_m=" << pti.tileM << " tile_n=" << pti.tileN
                                                                 << " tile_k=" << pti.tileK;
                                                llvm::outs() << "\n";
                                            } else if (!apval) {
                                                llvm::errs()
                                                    << "[TilingLinalg] DEBUG: policy arg kind="
                                                    << (int)policyArg.getKind() << " — could not obtain APValue\n";
                                            }
                                        }
                                    }
                                    if (!pti.policyResolved) {
                                        llvm::errs()
                                            << "[TilingLinalg] ERROR: Failed to resolve constexpr SpatialPolicy '"
                                            << policyName << "' from AST\n";
                                    }
                                }

                                // Re-assign shape from resolved policy fields
                                if (pti.policyResolved && macroDimM > 0 && macroDimN > 0 && macroDimK > 0) {
                                    if (pti.pattern == 0 && pti.distribution == 0) // Broadcast+Row -> A
                                        pti.shape = {macroDimM, macroDimK};
                                    else if (pti.pattern == 0 && pti.distribution == 1) // Broadcast+Col -> B
                                        pti.shape = {macroDimK, macroDimN};
                                    else // Gather/Scatter -> C
                                        pti.shape = {macroDimM, macroDimN};
                                }

                                parsedTensors.push_back(pti);
                                std::string tagInfo;
                                if (!policyName.empty())
                                    tagInfo = " policy=" + policyName;
                                else if (!spatialTag.empty())
                                    tagInfo = " spatial=" + spatialTag;
                                llvm::outs() << "[TilingLinalg] Tensor param: " << pti.varName << " [" << pti.shape[0]
                                             << "x" << pti.shape[1] << "] i" << pti.elementBitWidth
                                             << (pti.isInput ? " (input)" : " (output)") << tagInfo << "\n";
                            }
						}
					}

                    // ---- Compute derived tiling parameters from M/K/N + per-port policy ----
                    int effectiveMeshRows = localMeshRows > 0 ? localMeshRows : tilingMeshRows;
                    int effectiveMeshCols = localMeshCols > 0 ? localMeshCols : tilingMeshCols;
                    if (macroDimM > 0 && macroDimN > 0 && macroDimK > 0 && effectiveMeshRows > 0 &&
                        effectiveMeshCols > 0 && !parsedTensors.empty()) {
                        derivedTilingParams.tileRows = macroDimM / effectiveMeshRows;
                        derivedTilingParams.tileCols = macroDimN / effectiveMeshCols;
                        derivedTilingParams.kDim = macroDimK;

                        // ---- Two-level tiling: compute effective tile_m/tile_n/tile_k ----
                        // Memory budget from HW resource model (same for all current AIE generations)
                        auto hwResForMemCheck = makeResource("Gen2");
                        const int64_t AIE_DATA_MEM_BYTES = hwResForMemCheck->getTileMemoryBytes();
                        const int64_t AIE_USABLE_MEM_BYTES = hwResForMemCheck->getUsableDataBytes();

                        // Collect explicit tiling hints from any resolved policy
                        int explicitTileM = 0, explicitTileN = 0, explicitTileK = 0;
                        int elementBytes = 1; // default int8
                        for (const auto &pt : parsedTensors) {
                            if (pt.policyResolved) {
                                if (pt.tileM > 0)
                                    explicitTileM = pt.tileM;
                                if (pt.tileN > 0)
                                    explicitTileN = pt.tileN;
                                if (pt.tileK > 0)
                                    explicitTileK = pt.tileK;
                            }
                            int eb = pt.elementBitWidth / 8;
                            if (eb > elementBytes)
                                elementBytes = eb;
                        }

                        int64_t tileM_eff = explicitTileM > 0 ? explicitTileM : derivedTilingParams.tileRows;
                        int64_t tileN_eff = explicitTileN > 0 ? explicitTileN : derivedTilingParams.tileCols;
                        int64_t tileK_eff = explicitTileK > 0 ? explicitTileK : macroDimK;

                        // Memory check: A_local[tileM * tileK] + B_local[tileK * tileN] + C_local[tileM * tileN]
                        auto computeWorkingSet = [&](int64_t tm, int64_t tn, int64_t tk) -> int64_t {
                            return (tm * tk + tk * tn + tm * tn) * elementBytes;
                        };

                        int64_t workingSet = computeWorkingSet(tileM_eff, tileN_eff, tileK_eff);

                        if (explicitTileM > 0 || explicitTileN > 0 || explicitTileK > 0) {
                            // Explicit tiling — validate against memory budget
                            if (workingSet > AIE_DATA_MEM_BYTES) {
                                llvm::errs() << "[TilingLinalg] ERROR: Explicit tiling exceeds AIE data memory!\n"
                                             << "  tile_m=" << tileM_eff << " tile_n=" << tileN_eff
                                             << " tile_k=" << tileK_eff << " working_set=" << workingSet << " bytes"
                                             << " (limit=" << AIE_DATA_MEM_BYTES << ")\n"
                                             << "  A_local=" << tileM_eff * tileK_eff * elementBytes
                                             << " B_local=" << tileK_eff * tileN_eff * elementBytes
                                             << " C_local=" << tileM_eff * tileN_eff * elementBytes << "\n";
                            } else if (workingSet > AIE_USABLE_MEM_BYTES) {
                                llvm::outs() << "[TilingLinalg] WARNING: Explicit tiling uses " << workingSet
                                             << " bytes — close to limit, "
                                             << "little room for stack/ping-pong.\n";
                            }
                            derivedTilingParams.autoTiled = false;
                        } else if (workingSet > AIE_USABLE_MEM_BYTES) {
                            // Auto-tiling: working set exceeds memory budget, need to sub-tile
                            llvm::outs() << "[TilingLinalg] Working set " << workingSet
                                         << " bytes exceeds memory budget " << AIE_USABLE_MEM_BYTES
                                         << " — auto-tiling enabled.\n";

                            // Strategy: first try tiling K (cheapest — no spatial re-launch)
                            // Then reduce M/N if still too large
                            tileK_eff = tileK_eff; // start from full K
                            int64_t budget = AIE_USABLE_MEM_BYTES / elementBytes;

                            // Phase A: reduce K until A+B fit, keeping C stationary
                            // C_local = tileM * tileN (fixed), A_local = tileM * tileK, B_local = tileK * tileN
                            // Constraint: tileM * tileK + tileK * tileN + tileM * tileN <= budget
                            // => tileK * (tileM + tileN) <= budget - tileM * tileN
                            int64_t c_size = tileM_eff * tileN_eff;
                            if (c_size < budget) {
                                int64_t remaining = budget - c_size;
                                int64_t maxK = remaining / (tileM_eff + tileN_eff);
                                if (maxK < tileK_eff && maxK > 0) {
                                    // Round down to power of 2 for alignment
                                    int64_t k = 1;
                                    while (k * 2 <= maxK)
                                        k *= 2;
                                    tileK_eff = k;
                                }
                            }

                            // Phase B: if C_local alone exceeds budget, reduce M and N
                            if (c_size >= budget) {
                                // Solve: 2*tm*tk + tm^2 <= budget (assuming tm=tn, tk=tm for simplicity)
                                // Use heuristic: start with 64 and validate
                                for (int64_t candidate : {256, 128, 64, 32, 16, 8}) {
                                    tileM_eff = std::min((int64_t)candidate, derivedTilingParams.tileRows);
                                    tileN_eff = std::min((int64_t)candidate, derivedTilingParams.tileCols);
                                    // Re-derive tileK with new M/N
                                    c_size = tileM_eff * tileN_eff;
                                    if (c_size < budget) {
                                        int64_t remaining = budget - c_size;
                                        int64_t maxK = remaining / (tileM_eff + tileN_eff);
                                        int64_t k = 1;
                                        while (k * 2 <= maxK && k * 2 <= macroDimK)
                                            k *= 2;
                                        tileK_eff = k;
                                        if (computeWorkingSet(tileM_eff, tileN_eff, tileK_eff) <= AIE_USABLE_MEM_BYTES)
                                            break;
                                    }
                                }
                            }

                            // Final validation
                            workingSet = computeWorkingSet(tileM_eff, tileN_eff, tileK_eff);
                            if (workingSet > AIE_DATA_MEM_BYTES) {
                                llvm::errs() << "[TilingLinalg] ERROR: Auto-tiling failed to fit in memory!\n"
                                             << "  tile_m=" << tileM_eff << " tile_n=" << tileN_eff
                                             << " tile_k=" << tileK_eff << " working_set=" << workingSet << " bytes\n";
                            }
                            derivedTilingParams.autoTiled = true;
                        } else {
                            // Fits in memory without sub-tiling — use full tile dimensions
                            derivedTilingParams.autoTiled = false;
                        }

                        // Compute spatial rounds and K rounds
                        derivedTilingParams.tileM = tileM_eff;
                        derivedTilingParams.tileN = tileN_eff;
                        derivedTilingParams.effectiveK = tileK_eff;
                        derivedTilingParams.spatialMRounds = (tileM_eff > 0 && tileM_eff < derivedTilingParams.tileRows)
                                                                 ? derivedTilingParams.tileRows / tileM_eff
                                                                 : 1;
                        derivedTilingParams.spatialNRounds = (tileN_eff > 0 && tileN_eff < derivedTilingParams.tileCols)
                                                                 ? derivedTilingParams.tileCols / tileN_eff
                                                                 : 1;
                        derivedTilingParams.kRounds =
                            (tileK_eff > 0 && tileK_eff < macroDimK) ? macroDimK / tileK_eff : 1;

                        // ---- Compute per-port DMA round/buffer parameters ----
                        // Use effectiveK for DMA buffer sizing (temporal tiling of K)
                        int64_t dmaK = derivedTilingParams.effectiveK;

                        for (auto &pt : parsedTensors) {
                            DerivedTilingParams::PortParams pp;
                            int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                            int maxBuf = pt.maxBufferBytes > 0 ? pt.maxBufferBytes : 4096;

                            if (pt.isInput) {
                                if (pt.policyResolved) {
                                    if (pt.pattern == 0 && pt.distribution == 0) {
                                        // Input A: Broadcast+Row — tileM rows × effectiveK per k-round
                                        // pp_depth only controls physical ping-pong buffer count,
                                        // NOT data splitting. Only split when data exceeds maxBuf.
                                        int64_t perKRoundData = tileM_eff * dmaK;
                                        if (perKRoundData <= maxBuf) {
                                            pp.bufferSize = perKRoundData;
                                            pp.numRounds = 1;
                                        } else {
                                            int64_t rowsPerRound = maxBuf / dmaK;
                                            if (rowsPerRound <= 0)
                                                rowsPerRound = 1;
                                            pp.bufferSize = rowsPerRound * dmaK;
                                            pp.numRounds = (tileM_eff + rowsPerRound - 1) / rowsPerRound;
                                            llvm::outs() << "[TilingLinalg] WARNING: Input A tile_m(" << tileM_eff
                                                         << ") * effectiveK(" << dmaK << ") = " << perKRoundData
                                                         << " exceeds max_buffer_bytes(" << maxBuf
                                                         << "), splitting into " << pp.numRounds << " rounds\n";
                                        }
                                    } else if (pt.pattern == 0 && pt.distribution == 1) {
                                        // Input B: Broadcast+Col — tileN cols × effectiveK per k-round
                                        int64_t perKRoundData = tileN_eff * dmaK;
                                        if (perKRoundData <= maxBuf) {
                                            pp.bufferSize = perKRoundData;
                                            pp.numRounds = 1;
                                        } else {
                                            int64_t colsPerRound = maxBuf / dmaK;
                                            if (colsPerRound <= 0)
                                                colsPerRound = 1;
                                            pp.bufferSize = colsPerRound * dmaK;
                                            pp.numRounds = (tileN_eff + colsPerRound - 1) / colsPerRound;
                                            llvm::outs() << "[TilingLinalg] WARNING: Input B tile_n(" << tileN_eff
                                                         << ") * effectiveK(" << dmaK << ") = " << perKRoundData
                                                         << " exceeds max_buffer_bytes(" << maxBuf
                                                         << "), splitting into " << pp.numRounds << " rounds\n";
                                        }
                                    } else {
                                        // Other input patterns: default
                                        int64_t totalElements = 1;
                                        for (auto d : pt.shape)
                                            totalElements *= d;
                                        int64_t perTile = totalElements / (tilingMeshRows * tilingMeshCols);
                                        if (perTile <= maxBuf) {
                                            pp.bufferSize = perTile;
                                            pp.numRounds = 1;
                                        } else {
                                            pp.bufferSize = maxBuf;
                                            pp.numRounds = (perTile + maxBuf - 1) / maxBuf;
                                        }
                                    }
                                } else {
                                    // Unresolved policy — use defaults
                                    int64_t totalElements = pt.shape[0] * pt.shape[1] / effectiveMeshRows;
                                    if (totalElements <= maxBuf) {
                                        pp.bufferSize = totalElements;
                                        pp.numRounds = 1;
                                    } else {
                                        pp.bufferSize = maxBuf;
                                        pp.numRounds = (totalElements + maxBuf - 1) / maxBuf;
                                    }
                                }
                            } else {
                                // Output: use tileM * tileN as per-core output
                                // pp_depth only controls physical buffer count, not data splitting
                                int64_t outputPerCore = tileM_eff * tileN_eff;
                                if (outputPerCore <= maxBuf) {
                                    pp.bufferSize = outputPerCore;
                                    pp.numRounds = 1;
                                } else {
                                    pp.bufferSize = maxBuf;
                                    pp.numRounds = (outputPerCore + maxBuf - 1) / maxBuf;
                                }
                            }
                            derivedTilingParams.portParams.push_back(pp);
                        }
                        derivedTilingParams.valid = true;
                        llvm::outs() << "[TilingLinalg] Derived tiling: tileRows=" << derivedTilingParams.tileRows
                                     << " tileCols=" << derivedTilingParams.tileCols
                                     << " kDim=" << derivedTilingParams.kDim << "\n";
                        llvm::outs() << "[TilingLinalg] Two-level tiling: tileM=" << derivedTilingParams.tileM
                                     << " tileN=" << derivedTilingParams.tileN
                                     << " effectiveK=" << derivedTilingParams.effectiveK
                                     << " kRounds=" << derivedTilingParams.kRounds
                                     << " spatialMRounds=" << derivedTilingParams.spatialMRounds
                                     << " spatialNRounds=" << derivedTilingParams.spatialNRounds
                                     << (derivedTilingParams.autoTiled ? " (auto)" : " (explicit/default)") << "\n";
                        llvm::outs() << "[TilingLinalg] Memory estimate: "
                                     << computeWorkingSet(tileM_eff, tileN_eff, tileK_eff) << " bytes"
                                     << " (A=" << tileM_eff * tileK_eff * elementBytes
                                     << " B=" << tileK_eff * tileN_eff * elementBytes
                                     << " C=" << tileM_eff * tileN_eff * elementBytes << ")\n";
                        for (size_t i = 0; i < derivedTilingParams.portParams.size(); ++i) {
                            llvm::outs() << "[TilingLinalg]   port[" << i
                                         << "]: numRounds=" << derivedTilingParams.portParams[i].numRounds
                                         << " bufferSize=" << derivedTilingParams.portParams[i].bufferSize << "\n";
                        }

                        // Check: total ping-pong buffer requirement vs tile data memory
                        {
                            uint32_t totalBufBytes = 0;
                            for (size_t i = 0; i < derivedTilingParams.portParams.size(); i++) {
                                auto &pp = derivedTilingParams.portParams[i];
                                auto &pt = parsedTensors[i];
                                int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                                int elemBytes = pt.elementBitWidth / 8;
                                if (elemBytes <= 0)
                                    elemBytes = 1;
                                totalBufBytes += ppDepth * pp.bufferSize * elemBytes;
                            }
                            std::string errMsg;
                            if (!hwResForMemCheck->checkDataMemoryFits(totalBufBytes, &errMsg)) {
                                llvm::errs() << "[TilingLinalg] ERROR: " << errMsg << "\n";
                                for (size_t i = 0; i < derivedTilingParams.portParams.size(); i++) {
                                    auto &pp = derivedTilingParams.portParams[i];
                                    auto &pt = parsedTensors[i];
                                    int ppDepth = pt.pingPong > 0 ? pt.pingPong : 2;
                                    int elemBytes = pt.elementBitWidth / 8;
                                    if (elemBytes <= 0)
                                        elemBytes = 1;
                                    llvm::errs() << "  Port " << i << " (" << pt.varName << "): " << ppDepth << " x "
                                                 << pp.bufferSize << " x " << elemBytes << " = "
                                                 << ppDepth * pp.bufferSize * elemBytes << " bytes\n";
                                }
                            } else {
                                llvm::outs() << "[TilingLinalg] Buffer memory check passed: " << totalBufBytes << " / "
                                             << hwResForMemCheck->getUsableDataBytes() << " bytes\n";
                            }
                        }

                        // K-round shape adjustment removed: function args now keep full
                        // DDR tensor shapes ({M,K}/{K,N}/{M,N}).  K-round slicing is
                        // handled entirely by module attributes (routing.effective_k,
                        // routing.full_k, routing.k_rounds) consumed by downstream passes.
                    }

                    // ---- Store MeshKernelDesc for multi-kernel support ----
                    {
                        MeshKernelDesc mkd;
                        mkd.kernelName = currentLaunchKernel;
                        mkd.kernelFuncName = currentLaunchKernel;
                        mkd.meshRows = localMeshRows > 0 ? localMeshRows : tilingMeshRows;
                        mkd.meshCols = localMeshCols > 0 ? localMeshCols : tilingMeshCols;
                        mkd.partition = localPartition.isValid() ? localPartition : parsedPartition;
                        mkd.meshId = (int)parsedMeshKernels.size();
                        for (auto &pt : parsedTensors) {
                            mkd.tensors.push_back({pt.shape, pt.elementBitWidth, pt.isInput});
                        }
                        // Store per-kernel body from globalKernelBodies map
                        auto bodyIt = globalKernelBodies.find(currentLaunchKernel);
                        if (bodyIt != globalKernelBodies.end()) {
                            mkd.kernelBody = bodyIt->second;
                        }
                        // Build per-kernel SplitModel from parsedTensors
                        for (auto &pt : parsedTensors) {
                            if (pt.policyResolved) {
                                mkd.splitModel.tensorSplits.push_back(
                                    SplitModel::fromPolicyFields(pt.pattern, pt.distribution, pt.mergeOrder,
                                                                 pt.pingPong, pt.isInput, pt.maxBufferBytes));
                            } else if (!pt.spatialTag.empty()) {
                                mkd.splitModel.tensorSplits.push_back(
                                    SplitModel::fromSpatialTag(pt.spatialTag, pt.isInput));
                            } else {
                                mkd.splitModel.tensorSplits.push_back(SplitModel::fromPolicyFields(
                                    pt.isInput ? 0 : 3, 0, pt.isInput ? 0 : 1, 2, pt.isInput, pt.maxBufferBytes));
                            }
                        }
                        // Store per-kernel maxPPBytes (minimum across ports)
                        if (!parsedTensors.empty()) {
                            mkd.maxPPBytes = parsedTensors[0].maxBufferBytes;
                            for (auto &pt : parsedTensors) {
                                if (pt.maxBufferBytes > 0 && pt.maxBufferBytes < mkd.maxPPBytes)
                                    mkd.maxPPBytes = pt.maxBufferBytes;
                            }
                        }
                        // Store per-kernel derivedTilingParams and port var names
                        // for aie::get_*() replacement in multi-kernel mode
                        mkd.derivedParams = derivedTilingParams;
                        for (auto &pt : parsedTensors) {
                            mkd.portVarNames.push_back(pt.varName);
                        }
                        parsedMeshKernels.push_back(mkd);
                        llvm::outs() << "[TilingLinalg] Registered MeshKernelDesc: kernel=" << mkd.kernelName
                                     << " mesh=" << mkd.meshRows << "x" << mkd.meshCols << " meshId=" << mkd.meshId
                                     << "\n";
                    }
                }
				return true;
			}

			if (Callee && Callee->getNameAsString() =="XAie_LoadElfMem") {
                if (CE->getNumArgs() >= 3) {
                    const Expr *devExpr = CE->getArg(0);
                    const Expr *tileExpr = CE->getArg(1);
                    const Expr *kernelExpr = CE->getArg(2)->IgnoreParenImpCasts();

                    std::string devStr;
                    llvm::raw_string_ostream devOs(devStr);
                    devExpr->printPretty(devOs, nullptr, PrintingPolicy(LangOptions()));
                    devOs.flush();

                    std::string tileStr;
                    llvm::raw_string_ostream tileOs(tileStr);
                    tileExpr->printPretty(tileOs, nullptr, PrintingPolicy(LangOptions()));
                    tileOs.flush();

                    std::string kernelArgStr;
                    llvm::raw_string_ostream kernelArgOs(kernelArgStr);
                    kernelExpr->printPretty(kernelArgOs, nullptr, PrintingPolicy(LangOptions()));
                    kernelArgOs.flush();
                    boost::algorithm::trim(kernelArgStr);

                    if (kernelArgStr.find("_binary_kernel_") != std::string::npos) {
                        return true;
                    }

                    const Expr *kernelNameExpr = kernelExpr;
                    while (const auto *castExpr = dyn_cast<CastExpr>(kernelNameExpr)) {
                        kernelNameExpr = castExpr->getSubExpr()->IgnoreParenImpCasts();
                    }
                    if (const auto *unaryExpr = dyn_cast<UnaryOperator>(kernelNameExpr)) {
                        if (unaryExpr->getOpcode() == UO_AddrOf) {
                            kernelNameExpr = unaryExpr->getSubExpr()->IgnoreParenImpCasts();
                        }
                    }

                    std::string kernelName;
                    if (const auto *declRef = dyn_cast<DeclRefExpr>(kernelNameExpr)) {
                        kernelName = declRef->getDecl()->getNameAsString();
                    } else {
                        return true;
                    }

                    std::string replacement = "XAie_LoadElfMem(" + devStr + ", " + tileStr +
                                              ", (unsigned char *)_binary_kernel_" + kernelName + "_start)";
                    clang::SourceRange callExprRange = CE->getSourceRange();
                    Rewrite->ReplaceText(callExprRange, replacement);
                }
            } else {
					if (Callee) {
						for (auto x:kernel_name_list) {
							// std::cout << "kernel name is (" << x << ")" <<std::endl;
							if (Callee->getNameAsString() == x) {
								llvm::outs()<< "**************Function name is " << Callee->getNameInfo().getName().getAsString() <<"*********8\n";
							
								if (auto FD = CE->getDirectCallee()) {

									// 2) Check if it’s actually a specialization of a function template:
									if (auto  TSI = FD->getTemplateSpecializationInfo()) {
								
										// 3) Grab the FunctionTemplateDecl (the “template name”):
										auto FTD = FD->getPrimaryTemplate();
										llvm::outs() << "Template name: "
													<< FTD->getNameAsString() << "\n";
									
										// 4) Extract the explicit template-argument list:
										ArrayRef<TemplateArgument> Args = 
											TSI->TemplateArguments->asArray();
									
										llvm::outs() << "Template args: <";
										std::vector<std::string> knames;
										for (size_t i = 0; i < Args.size(); ++i) {
											const TemplateArgument &Arg = Args[i];
											llvm::SmallString<16> Buf;
											switch (Arg.getKind()) {
											case TemplateArgument::Integral:
												
												Arg.getAsIntegral().toString(Buf, 10);
												llvm::outs() << Buf;
												//std::string kname(Buf.begin(), Buf.end());
												//kname = Buf.str().str();
												knames.push_back(Buf.str().str());
												break;
											case TemplateArgument::Type:
												llvm::outs() << Arg.getAsType().getAsString();
												break;
											default:
												llvm::outs() << "(other)";
											}
											if (i + 1 < Args.size()) llvm::outs() << ", ";
										}
										llvm::outs() << ">\n\n";
										Aiefrontend->createKernelFunction(knames);
									}
								}
								
							}
						}
					} else {
						auto Call = CE->getCallee();
						if (Call) {
							//llvm::errs() << "Indirect function call or function pointer dereference\n";
							//Call->dump();  // You can inspect the callee expression
						}
					}
            // If Callee is null, it may be an indirect function call or a function pointer.
            // You can handle those cases accordingly.
            //llvm::outs() << "Indirect function call or function pointer\n";
			}
			return true;
    }
/*
		 bool VisitParmVarDecl(ParmVarDecl *PVD) {
        // Handle function parameters
        if (!PVD->hasInit()) {
          // Parameter does not have an initializer
          llvm::outs() << "Parameter without initializer: " << PVD->getNameAsString() << "\n";
        }
        return true;
      }

      bool VisitVarDecl(VarDecl *VD) {
        // Handle local variables
        if (!VD->hasInit()) {
          // Variable does not have an initializer
          llvm::outs() << "Variable without initializer: " << VD->getNameAsString() << "\n";
        }
        return true;
     }
		  bool VisitDecl(Decl *D) {
        // Check for "unknown type name 'windows'" error
				//std::cout << "unknow VisitDecl" <<std::endl;
				//return false;
				std::cout << "Decl is " << D->getDeclKindName() << std::endl;
        if (isa<UnresolvedUsingTypenameDecl>(D)) {
        }

        return true;
      }
			*/

		 void PrintCode() {
    	// Apply the rewrite to the source file
    	//Rewrite.commit(rewriter::Rewriter::Commit);
			//std::string RewrittenCode = Rewrite.getEditBuffer(Rewrite.getSourceMgr().getMainFileID()).getBuffer();
			 std::cout << "---------****---------" << std::endl;
			 ///*
			 const RewriteBuffer *RewriteBuf = Rewrite->getRewriteBufferFor(Rewrite->getSourceMgr().getMainFileID());
			 if (RewriteBuf) {
        //llvm::outs() << "Rewritten Source Code:\n" << RewriteBuf << "\n";
			//*/
			 //std::ofstream fd("./new_t.c");
				 RewriteBuf->write(llvm::outs());
		 }
			 /*
			 auto& sc = Rewrite.getSourceMgr();
			 auto mid = sc.getMainFileID();
			 llvm::StringRef sourceCode = sc.getBufferData(mid);
			 llvm::outs() << "Original Source Code:\n" << sourceCode << "\n";
			 */
			 std::cout << "------------------" << std::endl;
			//llvm::outs() << "Rewritten Source Code:\n" << RewrittenCode << "\n";
  	}


private:
		AieFrontEnd* Aiefrontend;
    ASTContext *Context;
		Rewriter *Rewrite;
};

class MyASTConsumer : public ASTConsumer {
public:
    //explicit MyASTConsumer(ASTContext *Context)
      //  : Visitor(Context) {}

    explicit MyASTConsumer(Rewriter* TheRewriter, ASTContext *Context, AieFrontEnd* Aiefrontend)
        : Visitor(TheRewriter, Context, Aiefrontend) {}

    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
		// std::cout << __FUNCTION__ << " get called " << std::endl;
    }

private:
    GlobalFunctionVisitor Visitor;
};

class AieDebugLevelPragmaHandler : public clang::PragmaHandler {
  public:
    AieDebugLevelPragmaHandler() : PragmaHandler("aie_debug_level") {}

    void HandlePragma(clang::Preprocessor &PP, clang::PragmaIntroducer Introducer, clang::Token &FirstToken) override {
        // Known flag macros (must match aie_runtime.h definitions)
        static const std::unordered_map<std::string, int> knownFlags = {
            {"AIE_DEBUG_FLAG_DISABLE_MULTID_DIM_DMA", 1 << 4},
            {"AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN", 1 << 5},
            {"AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER", 1 << 6},
        };

        clang::Token Tok;
        PP.Lex(Tok);

        // Skip optional opening paren: #pragma aie_debug_level(...)
        bool hasParen = Tok.is(clang::tok::l_paren);
        if (hasParen)
            PP.Lex(Tok);

        int result = 0;
        bool valid = false;

        // Parse: term ( '|' term )*
        // Each term is a numeric_constant or a known identifier flag
        while (true) {
            if (Tok.is(clang::tok::numeric_constant)) {
                llvm::SmallString<16> IntegerBuffer;
                bool Invalid = false;
                llvm::StringRef Spelling = PP.getSpelling(Tok, IntegerBuffer, &Invalid);
                if (Invalid)
                    break;
                int val = 0;
                // Support decimal, hex (0x...), octal (0...) via base 0
                if (Spelling.getAsInteger(0, val))
                    break;
                result |= val;
                valid = true;
            } else if (Tok.is(clang::tok::identifier)) {
                std::string name = PP.getSpelling(Tok);
                auto it = knownFlags.find(name);
                if (it == knownFlags.end()) {
                    llvm::errs() << "[aiehlc] Warning: unknown flag '" << name
                                 << "' in #pragma aie_debug_level, ignored\n";
                } else {
                    result |= it->second;
                    valid = true;
                }
            } else {
                break;
            }

            PP.Lex(Tok);
            if (Tok.is(clang::tok::pipe)) {
                PP.Lex(Tok); // consume '|', continue to next term
            } else {
                break;
            }
        }

        // Consume optional closing paren
        if (hasParen && Tok.is(clang::tok::r_paren))
            PP.Lex(Tok);

        if (valid) {
            parsedDebugLevel = result;
            llvm::outs() << "[aiehlc] Detected #pragma aie_debug_level " << parsedDebugLevel << "\n";
        }

        // Consume remaining tokens on the pragma line (if any)
        if (Tok.isNot(clang::tok::eod))
            PP.DiscardUntilEndOfDirective();
    }
};

class MyFrontendAction : public ASTFrontendAction {
public:
		MyFrontendAction() {
			FileMgr = std::make_unique<clang::FileManager>(clang::FileSystemOptions());
			aiefrontend.set_llvm_aie(use_llvm_aie);
		}
		void ReplaceGlobalKeyAndAddInclude(clang::SourceManager &SourceMgr) {
			std::unordered_map<std::string, std::string> replace_map={{"__global__","__attribute__((annotate(\"__global__\")))"},
																	  {"__kernel__","__attribute__((annotate(\"__kernel__\")))"}};
			//auto SourceCodeString = GetKeyReplaceAndAddInclude(SourceMgr, "__global__",
			//		"__attribute__((annotate(\"__global__\")))");
			auto SourceCodeString = GetKeyReplaceAndAddInclude(SourceMgr, replace_map);
			llvm::StringRef ModifiedCode = SourceCodeString;

            // Save a physical copy of the preprocessed file for debugging/inspection
            std::string newfilePath = std::string(AOUT) + "newfile.cpp";
            std::ofstream newfileOut(newfilePath, std::ios::out | std::ios::trunc);
            if (newfileOut.is_open()) {
                newfileOut << SourceCodeString;
                newfileOut.close();
                llvm::outs() << "Saved preprocessed file to: " << newfilePath << "\n";
            }

            std::unique_ptr<llvm::MemoryBuffer> ModifiedBuffer =
					 llvm::MemoryBuffer::getMemBufferCopy(ModifiedCode,"NewCode");
			// std::cout << "new ----------" << code << std::endl;
			// auto src = code.c_str();
			// std::cout << "new ---src-------" << src << std::endl;

			//const char* src = "__attribute__((annotate(\"__global__\"))) int func () {} \n int main() {}";
			//std::unique_ptr<llvm::MemoryBuffer> Buf = llvm::MemoryBuffer::getMemBuffer(src);
			const FileEntryRef SourceFile = FileMgr->getVirtualFileRef("newfile.cpp", ModifiedBuffer->getBufferSize(), 0);
			//FileEntryRef fileEntryRef = SourceMgr.translateFile(SourceFile);

			SourceMgr.overrideFileContents(SourceFile, std::move(ModifiedBuffer));
			FileID MainFileID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
			SourceMgr.setMainFileID(MainFileID);
		}
	 /* do pre-processing change the __global__ into annotation
	  */
	  bool BeginSourceFileAction(CompilerInstance &CI) override {
			//auto& ppopts = CI.getPreprocessorOpts();
			// std::cout <<"File is " << "Filename.str()" << std::endl;
 			clang::SourceManager &SourceMgr = CI.getSourceManager();

			ReplaceGlobalKeyAndAddInclude(SourceMgr);

            clang::Preprocessor &PP = CI.getPreprocessor();
            PP.AddPragmaHandler(new AieDebugLevelPragmaHandler());

            return true;
		}

		std::string GetKeyReplaceAndAddInclude(clang::SourceManager &SourceMgr, std::unordered_map<std::string,std::string>& funcMap) {
			std::string ret;
			// Get the main file entry
			const auto FileEntryRef = SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID());

			if (FileEntryRef) {
				// Get the file name
				llvm::StringRef FileName = FileEntryRef->getName();

                // Capture source directory for include path propagation
                {
                    std::string fnStr = FileName.str();
                    auto lastSlash = fnStr.rfind('/');
                    if (lastSlash != std::string::npos)
                        userSourceDir = fnStr.substr(0, lastSlash);
                    else
                        userSourceDir = ".";
                }

                // Print or use the file name as needed
				// llvm::outs() << "Processing file: " << FileName << "\n";
				clang::FileID MainFileID = SourceMgr.getMainFileID();

				// Get the source code buffer for the main file
				//const llvm::MemoryBuffer *MainFileBuffer = SourceMgr.getBuffer(MainFileID);
				const auto MainFileBufferOp = SourceMgr.getBufferOrNone(MainFileID);
				const llvm::MemoryBufferRef *MainFileBuffer = &MainFileBufferOp.value();
				// Get the source code as a StringRef
				llvm::StringRef SourceCode = MainFileBuffer->getBuffer();
				// std::cout << SourceCode.str() << std::endl;
				std::unique_ptr<llvm::MemoryBuffer> Buffer = llvm::MemoryBuffer::getMemBuffer(SourceCode, "", false);

				llvm::SourceMgr SM;
				SM.AddNewSourceBuffer(std::move(Buffer), llvm::SMLoc());
				//replace keyword with annotate
				//std::string KeywordToReplace = sOld;//"__global__";
				//std::string Replacement = sNew;//"__attribute__((annotate(\"__global__\")))";

				//llvm::StringRef UpdatedSource;
				std::string SourceCodeString = SourceCode.str();

                // Collect user #define macros before any rewriting
                {
                    userMacroDefines.clear();
                    std::istringstream iss(SourceCodeString);
                    std::string line;
                    std::string currentDefine;
                    bool inMultiLine = false;
                    while (std::getline(iss, line)) {
                        if (inMultiLine) {
                            currentDefine += "\n" + line;
                            if (line.empty() || line.back() != '\\') {
                                userMacroDefines.push_back(currentDefine);
                                inMultiLine = false;
                            }
                            continue;
                        }
                        size_t firstNonSpace = line.find_first_not_of(" \t");
                        if (firstNonSpace == std::string::npos)
                            continue;
                        std::string trimmed = line.substr(firstNonSpace);
                        if (trimmed.empty() || trimmed[0] != '#')
                            continue;
                        if (trimmed.substr(0, 8) == "#include")
                            continue;
                        if (line.find("AIEHLC_STUBS_DEFINED") != std::string::npos ||
                            line.find("AIEHLC_TILING_STUBS_DEFINED") != std::string::npos)
                            continue;
                        if (trimmed.substr(0, 7) == "#define" || trimmed.substr(0, 6) == "#undef" ||
                            trimmed.substr(0, 3) == "#if" || trimmed.substr(0, 5) == "#elif" ||
                            trimmed.substr(0, 5) == "#else" || trimmed.substr(0, 6) == "#endif") {
                            currentDefine = line;
                            if (!line.empty() && line.back() == '\\') {
                                inMultiLine = true;
                            } else {
                                userMacroDefines.push_back(currentDefine);
                            }
                        }
                    }
                }

                size_t KeywordPos = 0;
				for (auto x:funcMap ) {
					std::string KeywordToReplace = x.first;//"__global__";
					std::string Replacement = x.second;//"__attribute__((annotate(\"__global__\")))";
					size_t KeywordPos = 0;
					while ((KeywordPos =  SourceCodeString.find(KeywordToReplace, KeywordPos)) != std::string::npos) {
						int pos = KeywordPos;
						bool bgstr = false;
						// check whether the __global_ is inside a '"', actually just check the front '"'
						while(--pos >=0) {
							if (SourceCodeString[pos] == ' ') continue;
							if (SourceCodeString[pos] == '"' || SourceCodeString[pos] == '/' || SourceCodeString[pos] == '*') {
								bgstr = true;
							}
							break;
						}
						if (!bgstr) {
							//process the kernel logic
							auto curkeywordpos = KeywordPos;
							if (KeywordToReplace == "__kernel__") {
								auto bracepos = SourceCodeString.find("(", curkeywordpos);
								auto curpos = bracepos;
								while(--curpos >=0){
									if (SourceCodeString[curpos] == ' ') break;
								}
								curpos++;
							
								auto str = SourceCodeString.substr(curpos, bracepos - curpos);
								while(str[str.size() -1] == ' ') {
									str = str.substr(0, str.size() - 1);
								}
								kernel_name_list.push_back(str);
							}
							//replace
							SourceCodeString.replace(KeywordPos, KeywordToReplace.size(), Replacement);
							KeywordPos += Replacement.size();

                            // For __global__ functions, wrap body with #ifdef KERNEL_COMPILE
                            if (KeywordToReplace == "__global__") {
                                // Find the opening brace of the function body
                                auto openBracePos = SourceCodeString.find("{", KeywordPos);
                                if (openBracePos != std::string::npos) {
                                    // Insert #ifdef KERNEL_COMPILE after opening brace
                                    std::string ifdefGuard = "\n#ifdef KERNEL_COMPILE\n";
                                    SourceCodeString.insert(openBracePos + 1, ifdefGuard);

                                    // Find matching closing brace
                                    int braceCount = 1;
                                    size_t searchPos = openBracePos + 1 + ifdefGuard.size();
                                    while (braceCount > 0 && searchPos < SourceCodeString.size()) {
                                        if (SourceCodeString[searchPos] == '{')
                                            braceCount++;
                                        else if (SourceCodeString[searchPos] == '}')
                                            braceCount--;
                                        if (braceCount > 0)
                                            searchPos++;
                                    }
                                    // Insert #endif before closing brace
                                    if (braceCount == 0) {
                                        std::string endifGuard = "\n#endif\n";
                                        SourceCodeString.insert(searchPos, endifGuard);
                                    }
                                }
                            }

                        } else {
							KeywordPos += KeywordToReplace.size();
						}
					}
				}
				// Transform <<<mesh>>> kernel launch syntax (CUDA-style)
				// Pattern: funcname<<<varname>>>(args) -> __aie_launch("funcname", varname, args)
				{
					size_t launchPos = 0;
					while ((launchPos = SourceCodeString.find("<<<", launchPos)) != std::string::npos) {
						// Find the function name before <<<
						size_t funcEnd = launchPos;
						while (funcEnd > 0 && (std::isalnum(SourceCodeString[funcEnd-1]) || SourceCodeString[funcEnd-1] == '_'))
							funcEnd--;
						std::string funcName = SourceCodeString.substr(funcEnd, launchPos - funcEnd);

						// Find mesh variable name between <<< and >>>
						size_t meshStart = launchPos + 3;
						size_t meshEnd = SourceCodeString.find(">>>", meshStart);
						if (meshEnd == std::string::npos) break;
						std::string meshVar = SourceCodeString.substr(meshStart, meshEnd - meshStart);
						// trim whitespace
						meshVar.erase(0, meshVar.find_first_not_of(" \t"));
						meshVar.erase(meshVar.find_last_not_of(" \t") + 1);

						// Find args between ( and )
						size_t argsOpenParen = SourceCodeString.find("(", meshEnd + 3);
						if (argsOpenParen == std::string::npos) break;
						int depth = 1;
						size_t argsEnd = argsOpenParen + 1;
						while (depth > 0 && argsEnd < SourceCodeString.size()) {
							if (SourceCodeString[argsEnd] == '(') depth++;
							if (SourceCodeString[argsEnd] == ')') depth--;
							if (depth > 0) argsEnd++;
						}
						std::string args = SourceCodeString.substr(argsOpenParen + 1, argsEnd - argsOpenParen - 1);

                        // Build replacement: __aie_launch("funcName", meshVar, args)
                        // Note: byte sizes are injected later in EndSourceFileAction
                        // (post-AST) when parsedMeshKernels is populated.
                        std::string replacement = "__aie_launch(\"" + funcName + "\", " + meshVar + ", " + args + ")";

						// Replace from funcEnd to argsEnd+1 (include closing paren)
						SourceCodeString.replace(funcEnd, argsEnd + 1 - funcEnd, replacement);
						launchPos = funcEnd + replacement.size();

						// Track that we're in tiling mode
						isTilingLinalgMode = true;
					}
				}

                // ret = "#include <stdio.h>\n";
                // ret += "#include \"adf.h\"\n";

                // Add stub declarations for AIE-specific types that clang can't find
                // (because __AIENGINE__ isn't defined, so adf.h uses different code paths)
                // Function body is wrapped with #ifdef KERNEL_COMPILE so only type stubs needed
                ret +=
                    "// Stub type declarations for Clang parsing (function body skipped via #ifdef KERNEL_COMPILE)\n";
                ret += "#ifndef AIEHLC_STUBS_DEFINED\n";
                ret += "#define AIEHLC_STUBS_DEFINED\n";
                ret += "template<typename T> struct input_window {};\n";
                ret += "template<typename T> struct output_window {};\n";
                ret += "typedef int int32;\n";
                ret += "typedef input_window<int32> input_window_int32;\n";
                ret += "typedef output_window<int32> output_window_int32;\n";
                ret += "typedef signed char int8;\n";
                ret += "typedef input_window<int8> input_window_int8;\n";
                ret += "typedef output_window<int8> output_window_int8;\n";
                ret += "typedef short int16;\n";
                ret += "typedef input_window<int16> input_window_int16;\n";
                ret += "typedef output_window<int16> output_window_int16;\n";
                // Kernel intrinsic stubs (body is #ifdef KERNEL_COMPILE guarded)
                ret += "typedef unsigned char uint8_t;\n";
                ret += "typedef unsigned long uintptr_t;\n";
                ret += "typedef int int8_t __attribute__((mode(QI)));\n";
                ret += "typedef int int32_t __attribute__((mode(SI)));\n";
                ret += "typedef int v4int8 __attribute__((vector_size(4)));\n";
                ret += "typedef int v4int32 __attribute__((vector_size(16)));\n";
                ret += "inline unsigned get_coreid() { return 0; }\n";
                ret += "inline void klog(const char*, int) {}\n";
                ret += "template<typename T> inline void* acquire_input_window(T*) { return (void*)0; }\n";
                ret += "template<typename T> inline void* acquire_output_window(T*) { return (void*)0; }\n";
                ret += "template<typename T> inline void release_input_window(T*) {}\n";
                ret += "template<typename T> inline void release_output_window(T*) {}\n";
                ret += "#define BUF_SZ 16\n";
                ret += "#endif\n\n";

                // CUDA-style AIE API stubs for tiling mode
                if (isTilingLinalgMode) {
                    ret += "// CUDA-style AIE API stubs for Clang parsing\n";
                    ret += "#ifndef AIEHLC_TILING_STUBS_DEFINED\n";
                    ret += "#define AIEHLC_TILING_STUBS_DEFINED\n";
                    // SpatialPolicy struct + port<T, Policy> system (C++20 struct NTTP)
                    // Types and port template only — policy constants are user-defined
                    ret += "namespace aie {\n";
                    ret += "enum class Pattern  { Broadcast = 0, Scatter = 1, Multicast = 2, Gather = 3 };\n";
                    ret += "enum class Layout   { Row = 0, Col = 1, Grid = 2 };\n";
                    ret += "enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };\n";
                    ret += "struct SpatialPolicy {\n";
                    ret += "  Pattern pattern      = Pattern::Broadcast;\n";
                    ret += "  Layout  distribution = Layout::Row;\n";
                    ret += "  Flow    merge_order  = Flow::Default;\n";
                    ret += "  int     pp_depth     = 2;\n";
                    ret += "  int     max_buffer_bytes = 4096;\n";
                    ret += "  int     tile_m       = 0;\n";
                    ret += "  int     tile_n       = 0;\n";
                    ret += "  int     tile_k       = 0;\n";
                    ret += "};\n";
                    ret += "template<typename T, SpatialPolicy P> struct port { using type = T; };\n";
                    // Built-in query function stubs — Clang parses these but the compiler
                    // replaces the calls with computed integer literals during kernel body rewriting.
                    ret += "template<typename T> constexpr int get_num_rounds(T) { return 0; }\n";
                    ret += "template<typename T> constexpr int get_buffer_size(T) { return 0; }\n";
                    ret += "constexpr int get_tile_rows() { return 0; }\n";
                    ret += "constexpr int get_tile_cols() { return 0; }\n";
                    ret += "constexpr int get_data_row() { return 0; }\n";
                    ret += "constexpr int get_data_col() { return 0; }\n";
                    ret += "constexpr int get_k_dim() { return 0; }\n";
                    ret += "constexpr int get_tile_m() { return 0; }\n";
                    ret += "constexpr int get_tile_n() { return 0; }\n";
                    ret += "constexpr int get_effective_k() { return 0; }\n";
                    ret += "constexpr int get_k_rounds() { return 0; }\n";
                    ret += "constexpr int get_spatial_m_rounds() { return 0; }\n";
                    ret += "constexpr int get_spatial_n_rounds() { return 0; }\n";
                    ret += "}\n";
                    ret += "struct aiePartition {\n";
                    ret += "    int startCol, endCol, startRow, endRow;\n";
                    ret += "};\n";
                    // Forward declarations needed by aieArray::partition()/alloc()/free()/synchronizecpu()
                    ret += "struct XAie_DevInst;\n";
                    ret += "extern \"C\" XAie_DevInst *__Runtime_init_mesh_partition(int meshId, int startCol, int "
                           "numCols);\n";
                    ret += "extern \"C\" XAie_DevInst *__Runtime_get_partition_dev(int meshId);\n";
                    ret += "extern \"C\" void *__Runtime_alloc_buffer(XAie_DevInst *dev, __SIZE_TYPE__ size_bytes);\n";
                    ret += "extern \"C\" void __Runtime_free_buffer(XAie_DevInst *dev, void *ptr);\n";
                    ret +=
                        "extern \"C\" void __Runtime_sync_for_cpu(XAie_DevInst *dev, void *ptr, __SIZE_TYPE__ size);\n";
                    ret += "extern \"C\" void __Runtime_teardown_all();\n";
                    // New programming model types: aieMesh + aieArray
                    ret += "struct aieMesh {\n";
                    ret += "    int rows, cols;\n";
                    ret += "    aiePartition partition;\n";
                    ret += "    int meshId;\n";
                    ret += "};\n";
                    ret += "struct aieArray {\n";
                    ret += "    int nextMeshId = 0;\n";
                    ret += "    XAie_DevInst* _dev = nullptr;\n";
                    ret += "    aieMesh partition(aiePartition p, int rows, int cols) {\n";
                    ret += "        int meshId = nextMeshId++;\n";
                    ret += "        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol + "
                           "1);\n";
                    ret += "        return aieMesh{rows, cols, p, meshId};\n";
                    ret += "    }\n";
                    ret += "    void* alloc(__SIZE_TYPE__ size) { return __Runtime_alloc_buffer(_dev, size); }\n";
                    ret += "    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }\n";
                    ret += "    void synchronizecpu(void* ptr, __SIZE_TYPE__ size) { __Runtime_sync_for_cpu(_dev, ptr, "
                           "size); }\n";
                    ret += "};\n";
                    // Backward-compatible aieDim
                    ret += "struct aieDim {\n";
                    ret += "    int rows, cols;\n";
                    ret += "    aiePartition partition;\n";
                    ret += "    bool hasPartition;\n";
                    ret +=
                        "    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, hasPartition(false) {}\n";
                    ret += "    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), "
                           "hasPartition(true) {}\n";
                    ret += "};\n";
                    ret += "inline void aieSetDevice(int) {}\n";
                    ret += "inline void aieDeviceSynchronize() {}\n";
                    // Stub extern + __aie_launch for Clang AST parsing only.
                    // parsedTensors is empty at this point (pre-AST-visit), so we
                    // emit a no-op template.  The correct forwarding overload is
                    // generated later by TilingLinalgPipeline::runPipeline when
                    // appending user source to host.cc.
                    ret += "extern void host_canonicalized(...);\n";
                    ret += "template<typename... Args>\n";
                    ret += "inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {\n";
                    ret += "    (void)kernel; (void)mesh; (void)sizeof...(args);\n";
                    ret += "}\n";
                    ret += "template<typename... Args>\n";
                    ret += "inline void __aie_launch(const char* kernel, aieDim mesh, Args... args) {\n";
                    ret += "    (void)kernel; (void)mesh; (void)sizeof...(args);\n";
                    ret += "}\n";
                    ret += "#endif\n\n";
                }

                if (!kernel_name_list.empty()) {
                    ret += "#include \"xaiengine.h\"\n";
                    ret += "#ifndef AIEHLC_KERNEL_LAUNCH_HELPERS_DEFINED\n";
                    ret += "#define AIEHLC_KERNEL_LAUNCH_HELPERS_DEFINED\n";
                    ret += "static XAie_DevInst *__aiehlc_global_dev_inst = nullptr;\n";
                    ret += "#define AIEHLC_MAX_LAUNCHED_TILES 1024\n";
                    ret += "static XAie_LocType __aiehlc_launched_tiles[AIEHLC_MAX_LAUNCHED_TILES];\n";
                    ret += "static int __aiehlc_launched_tile_count = 0;\n";
                    ret += "inline void __use_device_instance(XAie_DevInst *DevInst) {\n";
                    ret += "    __aiehlc_global_dev_inst = DevInst;\n";
                    ret += "}\n";
                    ret += "inline void __sync_kernels() {\n";
                    ret += "    if (__aiehlc_global_dev_inst == nullptr) {\n";
                    ret += "        return;\n";
                    ret += "    }\n";
                    ret += "    for (int i = 0; i < __aiehlc_launched_tile_count; ++i) {\n";
                    ret += "        while (XAie_CoreWaitForDone(__aiehlc_global_dev_inst, __aiehlc_launched_tiles[i], "
                           "0) != XAIE_OK) {}\n";
                    ret += "    }\n";
                    ret += "    __aiehlc_launched_tile_count = 0;\n";
                    ret += "}\n";
                    ret += "#endif\n";
                }

                for (auto x:kernel_name_list) {
                    ret += "extern unsigned char _binary_kernel_" + x + "_start[];\n" +
                           "template <int Col, int Row, typename... Args>\n"
                           "inline void " +
                           x +
                           "(Args&&... args) {\n"
                           "    (void)sizeof...(args); \n"
                           "    XAie_DevInst *DevInst = __aiehlc_global_dev_inst;\n"
                           "    if (DevInst == nullptr) {\n"
                           "        return;\n"
                           "    }\n"
                           "    XAie_CoreDisable(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_CoreReset(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_LoadElfMem(DevInst, XAie_TileLoc(Col,Row), (unsigned char *)_binary_kernel_" +
                           x +
                           "_start);\n"
                           "    XAie_CoreUnreset(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    XAie_CoreEnable(DevInst, XAie_TileLoc(Col,Row));\n"
                           "    if (__aiehlc_launched_tile_count < AIEHLC_MAX_LAUNCHED_TILES) {\n"
                           "        __aiehlc_launched_tiles[__aiehlc_launched_tile_count++] = XAie_TileLoc(Col,Row);\n"
                           "    } else {\n"
                           "        while (XAie_CoreWaitForDone(DevInst, XAie_TileLoc(Col,Row), 0) != XAIE_OK) {}\n"
                           "    }\n"
                           "    return;\n"
                           "};\n";
                }
                ret += SourceCodeString;
				// std::cout << ret << std::endl;
				/*
					 llvm::StringRef ModifiedCode = SourceCodeString;
					 std::unique_ptr<llvm::MemoryBuffer> ModifiedBuffer =
					 llvm::MemoryBuffer::getMemBufferCopy(ModifiedCode," Filename");

					 SourceMgr.overrideFileContents(FileEntry, std::move(ModifiedBuffer));

				// Get the modified file name
				llvm::StringRef ModifiedFileName = SourceMgr.getFileEntryForID(SourceMgr.getMainFileID())->getName();

				// Print modified file name
				llvm::outs() << "Modified file name: " << ModifiedFileName  << ModifiedCode<< "\n";
				*/
			}
			return ret;
		}

		void EndSourceFileAction() override {
			//  std::cout << "---------EndSourceFileAction---------" << std::endl;

            // Stop immediately if Clang reported any compilation errors.
            // Running the MLIR pipeline on an erroneous AST produces misleading output.
            if (getCompilerInstance().getDiagnostics().hasErrorOccurred()) {
                llvm::errs() << "\n=========================================================\n"
                             << "Error: source file has compilation errors. Stopping.\n"
                             << "=========================================================\n";
                return;
            }

             const RewriteBuffer *RewriteBuf = TheRewriter.getRewriteBufferFor(TheRewriter.getSourceMgr().getMainFileID());
			 if (RewriteBuf) {
				 //RewriteBuf->write(llvm::outs());
        //llvm::outs() << "Rewritten Source Code:\n" << RewriteBuf << "\n";
	  		} else {
					llvm::errs() << "RewriteBuf is null\n";
				}

			if (isTilingLinalgMode) {
				// ---- NEW PATH: tilinglinalg pipeline ----
				llvm::outs() << "[TilingLinalg] Running multi-tile pipeline (mesh="
							 << tilingMeshRows << "x" << tilingMeshCols << ")\n";

                // Capture user's rewritten source for merging with pipeline output.
                // The RewriteBuf contains the entire user source with __global__ body
                // removed and <<<mesh>>> syntax replaced by __aie_launch() calls.
                std::string userRewrittenSource;
                if (RewriteBuf) {
                    llvm::raw_string_ostream rso(userRewrittenSource);
                    RewriteBuf->write(rso);
                    rso.flush();
                }

                // Post-process userRewrittenSource: inject byte sizes into __aie_launch() calls.
                // The <<<>>> rewriter ran in BeginSourceFileAction (before AST visit), so
                // parsedMeshKernels was empty at that time and no sizes were injected.
                // Now parsedMeshKernels is populated, so we can do it.
                if (!userRewrittenSource.empty() && !parsedMeshKernels.empty()) {
                    size_t searchPos = 0;
                    while ((searchPos = userRewrittenSource.find("__aie_launch(", searchPos)) != std::string::npos) {
                        // Find the opening paren of __aie_launch(
                        size_t openParen = searchPos + strlen("__aie_launch");
                        // Find matching close paren
                        int depth = 1;
                        size_t pos = openParen + 1;
                        while (depth > 0 && pos < userRewrittenSource.size()) {
                            if (userRewrittenSource[pos] == '(')
                                depth++;
                            if (userRewrittenSource[pos] == ')')
                                depth--;
                            if (depth > 0)
                                pos++;
                        }
                        // Extract full arg string between parens
                        std::string allArgs = userRewrittenSource.substr(openParen + 1, pos - openParen - 1);

                        // Split by commas respecting nested parens
                        auto splitComma = [](const std::string &s) -> std::vector<std::string> {
                            std::vector<std::string> result;
                            int d = 0;
                            size_t start = 0;
                            for (size_t j = 0; j < s.size(); ++j) {
                                if (s[j] == '(')
                                    d++;
                                else if (s[j] == ')')
                                    d--;
                                else if (s[j] == ',' && d == 0) {
                                    std::string a = s.substr(start, j - start);
                                    size_t b = a.find_first_not_of(" \t\n\r");
                                    size_t e = a.find_last_not_of(" \t\n\r");
                                    if (b != std::string::npos)
                                        result.push_back(a.substr(b, e - b + 1));
                                    start = j + 1;
                                }
                            }
                            std::string a = s.substr(start);
                            size_t b = a.find_first_not_of(" \t\n\r");
                            size_t e = a.find_last_not_of(" \t\n\r");
                            if (b != std::string::npos)
                                result.push_back(a.substr(b, e - b + 1));
                            return result;
                        };
                        std::vector<std::string> args = splitComma(allArgs);

                        // args[0] = kernel name string, args[1] = mesh variable, args[2..] = user args
                        // Extract kernel name from the quoted string literal (strip quotes)
                        if (args.size() >= 3) {
                            std::string kernelLit = args[0];
                            // Remove surrounding quotes: "matmul" -> matmul
                            size_t q1 = kernelLit.find('"');
                            size_t q2 = kernelLit.rfind('"');
                            std::string kernelName;
                            if (q1 != std::string::npos && q2 != std::string::npos && q2 > q1)
                                kernelName = kernelLit.substr(q1 + 1, q2 - q1 - 1);

                            const MeshKernelDesc *mkdPtr = nullptr;
                            for (auto &mk : parsedMeshKernels) {
                                if (mk.kernelName == kernelName) {
                                    mkdPtr = &mk;
                                    break;
                                }
                            }

                            if (mkdPtr) {
                                unsigned numPtrArgs = (unsigned)mkdPtr->tensors.size();
                                // Rebuild the call with interleaved sizes
                                std::string newCall = "__aie_launch(" + args[0] + ", " + args[1];
                                for (unsigned ai = 2; ai < args.size(); ++ai) {
                                    newCall += ", " + args[ai];
                                    unsigned tensorIdx = ai - 2;
                                    if (tensorIdx < numPtrArgs) {
                                        int64_t bytes = 1;
                                        for (auto dim : mkdPtr->tensors[tensorIdx].shape)
                                            bytes *= dim;
                                        bytes *= mkdPtr->tensors[tensorIdx].elementBitWidth / 8;
                                        newCall += ", (size_t)" + std::to_string(bytes);
                                    }
                                }
                                newCall += ")";
                                userRewrittenSource.replace(searchPos, pos + 1 - searchPos, newCall);
                                searchPos += newCall.size();
                                continue;
                            }
                        }
                        searchPos = pos + 1;
                    }
                }

                // Derive AIE generation string from the Clang preprocessor's AIE_GEN macro
                // (set via --extra-arg=-DAIE_GEN=<version> at runtime, not compile-time)
                std::string aieGenStr = "Gen2"; // default
                {
                    auto &PP = getCompilerInstance().getPreprocessor();
                    auto *II = PP.getIdentifierInfo("AIE_GEN");
                    if (II && II->hasMacroDefinition()) {
                        auto *MI = PP.getMacroInfo(II);
                        if (MI && MI->getNumTokens() == 1 &&
                            MI->getReplacementToken(0).is(clang::tok::numeric_constant)) {
                            llvm::StringRef valStr = MI->getReplacementToken(0).getLiteralData()
                                                         ? llvm::StringRef(MI->getReplacementToken(0).getLiteralData(),
                                                                           MI->getReplacementToken(0).getLength())
                                                         : "";
                            int aieGen = 2;
                            valStr.getAsInteger(10, aieGen);
                            if (aieGen == 1)
                                aieGenStr = "Gen1";
                            else if (aieGen >= 5)
                                aieGenStr = "Gen5";
                        }
                    }
                }
            llvm::outs() << "[TilingLinalg] AIE generation: " << aieGenStr << "\n";

            mlir::MLIRContext ctx;
            TilingLinalgPipeline::registerDialects(ctx);

            std::string outputDir = std::string(AOUT) + "worklocal/";
            bool isMultiKernel = parsedMeshKernels.size() > 1;

            // ---- Multi-kernel loop: run pipeline per kernel ----
            if (isMultiKernel) {
                llvm::outs() << "[TilingLinalg] Multi-kernel mode: " << parsedMeshKernels.size() << " kernels\n";

                for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                    auto &mkd = parsedMeshKernels[ki];
                    llvm::outs() << "[TilingLinalg] === Pipeline run " << ki << ": kernel=" << mkd.kernelName
                                 << " mesh=" << mkd.meshRows << "x" << mkd.meshCols << " ===\n";

                    int rows = mkd.meshRows > 0 ? mkd.meshRows : 2;
                    int cols = mkd.meshCols > 0 ? mkd.meshCols : 2;

                    // Use per-kernel tensors and splitModel
                    SplitModel &splitModel = mkd.splitModel;
                    if (splitModel.tensorSplits.empty())
                        splitModel = SplitModel::gemm();

                    // Build routing IR for this kernel
                    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, rows, cols, mkd.tensors, splitModel,
                                                                       mkd.partition, aieGenStr);

                    // Set K-round module attributes for downstream DMA passes
                    if (mkd.derivedParams.valid && mkd.derivedParams.kRounds > 1) {
                        mlir::OpBuilder attrBuilder(&ctx);
                        module->setAttr("routing.effective_k",
                                        attrBuilder.getI64IntegerAttr(mkd.derivedParams.effectiveK));
                        module->setAttr("routing.k_rounds", attrBuilder.getI64IntegerAttr(mkd.derivedParams.kRounds));
                        module->setAttr("routing.full_k", attrBuilder.getI64IntegerAttr(mkd.derivedParams.kDim));
                        module->setAttr("routing.tile_m", attrBuilder.getI64IntegerAttr(mkd.derivedParams.tileM));
                        module->setAttr("routing.tile_rows", attrBuilder.getI64IntegerAttr(mkd.derivedParams.tileRows));
                        module->setAttr("routing.m_rounds",
                                        attrBuilder.getI64IntegerAttr(mkd.derivedParams.spatialMRounds));
                        module->setAttr("routing.tile_n", attrBuilder.getI64IntegerAttr(mkd.derivedParams.tileN));
                        module->setAttr("routing.tile_cols", attrBuilder.getI64IntegerAttr(mkd.derivedParams.tileCols));
                        module->setAttr("routing.n_rounds",
                                        attrBuilder.getI64IntegerAttr(mkd.derivedParams.spatialNRounds));
                        llvm::outs() << "[TilingLinalg] Set K-round module attrs: effective_k="
                                     << mkd.derivedParams.effectiveK << " k_rounds=" << mkd.derivedParams.kRounds
                                     << " full_k=" << mkd.derivedParams.kDim << " tile_m=" << mkd.derivedParams.tileM
                                     << " tile_rows=" << mkd.derivedParams.tileRows
                                     << " m_rounds=" << mkd.derivedParams.spatialMRounds
                                     << " tile_n=" << mkd.derivedParams.tileN
                                     << " tile_cols=" << mkd.derivedParams.tileCols
                                     << " n_rounds=" << mkd.derivedParams.spatialNRounds << "\n";
                    }

                    // Replace aie::get_*() calls in kernel body with computed integer literals
                    std::string resolvedBody = mkd.kernelBody;
                    if (mkd.derivedParams.valid && !resolvedBody.empty()) {
                        llvm::outs() << "[TilingLinalg] Replacing aie::get_*() calls in kernel body for "
                                     << mkd.kernelName << "\n";

                        // Build port name -> index map from stored varNames
                        std::unordered_map<std::string, size_t> paramToPort;
                        for (size_t pi = 0; pi < mkd.portVarNames.size(); ++pi) {
                            paramToPort[mkd.portVarNames[pi]] = pi;
                        }

                        // Replace aie::get_num_rounds(win_x) and aie::get_buffer_size(win_x)
                        for (const auto &funcName : {"aie::get_num_rounds", "aie::get_buffer_size"}) {
                            std::string prefix = std::string(funcName) + "(";
                            size_t pos = 0;
                            while ((pos = resolvedBody.find(prefix, pos)) != std::string::npos) {
                                size_t argStart = pos + prefix.size();
                                size_t argEnd = resolvedBody.find(")", argStart);
                                if (argEnd == std::string::npos)
                                    break;
                                std::string argName = resolvedBody.substr(argStart, argEnd - argStart);
                                while (!argName.empty() && argName.front() == ' ')
                                    argName.erase(0, 1);
                                while (!argName.empty() && argName.back() == ' ')
                                    argName.pop_back();
                                auto it = paramToPort.find(argName);
                                if (it != paramToPort.end() && it->second < mkd.derivedParams.portParams.size()) {
                                    int64_t val;
                                    if (std::string(funcName) == "aie::get_num_rounds")
                                        val = mkd.derivedParams.portParams[it->second].numRounds;
                                    else
                                        val = mkd.derivedParams.portParams[it->second].bufferSize;
                                    std::string replacement = std::to_string(val);
                                    resolvedBody.replace(pos, argEnd + 1 - pos, replacement);
                                    pos += replacement.size();
                                } else {
                                    pos = argEnd + 1;
                                }
                            }
                        }

                        // Replace simple calls: get_tile_rows(), get_tile_cols(), get_k_dim()
                        auto replaceSimpleCall = [&](const std::string &call, int64_t val) {
                            std::string replacement = std::to_string(val);
                            size_t pos = 0;
                            while ((pos = resolvedBody.find(call, pos)) != std::string::npos) {
                                resolvedBody.replace(pos, call.size(), replacement);
                                pos += replacement.size();
                            }
                        };
                        replaceSimpleCall("aie::get_tile_rows()", mkd.derivedParams.tileM > 0
                                                                      ? mkd.derivedParams.tileM
                                                                      : mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_tile_cols()", mkd.derivedParams.tileN > 0
                                                                      ? mkd.derivedParams.tileN
                                                                      : mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_data_row()", mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_data_col()", mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_k_dim()", mkd.derivedParams.kDim);
                        // Two-level tiling query functions
                        replaceSimpleCall("aie::get_tile_m()", mkd.derivedParams.tileM > 0
                                                                   ? mkd.derivedParams.tileM
                                                                   : mkd.derivedParams.tileRows);
                        replaceSimpleCall("aie::get_tile_n()", mkd.derivedParams.tileN > 0
                                                                   ? mkd.derivedParams.tileN
                                                                   : mkd.derivedParams.tileCols);
                        replaceSimpleCall("aie::get_effective_k()", mkd.derivedParams.effectiveK > 0
                                                                        ? mkd.derivedParams.effectiveK
                                                                        : mkd.derivedParams.kDim);
                        replaceSimpleCall("aie::get_k_rounds()", mkd.derivedParams.kRounds);
                        replaceSimpleCall("aie::get_spatial_m_rounds()", mkd.derivedParams.spatialMRounds);
                        replaceSimpleCall("aie::get_spatial_n_rounds()", mkd.derivedParams.spatialNRounds);
                    }

                    // Prepend user macros to kernel body
                    std::string kernelBodyWithMacros = resolvedBody;
                    if (!userMacroDefines.empty() && !resolvedBody.empty()) {
                        std::string macroBlock = "// User macro definitions from source file\n";
                        for (const auto &macro : userMacroDefines) {
                            macroBlock += macro + "\n";
                        }
                        macroBlock += "\n";
                        kernelBodyWithMacros = macroBlock + resolvedBody;
                    }

                    // First kernel writes host.cc fresh; subsequent kernels append
                    bool appendMode = (ki > 0);
                    // In multi-kernel mode, always suffix host function with kernel name
                    std::string hostFuncSuffix = mkd.kernelName;
                    // Don't pass userRewrittenSource in multi-kernel mode — we emit it ourselves after all runs
                    unsigned hostDdrArgs = 0;
                    if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir, kernelBodyWithMacros,
                                                           mkd.kernelFuncName, parsedDebugLevel,
                                                           /*userRewrittenSource=*/"", {}, mkd.maxPPBytes, aieGenStr,
                                                           hostFuncSuffix, appendMode, &hostDdrArgs)) {
                        llvm::errs() << "[TilingLinalg] Pipeline FAILED for kernel: " << mkd.kernelName << "\n";
                        std::exit(1);
                    }
                    mkd.numHostDdrArgs = hostDdrArgs;
                    llvm::outs() << "[TilingLinalg] Pipeline completed for kernel: " << mkd.kernelName
                                 << " (hostDdrArgs=" << hostDdrArgs << ")\n";
                }

                // ---- Post-pipeline: append user source + unified __aie_launch to host.cc ----
                {
                    std::string hostPath = outputDir + "/host.cc";
                    std::error_code ec;
                    llvm::raw_fd_ostream stream(hostPath, ec, llvm::sys::fs::OF_Append);
                    if (ec) {
                        llvm::errs() << "Failed to open " << hostPath << " for appending: " << ec.message() << "\n";
                        std::exit(1);
                    }

                    stream << "\n// ===== User source + multi-kernel __aie_launch dispatch =====\n";
                    stream << "#define AIEHLC_TILING_STUBS_DEFINED\n";
                    stream << "#include <cstring>\n";
                    // Re-emit SpatialPolicy types so user source constexpr definitions compile
                    stream << "namespace aie {\n";
                    stream << "enum class Pattern  { Broadcast = 0, Scatter = 1, Multicast = 2, Gather = 3 };\n";
                    stream << "enum class Layout   { Row = 0, Col = 1, Grid = 2 };\n";
                    stream << "enum class Flow     { Default = 0, LeftToRight = 1, RightToLeft = 2 };\n";
                    stream << "struct SpatialPolicy {\n";
                    stream << "  Pattern pattern      = Pattern::Broadcast;\n";
                    stream << "  Layout  distribution = Layout::Row;\n";
                    stream << "  Flow    merge_order  = Flow::Default;\n";
                    stream << "  int     pp_depth     = 2;\n";
                    stream << "  int     max_buffer_bytes = 4096;\n";
                    stream << "  int     tile_m       = 0;\n";
                    stream << "  int     tile_n       = 0;\n";
                    stream << "  int     tile_k       = 0;\n";
                    stream << "};\n";
                    stream << "template<typename T, SpatialPolicy P> struct port { using type = T; };\n";
                    stream << "template<typename T> constexpr int get_num_rounds(T) { return 0; }\n";
                    stream << "template<typename T> constexpr int get_buffer_size(T) { return 0; }\n";
                    stream << "constexpr int get_tile_rows() { return 0; }\n";
                    stream << "constexpr int get_tile_cols() { return 0; }\n";
                    stream << "constexpr int get_data_row() { return 0; }\n";
                    stream << "constexpr int get_data_col() { return 0; }\n";
                    stream << "constexpr int get_k_dim() { return 0; }\n";
                    stream << "constexpr int get_tile_m() { return 0; }\n";
                    stream << "constexpr int get_tile_n() { return 0; }\n";
                    stream << "constexpr int get_effective_k() { return 0; }\n";
                    stream << "constexpr int get_k_rounds() { return 0; }\n";
                    stream << "constexpr int get_spatial_m_rounds() { return 0; }\n";
                    stream << "constexpr int get_spatial_n_rounds() { return 0; }\n";
                    stream << "}\n";
                    // aiePartition struct
                    stream << "struct aiePartition {\n";
                    stream << "    int startCol, endCol, startRow, endRow;\n";
                    stream << "};\n";
                    // aieMesh + aieArray
                    // Note: XAie_DevInst, __Runtime_explicit_init, __Runtime_alloc_buffer,
                    // __Runtime_free_buffer are already declared via #include "aie_runtime.h"
                    stream << "struct aieMesh {\n";
                    stream << "    int rows, cols;\n";
                    stream << "    aiePartition partition;\n";
                    stream << "    int meshId;\n";
                    stream << "};\n";
                    stream << "struct aieArray {\n";
                    stream << "    int nextMeshId = 0;\n";
                    stream << "    XAie_DevInst* _dev = nullptr;\n";
                    stream << "    aieMesh partition(aiePartition p, int rows, int cols) {\n";
                    stream << "        int meshId = nextMeshId++;\n";
                    stream << "        _dev = __Runtime_init_mesh_partition(meshId, p.startCol, p.endCol - p.startCol "
                              "+ 1);\n";
                    stream << "        return aieMesh{rows, cols, p, meshId};\n";
                    stream << "    }\n";
                    stream << "    void* alloc(size_t size) { return __Runtime_alloc_buffer(_dev, size); }\n";
                    stream << "    void free(void* ptr) { __Runtime_free_buffer(_dev, ptr); }\n";
                    stream << "    void synchronizecpu(void* ptr, size_t size) { __Runtime_sync_for_cpu(_dev, ptr, "
                              "size); }\n";
                    stream << "};\n";
                    // aieDim (backward compat)
                    stream << "struct aieDim {\n";
                    stream << "    int rows, cols;\n";
                    stream << "    aiePartition partition;\n";
                    stream << "    bool hasPartition;\n";
                    stream << "    aieDim(int r, int c) : rows(r), cols(c), partition{-1,-1,-1,-1}, "
                              "hasPartition(false) {}\n";
                    stream << "    aieDim(int r, int c, aiePartition p) : rows(r), cols(c), partition(p), "
                              "hasPartition(true) {}\n";
                    stream << "};\n";
                    stream << "inline void aieSetDevice(int) {}\n";
                    stream << "inline void aieDeviceSynchronize() {}\n";

                    // Emit extern declarations for per-kernel binary symbols
                    // (must appear before __aie_launch which references them)
                    for (auto &mkd : parsedMeshKernels) {
                        stream << "extern unsigned char _binary_kernel_" << mkd.kernelName << "_start[];\n";
                        stream << "extern unsigned char _binary_kernel_" << mkd.kernelName << "_end[];\n";
                        stream << "extern unsigned int _binary_kernel_" << mkd.kernelName << "_size;\n";
                    }

                    // Find max DDR args across all kernels for __aie_launch signature
                    unsigned maxDdrArgs = 0;
                    for (auto &mkd : parsedMeshKernels) {
                        if (mkd.numHostDdrArgs > maxDdrArgs)
                            maxDdrArgs = mkd.numHostDdrArgs;
                    }

                    // Helper: emit XAie_MemSyncForDevVAddr calls for ALL buffers before DMA.
                    // On ARM (baremetal), SyncForDev flushes+invalidates cache lines, which is
                    // needed for outputs too: dirty cache lines (e.g. zeroed output buffer) must
                    // be flushed and invalidated BEFORE DMA writes results to DDR, otherwise a
                    // post-DMA invalidate (clean+invalidate) would flush stale zeros over the
                    // DMA results. No post-launch sync is needed.
                    auto emitSyncCalls = [](llvm::raw_fd_ostream &os, const MeshKernelDesc &mkd,
                                            const std::string &indent, bool beforeLaunch) {
                        if (!beforeLaunch)
                            return; // no post-launch sync needed on ARM
                        unsigned limit = mkd.numHostDdrArgs;
                        if (limit > mkd.tensors.size())
                            limit = mkd.tensors.size();
                        for (unsigned i = 0; i < limit; ++i) {
                            os << indent << "XAie_MemSyncForDevVAddr(dev, _t" << i << ", (uint64_t)_s" << i << ");\n";
                        }
                    };

                    // Emit unified __aie_launch with strcmp dispatch (aieMesh overload)
                    if (maxDdrArgs > 0) {
                        stream << "inline void __aie_launch(const char* kernel, aieMesh mesh";
                        for (unsigned i = 0; i < maxDdrArgs; ++i)
                            stream << ", void* _t" << i << ", size_t _s" << i;
                        stream << ", ...) {\n";
                        stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                        for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                            auto &mkd = parsedMeshKernels[ki];
                            std::string hostFunc = "host_canonicalized_" + mkd.kernelName;
                            stream << "    " << (ki == 0 ? "if" : "} else if") << " (strcmp(kernel, \""
                                   << mkd.kernelName << "\") == 0) {\n";
                            stream << "        __Runtime_set_kernel_elf(_binary_kernel_" << mkd.kernelName
                                   << "_start);\n";
                            // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                            emitSyncCalls(stream, mkd, "        ", /*beforeLaunch=*/true);
                            stream << "        " << hostFunc << "(dev";
                            for (unsigned i = 0; i < mkd.numHostDdrArgs; ++i)
                                stream << ", _t" << i;
                            stream << ");\n";
                            // No post-launch sync needed (cache lines already invalidated)
                            emitSyncCalls(stream, mkd, "        ", /*beforeLaunch=*/false);
                        }
                        stream << "    }\n";
                        stream << "}\n";

                        // aieDim overload (backward compat)
                        stream << "inline void __aie_launch(const char* kernel, aieDim mesh";
                        for (unsigned i = 0; i < maxDdrArgs; ++i)
                            stream << ", void* _t" << i << ", size_t _s" << i;
                        stream << ", ...) {\n";
                        stream << "    (void)kernel;\n";
                        stream << "    XAie_DevInst* dev;\n";
                        stream << "    if (mesh.hasPartition) {\n";
                        stream << "        dev = __Runtime_explicit_init_partition(mesh.partition.startCol, "
                                  "mesh.partition.endCol - mesh.partition.startCol + 1);\n";
                        stream << "    } else {\n";
                        stream << "        dev = __Runtime_explicit_init();\n";
                        stream << "    }\n";
                        // aieDim only supports single kernel (backward compat) — dispatch first kernel
                        if (!parsedMeshKernels.empty()) {
                            auto &mkd0 = parsedMeshKernels[0];
                            std::string hostFunc = "host_canonicalized_" + mkd0.kernelName;
                            stream << "    __Runtime_set_kernel_elf(_binary_kernel_" << mkd0.kernelName << "_start);\n";
                            // Flush+invalidate ALL buffers (inputs AND outputs) before DMA
                            emitSyncCalls(stream, mkd0, "    ", /*beforeLaunch=*/true);
                            stream << "    " << hostFunc << "(dev";
                            for (unsigned i = 0; i < mkd0.numHostDdrArgs; ++i)
                                stream << ", _t" << i;
                            stream << ");\n";
                            // No post-launch sync needed (cache lines already invalidated)
                            emitSyncCalls(stream, mkd0, "    ", /*beforeLaunch=*/false);
                        }
                        stream << "    __Runtime_explicit_teardown(dev);\n";
                        stream << "}\n";
                    } else {
                        // No DDR args variant (template)
                        stream << "template<typename... Args>\n";
                        stream << "inline void __aie_launch(const char* kernel, aieMesh mesh, Args... args) {\n";
                        stream << "    XAie_DevInst* dev = __Runtime_get_partition_dev(mesh.meshId);\n";
                        for (size_t ki = 0; ki < parsedMeshKernels.size(); ++ki) {
                            auto &mkd = parsedMeshKernels[ki];
                            std::string hostFunc = "host_canonicalized_" + mkd.kernelName;
                            stream << "    " << (ki == 0 ? "if" : "} else if") << " (strcmp(kernel, \""
                                   << mkd.kernelName << "\") == 0) {\n";
                            stream << "        __Runtime_set_kernel_elf(_binary_kernel_" << mkd.kernelName
                                   << "_start);\n";
                            stream << "        " << hostFunc << "(dev);\n";
                        }
                        stream << "    }\n";
                        stream << "}\n";
                    }

                    // Append user's rewritten source
                    stream << userRewrittenSource << "\n";
                    stream.close();
                    llvm::outs() << "[TilingLinalg] Multi-kernel __aie_launch + user source appended to host.cc\n";
                }
            } else {
                // ---- Single-kernel path (backward compatible) ----
                // Build tensor params — use defaults matching unitest if not parsed
                std::vector<TensorParam> tensors;
                SplitModel splitModel;
                std::string singleKernelBody = userKernelBody;
                std::string singleKernelFuncName = userKernelFuncName;
                int64_t effectiveMaxPPBytes = 4096;

                if (!parsedMeshKernels.empty()) {
                    // Use data from the single MeshKernelDesc
                    auto &mkd = parsedMeshKernels[0];
                    tensors = mkd.tensors;
                    splitModel = mkd.splitModel;
                    singleKernelBody = mkd.kernelBody;
                    singleKernelFuncName = mkd.kernelFuncName;
                    effectiveMaxPPBytes = mkd.maxPPBytes;
                } else if (parsedTensors.empty()) {
                    tensors.push_back({{16, 16}, 8, true});
                    splitModel = SplitModel::gemm();
                } else {
                    for (auto &pt : parsedTensors) {
                        tensors.push_back({pt.shape, pt.elementBitWidth, pt.isInput});
                    }
                    splitModel = SplitModel::gemm();
                }
                if (splitModel.tensorSplits.empty())
                    splitModel = SplitModel::gemm();

                // Use default mesh if not parsed
                int rows = tilingMeshRows > 0 ? tilingMeshRows : 2;
                int cols = tilingMeshCols > 0 ? tilingMeshCols : 2;

                // Build routing IR
                auto module = TilingLinalgPipeline::buildRoutingIR(ctx, rows, cols, tensors, splitModel,
                                                                   parsedPartition, aieGenStr);

                // Set K-round module attributes for downstream DMA passes
                if (derivedTilingParams.valid && derivedTilingParams.kRounds > 1) {
                    mlir::OpBuilder attrBuilder(&ctx);
                    module->setAttr("routing.effective_k",
                                    attrBuilder.getI64IntegerAttr(derivedTilingParams.effectiveK));
                    module->setAttr("routing.k_rounds", attrBuilder.getI64IntegerAttr(derivedTilingParams.kRounds));
                    module->setAttr("routing.full_k", attrBuilder.getI64IntegerAttr(derivedTilingParams.kDim));
                    module->setAttr("routing.tile_m", attrBuilder.getI64IntegerAttr(derivedTilingParams.tileM));
                    module->setAttr("routing.tile_rows", attrBuilder.getI64IntegerAttr(derivedTilingParams.tileRows));
                    module->setAttr("routing.m_rounds",
                                    attrBuilder.getI64IntegerAttr(derivedTilingParams.spatialMRounds));
                    module->setAttr("routing.tile_n", attrBuilder.getI64IntegerAttr(derivedTilingParams.tileN));
                    module->setAttr("routing.tile_cols", attrBuilder.getI64IntegerAttr(derivedTilingParams.tileCols));
                    module->setAttr("routing.n_rounds",
                                    attrBuilder.getI64IntegerAttr(derivedTilingParams.spatialNRounds));
                    llvm::outs() << "[TilingLinalg] Set K-round module attrs: effective_k="
                                 << derivedTilingParams.effectiveK << " k_rounds=" << derivedTilingParams.kRounds
                                 << " full_k=" << derivedTilingParams.kDim << " tile_m=" << derivedTilingParams.tileM
                                 << " tile_rows=" << derivedTilingParams.tileRows
                                 << " m_rounds=" << derivedTilingParams.spatialMRounds
                                 << " tile_n=" << derivedTilingParams.tileN
                                 << " tile_cols=" << derivedTilingParams.tileCols
                                 << " n_rounds=" << derivedTilingParams.spatialNRounds << "\n";
                }

                // Replace aie::get_*() calls in kernel body with computed integer literals
                if (derivedTilingParams.valid && !singleKernelBody.empty()) {
                    llvm::outs() << "[TilingLinalg] Replacing aie::get_*() calls in kernel body\n";

                    std::unordered_map<std::string, size_t> paramToPort;
                    for (size_t i = 0; i < parsedTensors.size(); ++i) {
                        paramToPort[parsedTensors[i].varName] = i;
                    }

                    for (const auto &funcName : {"aie::get_num_rounds", "aie::get_buffer_size"}) {
                        std::string prefix = std::string(funcName) + "(";
                        size_t pos = 0;
                        while ((pos = singleKernelBody.find(prefix, pos)) != std::string::npos) {
                            size_t argStart = pos + prefix.size();
                            size_t argEnd = singleKernelBody.find(")", argStart);
                            if (argEnd == std::string::npos)
                                break;
                            std::string argName = singleKernelBody.substr(argStart, argEnd - argStart);
                            while (!argName.empty() && argName.front() == ' ')
                                argName.erase(0, 1);
                            while (!argName.empty() && argName.back() == ' ')
                                argName.pop_back();
                            auto it = paramToPort.find(argName);
                            if (it != paramToPort.end() && it->second < derivedTilingParams.portParams.size()) {
                                int64_t val;
                                if (std::string(funcName) == "aie::get_num_rounds")
                                    val = derivedTilingParams.portParams[it->second].numRounds;
                                else
                                    val = derivedTilingParams.portParams[it->second].bufferSize;
                                std::string replacement = std::to_string(val);
                                singleKernelBody.replace(pos, argEnd + 1 - pos, replacement);
                                pos += replacement.size();
                            } else {
                                pos = argEnd + 1;
                            }
                        }
                    }

                    auto replaceSimpleCall = [&](const std::string &call, int64_t val) {
                        std::string replacement = std::to_string(val);
                        size_t pos = 0;
                        while ((pos = singleKernelBody.find(call, pos)) != std::string::npos) {
                            singleKernelBody.replace(pos, call.size(), replacement);
                            pos += replacement.size();
                        }
                    };
                    replaceSimpleCall("aie::get_tile_rows()", derivedTilingParams.tileM > 0
                                                                  ? derivedTilingParams.tileM
                                                                  : derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_tile_cols()", derivedTilingParams.tileN > 0
                                                                  ? derivedTilingParams.tileN
                                                                  : derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_data_row()", derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_data_col()", derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_k_dim()", derivedTilingParams.kDim);
                    // Two-level tiling query functions
                    replaceSimpleCall("aie::get_tile_m()", derivedTilingParams.tileM > 0
                                                               ? derivedTilingParams.tileM
                                                               : derivedTilingParams.tileRows);
                    replaceSimpleCall("aie::get_tile_n()", derivedTilingParams.tileN > 0
                                                               ? derivedTilingParams.tileN
                                                               : derivedTilingParams.tileCols);
                    replaceSimpleCall("aie::get_effective_k()", derivedTilingParams.effectiveK > 0
                                                                    ? derivedTilingParams.effectiveK
                                                                    : derivedTilingParams.kDim);
                    replaceSimpleCall("aie::get_k_rounds()", derivedTilingParams.kRounds);
                    replaceSimpleCall("aie::get_spatial_m_rounds()", derivedTilingParams.spatialMRounds);
                    replaceSimpleCall("aie::get_spatial_n_rounds()", derivedTilingParams.spatialNRounds);
                }

                // Prepend user macros to kernel body for multi-tile path
                std::string kernelBodyWithMacros = singleKernelBody;
                if (!userMacroDefines.empty() && !singleKernelBody.empty()) {
                    std::string macroBlock = "// User macro definitions from source file\n";
                    for (const auto &macro : userMacroDefines) {
                        macroBlock += macro + "\n";
                    }
                    macroBlock += "\n";
                    kernelBodyWithMacros = macroBlock + singleKernelBody;
                }

                // Run pipeline (single kernel — no suffix, no append mode)
                if (!TilingLinalgPipeline::runPipeline(ctx, module, outputDir, kernelBodyWithMacros,
                                                       singleKernelFuncName, parsedDebugLevel, userRewrittenSource,
                                                       tensors, effectiveMaxPPBytes, aieGenStr)) {
                    llvm::errs() << "[TilingLinalg] Pipeline FAILED.\n";
                    std::exit(1);
                }
            }

            llvm::outs() << "[TilingLinalg] Pipeline completed. Output in: " << outputDir << "\n";
            // Copy user #include "..." headers to outputDir so host.cc is self-contained.
            if (!userSourceDir.empty() && !userRewrittenSource.empty()) {
                std::regex includeRegex(R"(#\s*include\s*\"([^\"]+)\")");
                auto begin = std::sregex_iterator(userRewrittenSource.begin(), userRewrittenSource.end(), includeRegex);
                auto end = std::sregex_iterator();
                for (auto it = begin; it != end; ++it) {
                    std::string headerName = (*it)[1].str();
                    // Skip runtime headers already on the compiler include path
                    if (headerName == "aie_runtime.h" || headerName == "aie_runtime_debug.h" ||
                        headerName == "xaiengine.h")
                        continue;
                    std::string srcPath = userSourceDir + "/" + headerName;
                    std::string dstPath = outputDir + "/" + headerName;
                    // Only copy if the file exists in the user source directory
                    std::ifstream src(srcPath, std::ios::binary);
                    if (src.good()) {
                        std::ofstream dst(dstPath, std::ios::binary);
                        dst << src.rdbuf();
                        llvm::outs() << "[TilingLinalg] Copied user header: " << headerName << " -> " << dstPath
                                     << "\n";
                    }
                }
            }
        } else {
            // ---- EXISTING PATH: single-tile ----
            std::error_code error_code;
            llvm::outs() << "Exporting File: " << std::string(AOUT) + "./host.cc" << "\n";
            llvm::raw_fd_ostream outFile(std::string(AOUT) + "./host.cc", error_code, llvm::sys::fs::OF_None);
            if (error_code) {
                llvm::errs() << "Error opening file: " << error_code.message() << "\n";
                return; // Exit early if there's an error
            }
            for (auto &kv : globalKernelFuncs) {
                const std::string &kname = kv.first;
                std::string dmOffsetsPath = std::string(AOUT) + "./kernelcfg/" + kname + "/dm_offsets.h";
                if (llvm::sys::fs::exists(dmOffsetsPath)) {
                    outFile << "#include \"kernelcfg/" << kname << "/dm_offsets.h\"\n";
                    llvm::outs() << "[aiehlc] Injected dm_offsets.h for " << kname << " into host.cc\n";
                }
            }
            // Emit strong g_runtime_debug_level override if user set #pragma aie_debug_level
            if (parsedDebugLevel >= 0) {
                outFile << "// Override runtime debug level (from #pragma aie_debug_level)\n";
                outFile << "int g_runtime_debug_level = " << parsedDebugLevel << ";\n\n";
            }
            // fd.write(RewriteBuf->data(), RewriteBuf->size());
            RewriteBuf->write(outFile);
            /*
            auto& sc = Rewrite.getSourceMgr();
            auto mid = sc.getMainFileID();
            llvm::StringRef sourceCode = sc.getBufferData(mid);
            llvm::outs() << "Original Source Code:\n" << sourceCode << "\n";
            */
            const std::string fname = "aie.mlir";
            auto mlirstr = aiefrontend.dumpir();
            std::ofstream ofs(fname);
            if (ofs.is_open()) {
                ofs << mlirstr;
                ofs.close();
            }
            aiefrontend.RunPass(fname);
        }
        //  std::cout << "----------------end-----------------" << std::endl;
  	}
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                   StringRef file) override {

				//disable the error information
			    // auto& dg = CI.getDiagnostics();
				// auto& Context = CI.getASTContext();
				// unsigned IgnoredErrorID = Context.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
				// 		"This error is ignored");
				// dg.setClient(new IgnoringDiagConsumer(), /*ShouldOwn=*/true);
				// CI.createDiagnostics(new IgnoringDiagConsumer(), /*ShouldOwnClient=*/false);

				TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());			
        return std::make_unique<MyASTConsumer>(&TheRewriter,&CI.getASTContext(), &aiefrontend);
    }
private:
		std::string code;
		AieFrontEnd aiefrontend;
		Rewriter TheRewriter;
		std::unique_ptr<clang::FileManager> FileMgr;
		//FileManager FileMgr;
};
extern void dumpmlir();

void trycreatefolder(std::string dirPath) {
	///*
	std::string command = "mkdir -p " + dirPath; 
	int result = system(command.c_str());           // Run the command

	if (result == 0) {
		// std::cout << "Directory created." << std::endl;
	} else {
		std::cerr << "Failed to create directory." << std::endl;
	}
}

static llvm::cl::opt<std::string> IncludePath(
			"I", llvm::cl::desc("Specify include path (-I)"), llvm::cl::value_desc("path"));

llvm::cl::OptionCategory MyToolCategory("aiehlc options");
static llvm::cl::opt<bool> LLVMAie(
			"use-llvm-aie", llvm::cl::desc("Enable use of LLVM AIE"), llvm::cl::init(false), llvm::cl::cat(MyToolCategory));

int main(int argc, const char **argv) {

	trycreatefolder(AOUT);
	// std::cout << "main--" << IncludePath.getValue() << std::endl;
	// /*
    //CommonOptionsParser exp(argc, argv, MyToolCategory);
    ///*
	if (argc > 2) {
		if (strcmp(argv[1], "pass") == 0) {
			const char * mlirf = argv[2];
			AieFrontEnd af;
			std::string smlir(mlirf);
			af.Parser(smlir);
			return 0;
		}
	}
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory, llvm::cl::OneOrMore);
    if (!ExpectedParser) {
		llvm::errs() << "Error: " << llvm::toString(ExpectedParser.takeError()) << "\n";
        // Handle error
        return 1;
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();
	use_llvm_aie = LLVMAie;
    //*/
    /*
    auto& OriginalCompilations = OptionsParser.getCompilations();
    std::vector<CompileCommand> newCommands;
    for (const auto& SourceFile : OptionsParser.getSourcePathList()) {
        auto CompileCommandsForFile = OriginalCompilations.getCompileCommands(SourceFile);
	for (auto &Command : CompileCommandsForFile) {
            Command.CommandLine.push_back("-x");
            Command.CommandLine.push_back("c++");
	    newCommands.push_back(Command);
	}

    }
    /*
    FixedCompilationDatabase ( "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/", newCommands);
    auto NewCompilations = std::make_shared<FixedCompilationDatabase>(newCommands);
		        //    "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/", newCommands);
    //ClangTool Tool(*NewCompilations,
    //               OptionsParser.getSourcePathList());
    
    std::string Directory = "/scratch/staff/huaj/mlir/acompiler/build/../src/llvm/";
    const char* CompileArgs[]= {"-std=c++11", " -x", "c++"};
    int n = 3;
    auto Compilations = FixedCompilationDatabase::loadFromCommandLine(n, CompileArgs, Directory);
    */
		//dumpmlir();
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());

	if (!IncludePath.empty()) {
		// Add the -I include path argument before the source file arguments
		Tool.appendArgumentsAdjuster(
				getInsertArgumentAdjuster(("-I" + IncludePath).c_str(), ArgumentInsertPosition::BEGIN));
	}

    // Force C++20 for struct NTTP support in aie::port<T, SpatialPolicy>
    Tool.appendArgumentsAdjuster(getInsertArgumentAdjuster("-std=c++20", ArgumentInsertPosition::BEGIN));
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
