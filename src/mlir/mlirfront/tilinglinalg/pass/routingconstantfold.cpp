#include "routingconstantfold.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/SymbolTable.h"

static mlir::Attribute findConstantValue(mlir::MLIRContext *context, mlir::PatternRewriter &rewriter, 
  std::string calleeName,mlir::Value value) 
{
  if (auto definingOp = value.getDefiningOp()) {
    //if (auto arithConst = dyn_cast<mlir::arith::ConstantOp>(definingOp)) {
    //  return arithConst.getValue();
    //}
    if (auto callOp = dyn_cast<mlir::emitc::CallOp>(definingOp)) {
      if (1) {//callOp.getCallee() == "XAie_TileLoc") {
        llvm::SmallVector<int64_t, 2> argValues;
        for (mlir::Value operand : callOp.getOperands()) {
           auto *definingOp = operand.getDefiningOp();
          // The pattern can only succeed if the operand is a constant.
           if (!definingOp) {
                return nullptr;
           }
            // Check if the defining op is an arith.constant.
           if (auto constOp = llvm::dyn_cast<mlir::emitc::ConstantOp>(definingOp)) {
                // Extract the integer attribute.
                if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                    argValues.push_back(intAttr.getInt());
                } else {
                    return nullptr;
                }
            } else {
                // If the defining op is not a constant, this pattern cannot apply.
                return nullptr;
            }
        }
        std::stringstream ss;
        ss << callOp.getCallee().str() << "(";
        for (size_t i = 0; i < argValues.size(); ++i) {
            ss << argValues[i];
            if (i < argValues.size() - 1) {
                ss << ",";
            }
        }
        ss << ")";
        std::string resultString = ss.str();
        return mlir::emitc::OpaqueAttr::get(context, resultString);
      }
      
    }
    if (auto emitcConst = dyn_cast<mlir::emitc::ConstantOp>(definingOp)) {
      if (auto strAttr = emitcConst.getValue().dyn_cast<mlir::StringAttr>()) {
        return mlir::emitc::OpaqueAttr::get(rewriter.getContext(), strAttr);
      }
      return emitcConst.getValue();
    }
  }
  return nullptr;
}

struct FoldConstantOperandsIntoCall : public mlir::OpRewritePattern<mlir::emitc::CallOp> {
  using OpRewritePattern<mlir::emitc::CallOp>::OpRewritePattern;
  FoldConstantOperandsIntoCall(MLIRContext* ctx, RoutingTopology & router):OpRewritePattern(ctx) {

  }

  mlir::LogicalResult matchAndRewrite(mlir::emitc::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const override {
    // --- Part 1: Deconstruct the call to find new operands and args ---
    // (This logic is the same as the previous "universal" pattern)
    std::string calleeName = callOp.getCallee().str();
    llvm::SmallVector<mlir::Attribute> newArgs;
    auto existingArgsAttr = callOp->getAttrOfType<mlir::ArrayAttr>("args");
    if (existingArgsAttr) {
      newArgs.assign(existingArgsAttr.begin(), existingArgsAttr.end());
    } else {
      newArgs.resize(callOp.getNumOperands(), rewriter.getIndexAttr(0));
    }

    llvm::SmallVector<mlir::Value> newOperands;
    bool changed = false;
    unsigned operandIndex = 0;

    int operandIdx = 0;
    for (size_t i = 0; i < newArgs.size(); ++i) {
      if (auto intAttr = newArgs[i].dyn_cast<mlir::IntegerAttr>()) {
          if (intAttr.getType().isa<mlir::IndexType>()) {
            if (operandIndex >= callOp.getNumOperands()) return mlir::failure();
            mlir::Value currentOperand = callOp.getOperands()[operandIndex];
            if (mlir::Attribute constValue = findConstantValue(getContext(), rewriter,calleeName,currentOperand)) {
              newArgs[i] = constValue;
              changed = true;
            } else {
              //all the func parameter should present in the newArgs, for the operand ,it should have a slot in newArgs but
              //declare it is operand by using n:Index, n is the index of operand paramerter, for example func(%a,%b) , 
              //1:Index means the second operand , it is %b, the newArgs can be [1:i32, 1:Index, 0:Index] then after convert
              //into c like func(1, %b, %a)
              newArgs[i] = rewriter.getIndexAttr(operandIdx++);
              newOperands.push_back(currentOperand);
            }
            operandIndex++;
        }
      }
    }

    if (!changed) {
      return mlir::failure();
    }

    // --- Part 2: Find and update the function declaration ---

    // Find the module, which acts as the symbol table.
    auto module = callOp->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      return callOp.emitError("must be nested inside a module");
    }

    // Look up the function declaration by its symbol name.
    auto funcDecl = module.lookupSymbol<mlir::emitc::FuncOp>(callOp.getCallee());
    if (!funcDecl) {
      // If the function is not declared, we can't update it.
      // This might be an error or intentional, depending on the toolchain.
      // For safety, we fail the pattern.
      return callOp.emitError("cannot find declaration for callee");
    }

    // Create the new function type based on the remaining operands.
    auto uses = mlir::SymbolTable::getSymbolUses(funcDecl, module);
    if (uses && llvm::hasSingleElement(*uses) && uses->begin()->getUser() == callOp)  {
        // It's safe to delete. The only user is the op we are about to replace.
        rewriter.eraseOp(funcDecl);
    } else {
        //llvm::outs()<<"cannot delete func decl; it has other uses\n";
    }
    // --- Part 3: Build the new call op and replace the old one ---

    mlir::ArrayAttr finalArgsAttr = rewriter.getArrayAttr(newArgs);
    mlir::NamedAttribute argsNamedAttr(rewriter.getStringAttr("args"), finalArgsAttr);

    mlir::OperationState state(callOp.getLoc(), mlir::emitc::CallOpaqueOp::getOperationName());

    mlir::emitc::CallOpaqueOp::build( rewriter,state,callOp.getResultTypes(),rewriter.getStringAttr(calleeName), newOperands);
    
    state.addAttributes(argsNamedAttr);
    /*
    for (const auto &attr : callOp->getAttrs()) {
      if (attr.getName() != "args" && attr.getName() != "callee" && attr.getName() != "operand_segment_sizes") {
        state.addAttribute(attr.getName(), attr.getValue());
      }
    }
    */

    auto newCallOp = rewriter.create(state);
    rewriter.replaceOp(callOp, newCallOp->getResults());

    return mlir::success();
  };
};

struct RemoveDeadCallOp : public mlir::OpRewritePattern<mlir::emitc::CallOp> {
  using OpRewritePattern<mlir::emitc::CallOp>::OpRewritePattern;
  RemoveDeadCallOp(MLIRContext* ctx):OpRewritePattern(ctx) {}
  mlir::LogicalResult matchAndRewrite(mlir::emitc::CallOp callOp,
                                      mlir::PatternRewriter &rewriter) const override {
    std::string calleeName = callOp.getCallee().str();
    if (callOp.use_empty()) {
      rewriter.eraseOp(callOp);
    }
    return success();
  }
};

struct RemoveDeadCallopOpaqueOp : public mlir::OpRewritePattern<mlir::emitc::CallOpaqueOp> {
  using OpRewritePattern<mlir::emitc::CallOpaqueOp>::OpRewritePattern;
  RemoveDeadCallopOpaqueOp(MLIRContext* ctx):OpRewritePattern(ctx) {}
  mlir::LogicalResult matchAndRewrite(mlir::emitc::CallOpaqueOp callOp,
                                      mlir::PatternRewriter &rewriter) const override {
    std::string calleeName = callOp.getCallee().str();
    if (callOp.use_empty() && calleeName == "XAie_TileLoc") {
      rewriter.eraseOp(callOp);
    }
    return success();
  }
};

struct RemoveConstantOp : public mlir::OpRewritePattern<mlir::emitc::ConstantOp> {
  using OpRewritePattern<mlir::emitc::ConstantOp>::OpRewritePattern;
  RemoveConstantOp(MLIRContext* ctx):OpRewritePattern(ctx) {}
  mlir::LogicalResult matchAndRewrite(mlir::emitc::ConstantOp constantop,
                                      mlir::PatternRewriter &rewriter) const override {
    if (constantop.use_empty()) {
      rewriter.eraseOp(constantop);
    }
    return success();
  }
};

void RoutingConstantFoldPass::runOnOperation() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    // Add our pattern to the set of patterns to be applied.
    //FIXE ME
    RoutingTopology RR("Gen2");
    patterns.add<FoldConstantOperandsIntoCall>(&ctx, RR);
    patterns.add<RemoveDeadCallOp>(&ctx);
    patterns.add<RemoveDeadCallopOpaqueOp>(&ctx);
    patterns.add<RemoveConstantOp>(&ctx);
    
    for (auto* dialect : ctx.getLoadedDialects()) {
        dialect->getCanonicalizationPatterns(patterns);
    }
    for (mlir::RegisteredOperationName op : ctx.getRegisteredOperations()) {
        op.getCanonicalizationPatterns(patterns, &ctx);
    }

    // Apply the patterns greedily.
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
}
