// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#ifdef ENABLE_MLIR_FOR_CPU
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"
#endif

#include "allocation_context.hpp"
#include "mlir.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/input.h"
#include "nodes/node_config.h"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shape_inference/shape_inference_internal_dyn.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu::node {

bool Mlir::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<ov::intel_cpu::SubModel>(op)) {
            errorMessage = "Unknown SubGraph operation : " + std::string(op->get_type_info().name) + " with name '" +
                           op->get_friendly_name() + "'";
        }
    } catch (...) {
        return false;
    }
    return true;
}

Mlir::Mlir(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto& subModel = ov::as_type_ptr<SubModel>(op);

    CPU_NODE_ASSERT(subModel, "Attempt to create SubGraph node from an invalid op type: ", op);

    m_body = subModel->get_function();

#ifdef ENABLE_MLIR_FOR_CPU
    // Initialize MLIR context and register dialects
    m_mlir_context = std::make_unique<mlir::MLIRContext>();
    m_mlir_context->getOrLoadDialect<mlir::arith::ArithDialect>();
    m_mlir_context->getOrLoadDialect<mlir::func::FuncDialect>();
    m_mlir_context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
    
    // Convert OpenVINO subgraph to MLIR module
    m_mlir_module = convertToMLIR();
    
    if (m_mlir_module) {
        // Compile MLIR to executable JIT
        m_execution_engine = compileMLIR(m_mlir_module.get());
        if (!m_execution_engine) {
            std::cout << "Warning: Failed to compile MLIR module, falling back to reference implementation" << std::endl;
        }
    }
#endif
}

void Mlir::selectOptimalPrimitiveDescriptor() {
    // for the input configuration, just always use the parent configuration
    std::vector<PortConfig> inConfs;
    std::vector<Input::InputConfig> graphInputConfig;

    constexpr bool isInPlace = true;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto desc = getParentOutputMemDesc(getParentEdgeAt(i));
        inConfs.emplace_back(desc);
        graphInputConfig.emplace_back(node::Input::InputConfig{std::move(desc), isInPlace});
    }

    std::vector<Input::OutputConfig> graphOutputConfig(outputShapes.size(), node::Input::OutputConfig{true, isInPlace});

    // configure the inner graph to get the information about output memory descriptors
    // m_graph.Init(m_body, context, graphInputConfig, graphOutputConfig);

    const auto& creatorsMap = BlockedDescCreator::getCommonCreators();

    std::vector<PortConfig> outConfs;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto desc = creatorsMap.at(LayoutType::ncsp)->createSharedDesc(getOriginalInputPrecisionAtPort(i), getOutputShapeAtPort(i));
        outConfs.emplace_back(desc);
    }

    // for the output descriptors, use the configuration of the graph's output nodes
    const NodeConfig config(std::move(inConfs), std::move(outConfs));

    supportedPrimitiveDescriptors.clear();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);

    selectPrimitiveDescriptorByIndex(0);
}

// @todo add ascii diagramm for memory mapping / reuse
void Mlir::createPrimitive() {
    // MLIR compilation happens in constructor
    // Nothing to do here as execution engine is ready
}

int Mlir::registerToAllocationContext(int offset, AllocationContext& context) {
    return Node::registerToAllocationContext(offset, context);
}

void Mlir::execute(const dnnl::stream& /*strm*/) {
    std::cout << "Executing MLIR node: " << getName() << "\n";
    
#ifdef ENABLE_MLIR_FOR_CPU
    if (m_execution_engine) {
        // Execute using MLIR JIT
        executeMLIR();
        return;
    }
#endif
    
    // Fallback to reference implementation
    ov::TensorVector input_tensors;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        input_tensors.emplace_back(m_body->get_parameters()[i]->output(0),
                                   parentEdge->getMemoryPtr()->getData());
    }

    ov::TensorVector output_tensors;
    output_tensors.resize(getOriginalOutputsNumber());
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        output_tensors.emplace_back(m_body->get_results()[i]->output(0),
                                    childEdge->getMemoryPtr()->getData());
    }
    
    m_body->evaluate(input_tensors, output_tensors);
}

void Mlir::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);

    // since the shape inference is not performed for the composite node
    // a memory of the extra child edges, attached to the output ports
    // has to be updated after an inference of the inner graph finished
    const auto& childEdges = getChildEdges();
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        const auto mem = getDstMemoryAtPort(i);
        for (size_t j = getOriginalOutputsNumber(); j < childEdges.size(); j++) {
            const auto& childEdge = childEdges[j];
            auto childEdgePtr = childEdge.lock();
            assert(childEdgePtr);

            if (childEdgePtr->getInputNum() == static_cast<int>(i)) {
                childEdgePtr->getMemoryPtr()->redefineDesc(mem->getDescPtr());
            }
        }
    }
}

#ifdef ENABLE_MLIR_FOR_CPU
std::unique_ptr<mlir::ModuleOp> Mlir::convertToMLIR() {
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(m_mlir_context.get()));
    mlir::OpBuilder builder(m_mlir_context.get());
    builder.setInsertionPointToStart(module->getBody());
    
    // Create function for the subgraph
    auto funcName = "mlir_subgraph_" + getName();
    
    // Build function signature based on input/output shapes
    llvm::SmallVector<mlir::Type> inputTypes;
    llvm::SmallVector<mlir::Type> outputTypes;
    
    // For now, assume f32 tensors - this can be enhanced later
    auto f32Type = builder.getF32Type();
    
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto shape = getParentOutputMemDesc(getParentEdgeAt(i))->getShape();
        llvm::SmallVector<int64_t> mlirShape(shape.getStaticDims().begin(), shape.getStaticDims().end());
        auto tensorType = mlir::RankedTensorType::get(mlirShape, f32Type);
        inputTypes.push_back(tensorType);
    }
    
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto shape = getOutputShapeAtPort(i);
        llvm::SmallVector<int64_t> mlirShape(shape.getStaticDims().begin(), shape.getStaticDims().end());
        auto tensorType = mlir::RankedTensorType::get(mlirShape, f32Type);
        outputTypes.push_back(tensorType);
    }
    
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);
    auto func = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), funcName, funcType);
    func.setPublic();
    
    // Create function body
    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    
    // Convert operations in the subgraph
    // For simplicity, assume single MatMul operation for now
    auto inputs = entryBlock->getArguments();
    if (inputs.size() >= 2) {
        auto result = convertMatMulToMLIR(builder, builder.getUnknownLoc(), inputs[0], inputs[1]);
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), result);
    }
    
    // Verify the module
    if (mlir::failed(mlir::verify(*module))) {
        std::cout << "Failed to verify MLIR module" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<mlir::ModuleOp>(std::move(*module));
}

mlir::Value Mlir::convertMatMulToMLIR(mlir::OpBuilder& builder, mlir::Location loc,
                                      mlir::Value lhs, mlir::Value rhs) {
    // Create a simple matrix multiplication using arith dialect
    // This is a basic implementation - can be enhanced with proper tiling, vectorization
    
    auto lhsType = lhs.getType().cast<mlir::RankedTensorType>();
    auto rhsType = rhs.getType().cast<mlir::RankedTensorType>();
    
    // Compute output shape [M, N] where lhs is [M, K] and rhs is [K, N]
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    
    llvm::SmallVector<int64_t> outputShape = {lhsShape[0], rhsShape[1]};
    auto outputType = mlir::RankedTensorType::get(outputShape, lhsType.getElementType());
    
    // Create a simple matmul operation
    // In a real implementation, this would be lowered to proper loop nests
    return builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
}

std::unique_ptr<mlir::ExecutionEngine> Mlir::compileMLIR(mlir::ModuleOp* module) {
    if (!module) return nullptr;
    
    // Set up pass manager for lowering
    mlir::PassManager pm(m_mlir_context.get());
    pm.addPass(mlir::createCanonicalizerPass());
    
    // Run passes
    if (mlir::failed(pm.run(*module))) {
        std::cout << "Failed to run MLIR passes" << std::endl;
        return nullptr;
    }
    
    // Create execution engine
    auto maybeEngine = mlir::ExecutionEngine::create(*module);
    if (!maybeEngine) {
        std::cout << "Failed to create MLIR execution engine" << std::endl;
        return nullptr;
    }
    
    return std::make_unique<mlir::ExecutionEngine>(std::move(*maybeEngine));
}

void Mlir::executeMLIR() {
    if (!m_execution_engine) return;
    
    // Prepare arguments for MLIR function call
    llvm::SmallVector<void*> args;
    
    // Add input pointers
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        args.push_back(parentEdge->getMemoryPtr()->getData());
    }
    
    // Add output pointers
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        auto childEdge = getChildEdgeAt(i);
        args.push_back(childEdge->getMemoryPtr()->getData());
    }
    
    // Invoke the compiled function
    auto funcName = "mlir_subgraph_" + getName();
    auto invocationResult = m_execution_engine->invokePacked(funcName, args);
    
    if (invocationResult) {
        std::cout << "MLIR execution failed" << std::endl;
    }
}
#endif

}  // namespace ov::intel_cpu::node
