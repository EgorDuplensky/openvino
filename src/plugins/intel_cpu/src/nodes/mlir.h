// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "allocation_context.hpp"
#include "cpu_types.h"
#include "graph.h"
#include "graph_context.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

#ifdef ENABLE_MLIR_FOR_CPU
// Forward declarations for MLIR types
namespace mlir {
    class MLIRContext;
    class ModuleOp;
    class ExecutionEngine;
}
#endif

namespace ov {
namespace intel_cpu {
namespace node {

class Mlir : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    Mlir(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override {
        return getType() == Type::SubModel;
    }

    bool needShapeInfer() const override {
        return false;
    }

    bool needPrepareParams() const override {
        return false;
    }

    bool neverExecute() const override {
        return false;
    }

    bool isExecutable() const override {
        return true;
    }

    void getSupportedDescriptors() override {};
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(const dnnl::stream&) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    int registerToAllocationContext(int offset, AllocationContext& context) override;

    const Graph& graph() const {
        return m_graph;
    }

private:
    std::shared_ptr<const ov::Model> m_body;
    Graph m_graph;
    std::shared_ptr<Executor> m_executor;

#ifdef ENABLE_MLIR_FOR_CPU
    // MLIR-specific members
    std::unique_ptr<mlir::MLIRContext> m_mlir_context;
    std::unique_ptr<mlir::ModuleOp> m_mlir_module;
    std::unique_ptr<mlir::ExecutionEngine> m_execution_engine;
    
    // MLIR helper methods
    std::unique_ptr<mlir::ModuleOp> convertToMLIR();
    std::unique_ptr<mlir::ExecutionEngine> compileMLIR(mlir::ModuleOp* module);
    void executeMLIR();
    mlir::Value convertMatMulToMLIR(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value lhs, mlir::Value rhs);
#endif
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
