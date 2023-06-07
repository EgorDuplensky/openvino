// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <ie_common.h>
#include <string>
#include <vector>
#include <array>
#include "graph_aligners.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "common/dnnl_executor.h"
#include "executors/matmul.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/matmul_factory.hpp"
#include <graph.h>
#include "graph_aligners.hpp"
#include "nodes/executors/mvn_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class MatMul : public Node {
public:
    MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    bool canFuse(const NodePtr& node) const override;
    bool created() const override;

    InferenceEngine::Precision getRuntimePrecision() const override;
    size_t descInputNumbers() override {
        return getOriginalInputsNumber();
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() - 1;
    }

    ExecutorPtr createExecutor() override;
    void prepareParams() override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    const std::vector<impl_desc_type>& getDefaultImplPriority() override;
    bool canBeExecutedInInt8() const override;

    std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>>
    getSupportedMemoryConfigs(const std::vector<InferenceEngine::Precision>& inputPrecisions,
                              const std::vector<Shape>& inputShapes,
                              const std::vector<InferenceEngine::Precision>& outputPrecisions,
                              const std::vector<Shape>& outputShapes) override;

    enum class Feature {
        fuseMultiply,
        transposeInput,
    };

    void requestFeature(const Feature feature) {
        requestedFeatures.push_back(feature);
    }

// protected:
//     AttrPtr initPrimitiveAttr() override;
//     AttrPtr initPrimitiveAttr(const VectorDims& dims);

private:
    // void setPostOps(dnnl::primitive_attr &attr, const VectorDims& dims, bool initWeights);
    std::vector<InferenceEngine::Precision> inputPrecisions;
    std::vector<InferenceEngine::Precision> outputPrecisions;

    std::string errorPrefix;
    MatMulAttrs matmulAttrs;
    ExecutorPtr execPtr = nullptr;
    std::unique_ptr<Graph> execGraph = nullptr;
    std::vector<Feature> requestedFeatures;
    MatMulExecutorFactoryPtr factory;
    ExecutorContextPtr executionContext;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
