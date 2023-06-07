// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include "dnnl_matmul.hpp"
#include "ie_precision.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/matmul.hpp"
#include "nodes/executors/matmul_builder.hpp"
#include "nodes/fake_quantize.h"
#include "utils/general_utils.h"
#include "dnnl_postops_composer_new.h"
#include "nodes/executors/matmul_key.hpp"

namespace ov {
namespace intel_cpu {

namespace precision {
#define DEFINE_PRECISION_CONSTANT(x) const InferenceEngine::Precision x = InferenceEngine::Precision::x
DEFINE_PRECISION_CONSTANT(UNSPECIFIED);
DEFINE_PRECISION_CONSTANT(MIXED);
DEFINE_PRECISION_CONSTANT(FP32);
DEFINE_PRECISION_CONSTANT(FP16);
DEFINE_PRECISION_CONSTANT(BF16);
DEFINE_PRECISION_CONSTANT(FP64);
DEFINE_PRECISION_CONSTANT(Q78);
DEFINE_PRECISION_CONSTANT(I16);
DEFINE_PRECISION_CONSTANT(U4);
DEFINE_PRECISION_CONSTANT(U8);
DEFINE_PRECISION_CONSTANT(I4);
DEFINE_PRECISION_CONSTANT(I8);
DEFINE_PRECISION_CONSTANT(U16);
DEFINE_PRECISION_CONSTANT(I32);
DEFINE_PRECISION_CONSTANT(U32);
DEFINE_PRECISION_CONSTANT(I64);
DEFINE_PRECISION_CONSTANT(U64);
DEFINE_PRECISION_CONSTANT(BIN);
DEFINE_PRECISION_CONSTANT(BOOL);
DEFINE_PRECISION_CONSTANT(CUSTOM);
#undef DEFINE_PRECISION_CONSTANT
} // namespace precision

using namespace precision;

class DnnlMatMulExecutorBuilder : public MatMulExecutorBuilder {
public:
    bool isSupported(const MatMulKey& key) const override {
        // TODO: add correct conditions
        return true;
    }

    std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>>
    getMemoryConfig(const std::vector<InferenceEngine::Precision>& inputPrecisions,
                    const std::vector<Shape>& inputShapes,
                    const std::vector<InferenceEngine::Precision>& outputPrecisions,
                    const std::vector<Shape>& outputShapes) const override {
        auto createMemoryDesc = [](const LayoutType layoutType, const InferenceEngine::Precision precision, const Shape& shape) {
            const auto& creatorsMap = BlockedDescCreator::getCommonCreators();
            return creatorsMap.at(layoutType)->createSharedDesc(precision, shape);
        };

        static std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>> memoryConfigs {
            {{createMemoryDesc(LayoutType::ncsp, FP32, inputShapes[0]), createMemoryDesc(LayoutType::ncsp, FP32, inputShapes[1])},
             {createMemoryDesc(LayoutType::ncsp, FP32, outputShapes[0])}}
        };

        return memoryConfigs;
    }

    MatMulExecutorPtr instantiate(const MatMulKey& key, ExecutorContext::CPtr context) override {
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        std::unordered_map<int, MemoryPtr> postOpsArgs;
        fuse(key, attr, postOpsArgs, context);
        return emit(key, attr, postOpsArgs, context);
    }

    // should also return actual src and dest descs to support any format
    MatMulExecutorPtr emit(const MatMulKey& key,
                           const dnnl::primitive_attr& attr,
                           std::unordered_map<int, MemoryPtr>& postOpsArgs, ExecutorContext::CPtr context) const {
        auto builder = [&](const MatMulKey& key) -> DnnlMatMulExecutorPtr {
            return std::make_shared<DnnlMatMulExecutor>(context, key.matmulAttrs, key.srcDescs, key.dstDescs, attr);
        };

        auto res = context->getRuntimeCache().lock()->getOrCreate(key, builder);
        auto& executor = res.first;
        executor->setPostOpArgs(postOpsArgs);

        return executor;
    }

    // @todo pass only necessary info instead of full key
    void fuse(const MatMulKey& key, dnnl::primitive_attr& attr, std::unordered_map<int, MemoryPtr>& postOpsArgs, ExecutorContext::CPtr _context) const {
        if (key.postOps.empty())
            return;

        auto dims = key.dstDescs[0]->getShape().getStaticDims();
        auto isINT8 = one_of(key.srcDescs[0]->getPrecision(), InferenceEngine::Precision::U8, InferenceEngine::Precision::I8)
            && key.srcDescs[1]->getPrecision() == InferenceEngine::Precision::I8;
        auto outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(key.dstDescs[0]->getPrecision());

        DnnlPostOpsComposerNew dnnlpoc(key.postOps, _context->getEngine(), attr, postOpsArgs, dims, dims.size() - 1,
                                       isINT8, 1 << (dims.size() - 1), {}, key.matmulAttrs.withBias, outputDataType);
        dnnlpoc.compose();
    }

// private:
    // std::shared_ptr<DnnlMatMulExecutor> executor;
    // std::vector<MemoryDescPtr> _srcDescs;
    // std::vector<MemoryDescPtr> _dstDescs;
    // MatMulAttrs _matmulAttrs;
    // ExecutorContext::CPtr _context;
};

}   // namespace intel_cpu
}   // namespace ov
