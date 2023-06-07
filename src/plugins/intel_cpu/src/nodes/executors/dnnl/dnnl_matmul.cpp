// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_matmul.hpp"
#include "dnnl_postops_composer_new.h"
#include "nodes/executors/matmul_key.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include "nodes/executors/matmul.hpp"
#include "onednn/dnnl.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

dnnl::matmul::primitive_desc getPrimitiveDesc(const dnnl::engine& engine,
                                              const MatMulAttrs& matmulAttrs,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs,
                                              const dnnl::primitive_attr &attr,
                                              const std::vector<impl_desc_type>& implPriorities);

DnnlMatMulExecutor::DnnlMatMulExecutor(const MatMulArgs &args) :
    MatMulExecutor(args.context, args.key.matmulAttrs),
    Node("MatMul", "DnnlMatMulExecutor", args.context->getGraphContext()),
    stream(dnnl::stream(MatMulExecutor::context->getEngine())) {
    const auto& srcDescs = args.key.srcDescs;
    const auto& dstDescs = args.key.dstDescs;
    dnnl::primitive_attr attr;
    fuse(args.key, attr, MatMulExecutor::context);
    // @todo should scratchpad mode be a part of configuration or graph context;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // @todo use only key or matmulArgs
    auto builder = [&](const MatMulKey& key) -> std::pair<std::shared_ptr<dnnl::matmul>, MemoryPtr> {
        auto prim_desc = getPrimitiveDesc(MatMulExecutor::context->getEngine(), matmulAttrs, srcDescs, dstDescs, attr, MatMulExecutor::context->getImplPriorities());
        implType = parse_impl_name(prim_desc.impl_info_str());

        if (!prim_desc) {
            IE_THROW() << "Failed to create DnnlMatMulExecutor";
        }

        auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(prim_desc.scratchpad_desc());
        scratchpadMemory = MatMulExecutor::context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);

        return {std::make_shared<dnnl::matmul>(prim_desc), scratchpadMemory};
    };

    auto res = MatMulExecutor::context->getRuntimeCache().lock()->getOrCreate(args.key, builder);
    prim = res.first.first;
    scratchpadMemory = res.first.second;

    if (!prim)
        IE_THROW() << "Failed to create DnnlMatMulExecutor";
}

/* Example MatMul:
 * 2x128x512(T) * 2x128x512 = 2x512x512
 * First input 2x128x512(T) should be transposed
 * oneDNN requires memory::desc for this input to:
 * - change shapes configuration as if input already transposed (2x128x512) -> (2x512x128)
 * - provide transposed strides (66536, 128, 1) -> (66536, 1, 512)
 */
static VectorDims getStridesAndModifyShape(Shape& shape, const bool transpose) {
    const auto getRank = shape.getRank();

    VectorDims strides(getRank, 1);
    const auto& staticDims = shape.getStaticDims();
    for (size_t i = 1; i < getRank; i++) {
        strides[getRank - i - 1 ] = strides[getRank - i] * staticDims[getRank - i];
    }

    if (transpose && getRank > 1) {
        // form new shape
        auto dims = staticDims;
        std::swap(dims[getRank - 2], dims[getRank - 1]);
        shape = Shape{dims};
        // update strides
        strides[getRank - 1] = staticDims[getRank - 2];
        strides[getRank - 2] = 1;
    }

    return strides;
}

static dnnl::matmul::primitive_desc createDescriptor(const dnnl::engine& engine,
                                                     const MatMulAttrs& matmulAttrs,
                                                     const std::vector<MemoryDescPtr>& srcDescs,
                                                     const std::vector<MemoryDescPtr>& dstDescs,
                                                     const dnnl::primitive_attr &attr) {
    auto inputShape0 = srcDescs[0]->getShape();
    const VectorDims inStrides0 = getStridesAndModifyShape(inputShape0, matmulAttrs.transposeA);
    auto inDataDesc0 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[0]->getPrecision(), inputShape0, inStrides0);

    auto inputShape1 = srcDescs[1]->getShape();
    const VectorDims inStrides1 = getStridesAndModifyShape(inputShape1, matmulAttrs.transposeB);
    auto inDataDesc1 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[1]->getPrecision(), inputShape1, inStrides1);

    auto outputShape = dstDescs[0]->getShape();
    auto outDataDesc = std::make_shared<DnnlBlockedMemoryDesc>(dstDescs[0]->getPrecision(), outputShape);

    dnnl::matmul::primitive_desc matmul_desc;
    if (matmulAttrs.withBias) {
        // oneDNN matmul requires shape for bias desc to be the same rank
        VectorDims biasDims(outputShape.getRank(), 1);
        const auto& outDims = outputShape.getStaticDims();
        const auto chIdx = outputShape.getRank() - 1;
        biasDims[chIdx] = outDims[chIdx];
        const auto bdt = DnnlExtensionUtils::IEPrecisionToDataType(srcDescs[2]->getPrecision());
        auto biasDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasDims), bdt, dnnl::memory::format_tag::any);

        matmul_desc = dnnl::matmul::primitive_desc(engine,
                                                   inDataDesc0->getDnnlDesc(),
                                                   inDataDesc1->getDnnlDesc(),
                                                   biasDesc,
                                                   outDataDesc->getDnnlDesc(),
                                                   attr);
    } else {
        matmul_desc = dnnl::matmul::primitive_desc(engine,
                                                   inDataDesc0->getDnnlDesc(),
                                                   inDataDesc1->getDnnlDesc(),
                                                   outDataDesc->getDnnlDesc(),
                                                   attr);
    }

    return matmul_desc;
}

void DnnlMatMulExecutor::fuse(const MatMulKey& key, dnnl::primitive_attr& attr, ExecutorContext::CPtr _context) {
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

dnnl::matmul::primitive_desc getPrimitiveDesc(const dnnl::engine& engine,
                                              const MatMulAttrs& matmulAttrs,
                                              const std::vector<MemoryDescPtr>& srcDescs,
                                              const std::vector<MemoryDescPtr>& dstDescs,
                                              const dnnl::primitive_attr &attr,
                                              const std::vector<impl_desc_type>& implPriorities) {
    auto prim_desc = createDescriptor(engine, matmulAttrs, srcDescs, dstDescs, attr);
    auto first_desc = dnnl::matmul::primitive_desc(prim_desc.get());

    for (auto preferredImplType : implPriorities) {
        const bool found = DnnlExtensionUtils::find_implementation(prim_desc, preferredImplType);

        if (found)
            return prim_desc;
    }

    return first_desc;
}

void DnnlMatMulExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    std::unordered_map<int, dnnl::memory> primArgs;

    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMemory->getPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src[0]->getPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src[1]->getPrimitive();
    primArgs[DNNL_ARG_DST] = dst[0]->getPrimitive();
    if (matmulAttrs.withBias)
        primArgs[DNNL_ARG_BIAS] = src[2]->getPrimitive();

    for (auto & entry : postOpsArgs) {
        primArgs[entry.first] = entry.second->getPrimitive();
    }

    prim->execute(stream, primArgs);
}

// std::pair<Shape, Shape> DnnlMatMulExecutor::makeDummyInputShapes(const MatMulAttrs& matmulAttrs, const Shape& in0, const Shape& in1) {
//     if (in0.getRank() < 2 || in1.getRank() < 2) {
//         IE_THROW() << "Can't create dummy inputs with rank less 2";
//     }

//     if (in0.getRank() != in1.getRank()) {
//         IE_THROW() << "Can't create dummy inputs if input's rank not equal";
//     }

//     auto swapTranspDims = [&](VectorDims& in0, VectorDims& in1) {
//         if (matmulAttrs.transposeA) {
//             std::swap(in0[in0.size() - 1], in0[in0.size() - 2]);
//         }
//         if (matmulAttrs.transposeB) {
//             std::swap(in1[in1.size() - 1], in1[in1.size() - 2]);
//         }
//     };

//     auto inDims0 = in0.getDims();
//     auto inDims1 = in1.getDims();

//     auto minDims0 = in0.getMinDims();
//     auto maxDims0 = in0.getMaxDims();
//     auto minDims1 = in1.getMinDims();
//     auto maxDims1 = in1.getMaxDims();

//     swapTranspDims(inDims0, inDims1);
//     swapTranspDims(minDims0, minDims1);
//     swapTranspDims(maxDims0, maxDims1);

//     auto fillDummy = [&](size_t idx0, size_t idx1) {
//         if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
//             inDims0[idx0] = inDims1[idx1] = std::min(std::min(maxDims0[idx0], maxDims1[idx1]),
//                                             std::max(std::max(minDims0[idx0], minDims1[idx1]), static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
//         } else {
//             if (inDims0[idx0] == Shape::UNDEFINED_DIM && inDims1[idx1] != Shape::UNDEFINED_DIM) {
//                 if (inDims1[idx1] == 1 && minDims0[idx0] != Shape::UNDEFINED_DIM) {
//                     inDims0[idx0] = std::max<Dim>(minDims0[idx0], 1);
//                 } else {
//                     inDims0[idx0] = inDims1[idx1];
//                 }
//             } else if (inDims0[idx0] != Shape::UNDEFINED_DIM && inDims1[idx1] == Shape::UNDEFINED_DIM) {
//                 if (inDims0[idx0] == 1 && minDims1[idx1] != Shape::UNDEFINED_DIM) {
//                     inDims1[idx1] = std::max<Dim>(minDims1[idx1], 1);
//                 } else {
//                     inDims1[idx1] = inDims0[idx0];
//                 }
//             }
//         }
//     };

//     // fill k
//     fillDummy(inDims0.size() - 1, inDims1.size() - 2);

//     // fill m, n
//     if (inDims0[inDims0.size() - 2] == Shape::UNDEFINED_DIM) {
//         inDims0[inDims0.size() - 2] = std::min(maxDims0[inDims0.size() - 2],
//                                                std::max(minDims0[inDims0.size() - 2], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
//     }
//     if (inDims1[inDims1.size() - 1] == Shape::UNDEFINED_DIM) {
//         inDims1[inDims1.size() - 1] = std::min(maxDims1[inDims1.size() - 1],
//                                                std::max(minDims1[inDims1.size() - 1], static_cast<Dim>(MemoryDescUtils::DEFAULT_DUMMY_VAL)));
//     }

//     // fill batches
//     for (size_t i = 0; i < inDims0.size() - 2; i++) {
//         fillDummy(i, i);
//     }

//     swapTranspDims(inDims0, inDims1);

//     return {Shape(inDims0), Shape(inDims1)};
// }

// DnnlMatMulExecutor::Key::Key(const MatMulAttrs& matmulAttrs,
//                              const std::vector<MemoryDescPtr>& srcDescs,
//                              const std::vector<MemoryDescPtr>& dstDescs,
//                              const dnnl::primitive_attr &attr) :
//     matmulAttrs(matmulAttrs),
//     inp0(MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[0])),
//     inp1(MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[1])),
//     bias(matmulAttrs.withBias ? MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[2]) : nullptr),
//     out(MemoryDescUtils::convertToDnnlMemoryDesc(dstDescs[0])),
//     attr(attr) {}

// size_t DnnlMatMulExecutor::Key::hash() const {
//     using namespace dnnl::impl;
//     using namespace dnnl::impl::primitive_hashing;

//     size_t seed = 0;
//     seed = hash_combine(seed, matmulAttrs.transposeA);
//     seed = hash_combine(seed, matmulAttrs.transposeB);
//     for (const auto& ptr : {inp0, inp1, bias, out}) {
//         if (ptr) {
//             seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
//         }
//     }

//     seed = hash_combine(seed, get_attr_hash(*attr.get()));
//     return seed;
// }

// bool DnnlMatMulExecutor::Key::operator==(const Key& rhs) const {
//     bool retVal = true;
//     retVal = retVal && matmulAttrs.transposeA == rhs.matmulAttrs.transposeA;
//     retVal = retVal && matmulAttrs.transposeB == rhs.matmulAttrs.transposeB;

//     if (inp0 != rhs.inp0) {
//         retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
//     }
//     if (inp1 != rhs.inp1) {
//         retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
//     }
//     if (bias != rhs.bias) {
//         retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
//     }
//     if (out != rhs.out) {
//         retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
//     }
//     retVal = retVal && *attr.get() == *rhs.attr.get();
//     return retVal;
// }

}   // namespace intel_cpu
}   // namespace ov
