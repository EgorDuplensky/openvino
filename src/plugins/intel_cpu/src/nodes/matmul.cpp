// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul.h"

#include "ie_common.h"
#include "ie_precision.hpp"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "cpu_types.h"
#include "eltwise.h"

#include <cstddef>
#include <numeric>
#include <string>
#include <vector>
#include <memory>
#include "common/cpu_memcpy.h"
#include <ngraph/opsets/opset1.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "fake_quantize.h"
#include "nodes/executors/matmul_factory.hpp"
#include "post_ops.hpp"
#include "utils/general_utils.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include "shape_inference/custom/matmul.hpp"
using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {
namespace {


bool canBeExecutedInInt81(const Precision& firstInput, const Precision& secondInput) {
    return one_of(firstInput, Precision::U8, Precision::I8) && secondInput == Precision::I8;
}
} // namespace

bool MatMul::canBeExecutedInInt8() const {
    auto firstInputPrecision = getOriginalInputPrecisionAtPort(0);
    auto secondInputPrecision = getOriginalInputPrecisionAtPort(1);

    return one_of(firstInputPrecision, Precision::U8, Precision::I8) && secondInputPrecision == Precision::I8;
}

bool MatMul::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);
        if (!matMul) {
            errorMessage = "Only opset1 MatMul operation is supported";
            return false;
        }

        for (size_t i = 0; i < matMul->get_input_size(); i++) {
            const auto inShapeRank = matMul->get_input_partial_shape(i).rank().get_length();
            if (inShapeRank < 2) {
                errorMessage = "Unsupported rank: " + std::to_string(inShapeRank) + " on " + std::to_string(i) + " input";
                return false;
            }
        }

        const auto outShapeRank = matMul->get_output_partial_shape(0).rank().get_length();
        if (outShapeRank < 2) {
            errorMessage = "Unsupported rank: " + std::to_string(outShapeRank) + " on output";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MatMul::MatMul(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) :
    Node(op, context, MMShapeInferFactory(op)) {
    std::string errorMessage;
    errorPrefix = "MatMul node with name '" + getName() + "'";

    if (!isSupportedOperation(op, errorMessage))
        IE_THROW(NotImplemented) << errorMessage;

    const auto matMul = std::dynamic_pointer_cast<const ngraph::opset1::MatMul>(op);

    if (!matMul) {
        IE_THROW(NotImplemented) << "Operation with name " << op->get_friendly_name() << ":" << op->get_type_name() <<
            " is not an instance of MatMul from opset1";
    }

    matmulAttrs.transposeA = matMul->get_transpose_a();
    matmulAttrs.transposeB = matMul->get_transpose_b();

    // @todo create executor context in constructor
    executionContext = std::make_shared<ExecutorContext>(context, getImplPriority());
    // @todo pass factory to constructor?
    // Do we need to have the distinct factories for every MatMul node?
    factory = std::make_shared<MatMulExecutorFactory>(executionContext);
}

bool MatMul::canFuse(const NodePtr& node) const {
    // WA for CVS-84056: oneDNN brgemm impl has problem with per-OC binary-postOps for MatMul with 6D inputs
    if (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            if (eltwiseNode->getBroadcastingPolicy() == Eltwise::BroadcastingPolicy::PerChannel) {
                auto rank = getInputShapeAtPort(0).getRank();
                if (rank > 4) {
                    DEBUG_LOG("skip fusing non-perTensor Eltwise:", eltwiseNode->getName(), " into 6D MatMul:", getName());
                    return false;
                }
            }
        }
    }

    //  Consider the case when Matmul doesn't support execution in int8, but is getting fused with FQ with int8 output.
    //  Then the Matmul will change its output precision to fp32. If fusing FQ into matmul, there would be reorder inserted
    //  after matmul. In some bert model, this reorder causes great perf degradation.
    //  Todo: Remove this if onednn primitive support U8 output with floating input.
    if (node->getType() == Type::FakeQuantize && one_of(node->getOriginalOutputPrecisionAtPort(0), Precision::I8, Precision::U8) &&
        !canBeExecutedInInt8() &&
        getOriginalInputPrecisionAtPort(0) == InferenceEngine::Precision::FP32 )
        return false;
    return canFuseSimpleOperation(node);
}

// void MatMul::setPostOps(dnnl::primitive_attr& attr, const VectorDims& dims, bool initWeights = false) {
//     dnnl::post_ops ops;

//     dnnl::memory::data_type outputDataType = DnnlExtensionUtils::IEPrecisionToDataType(outputPrecisions[0]);

//     bool isINT8 = canBeExecutedInInt8();

//     DnnlPostOpsComposer dnnlpoc(
//         getEngine(), attr, ops, postOpsArgs, dims, dims.size() - 1, isINT8, 1 << (dims.size() - 1), getDQScales(), matmulAttrs.withBias);

//     for (size_t i = 0; i < fusedWith.size(); ++i) {
//         auto& node = fusedWith[i];
//         bool isLastPostOp = (i == (fusedWith.size() - 1));

//         if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
//             eltwiseNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
//             continue;
//         }

//         if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
//             fakeQuantizeNode->appendAttrPostOps(dnnlpoc, isLastPostOp, outputDataType);
//             continue;
//         }

//         IE_THROW() << "Fusing of " << NameFromType(node->getType()) << " operation to " << NameFromType(this->getType())
//                    << " node is not implemented";
//     }

//     attr.set_post_ops(ops);
// }

// Node::AttrPtr MatMul::initPrimitiveAttr(const VectorDims &dims) {
//     auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());

//     setPostOps(*attr, dims, true);

//     (*attr).set_scratchpad_mode(dnnl::scratchpad_mode::user);

//     return attr;
// }

// Node::AttrPtr MatMul::initPrimitiveAttr() {
//     auto dummyShape = MemoryDescUtils::makeDummyShape(getOutputShapeAtPort(0));
//     return initPrimitiveAttr(dummyShape.getStaticDims());
// }

void MatMul::initSupportedPrimitiveDescriptors() {
    matmulAttrs.withBias = getOriginalInputsNumber() == 3;

    inputPrecisions = getOriginalInputPrecisions();
    outputPrecisions = getOriginalOutputPrecisions();
    if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41)) {
        if (inputPrecisions[0].size() != inputPrecisions[1].size())
            inputPrecisions[0] = inputPrecisions[1] = getMaxPrecision(getOriginalInputPrecisions());

        // fallback to fp32 for any precision that cannot be handled natively
        if ((!one_of(inputPrecisions[0] , Precision::U8, Precision::I8, Precision::BF16, Precision::FP32) ||
            !one_of(inputPrecisions[1] , Precision::I8, Precision::BF16, Precision::FP32))) {
            outputPrecisions[0] = inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        }

        if (!fusedWith.empty()) {
            outputPrecisions[0] = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        }

        if (!canBeExecutedInInt81(inputPrecisions[0], inputPrecisions[1]) && one_of(outputPrecisions[0], Precision::U8, Precision::I8))
            outputPrecisions[0] = Precision::FP32; // INT output is not supported for non-INT inputs
    } else {
        inputPrecisions[0] = inputPrecisions[1] = Precision::FP32;
        outputPrecisions[0] = Precision::FP32;
    }

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    NodeConfig config;

    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(-1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(inputPrecisions[i], getInputShapeAtPort(i)));

        config.inConfs.push_back(portConfig);
    }

    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        PortConfig portConfig;
        portConfig.inPlace(canBeInPlace() ? 0 : -1);
        portConfig.constant(false);
        portConfig.setMemDesc(creatorsMap.at(LayoutType::ncsp)->createSharedDesc(outputPrecisions[i], getOutputShapeAtPort(i)));

        config.outConfs.push_back(portConfig);
    }

    std::vector<MemoryDescPtr> srcMemoryDescs;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        srcMemoryDescs.push_back(config.inConfs[i].getMemDesc());
    }
    std::vector<MemoryDescPtr> dstMemoryDescs;
    for (size_t i = 0; i < config.outConfs.size(); i++) {
        dstMemoryDescs.push_back(config.outConfs[i].getMemDesc());
    }

    // auto attr = initPrimitiveAttr();
    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::undef);
}

bool MatMul::created() const {
    return getType() == Type::MatMul;
}

InferenceEngine::Precision MatMul::getRuntimePrecision() const {
    return getMaxPrecision(getInputPrecisions());
}

std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>>
MatMul::getSupportedMemoryConfigs(const std::vector<InferenceEngine::Precision>& inputPrecisions,
                                  const std::vector<Shape>& inputShapes,
                                  const std::vector<InferenceEngine::Precision>& outputPrecisions,
                                  const std::vector<Shape>& outputShapes) {
    std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>> result;
    // const auto& list = ov::intel_cpu::getMVNExecutorsList();
    // const auto& list = ov::intel_cpu::getMatMulExecutorsList();
    // for (const auto& executorDesc : getMatMulImplementations()) {
    //     const auto& memoryConfig = executorDesc.builder->getMemoryConfig(inputPrecisions, inputShapes, outputPrecisions, outputShapes);
    //     for (size_t i = 0; i < memoryConfig.size(); ++i) { // @todo use vector insert
    //         result.push_back(memoryConfig[i]);
    //     }
    // }
    return result;
}

enum class EltwiseKind {
    Activation,
    ScaleShift,
    // Binary?
};

static EltwiseKind getEltwiseKind(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseSqrt:
    case Algorithm::EltwiseRelu:
    case Algorithm::EltwiseTanh:
    case Algorithm::EltwiseElu:
    case Algorithm::EltwiseAbs:
    case Algorithm::EltwiseSoftRelu:
    case Algorithm::EltwiseSigmoid:
    case Algorithm::EltwiseExp:
    case Algorithm::EltwiseGeluErf:
    case Algorithm::EltwiseGeluTanh:
    case Algorithm::EltwiseClamp:
    case Algorithm::EltwiseSwish:
    case Algorithm::EltwiseHswish:
    case Algorithm::EltwiseMish:
    case Algorithm::EltwiseHsigmoid:
    case Algorithm::EltwiseRoundHalfToEven:
    case Algorithm::EltwiseRoundHalfAwayFromZero:
        return EltwiseKind::Activation;
    case Algorithm::EltwiseAdd:
    case Algorithm::EltwiseSubtract:
    case Algorithm::EltwiseDivide:
    case Algorithm::EltwiseMultiply:
    case Algorithm::EltwiseMulAdd:
    case Algorithm::EltwisePowerStatic:
    case Algorithm::EltwisePrelu:
        return EltwiseKind::ScaleShift;
    default:
        IE_THROW() << "Unexpected eltwise algorithm: " << algToString(alg);
     }
}

static ScaleShiftPostOp::Type convertToScaleShiftOpt(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseAdd:
        return ScaleShiftPostOp::add;
    case Algorithm::EltwiseSubtract:
        return ScaleShiftPostOp::subtract;
    case Algorithm::EltwiseDivide:
        return ScaleShiftPostOp::divide;
    case Algorithm::EltwiseMultiply:
        return ScaleShiftPostOp::multiply;
    case Algorithm::EltwiseMulAdd:
        return ScaleShiftPostOp::muladd;
    case Algorithm::EltwisePowerStatic:
        return ScaleShiftPostOp::powerstatic;
    case Algorithm::EltwisePrelu:
        return ScaleShiftPostOp::prelu;
    default:
        IE_THROW() << "Unexpected eltwise algorithm: " << algToString(alg);
     }
}

static ActivationPostOp::Type convertToActivationPostOpt(const Algorithm alg) {
    switch (alg) {
    case Algorithm::EltwiseSqrt:
        return ActivationPostOp::Type::sqrt;
    case Algorithm::EltwiseRelu:
        return ActivationPostOp::Type::relu;
    case Algorithm::EltwiseTanh:
        return ActivationPostOp::Type::tanh;
    case Algorithm::EltwiseElu:
        return ActivationPostOp::Type::elu;
    case Algorithm::EltwiseAbs:
        return ActivationPostOp::Type::abs;
    case Algorithm::EltwiseSoftRelu:
        return ActivationPostOp::Type::soft_relu;
    case Algorithm::EltwiseSigmoid:
        return ActivationPostOp::Type::logistic;
    case Algorithm::EltwiseExp:
        return ActivationPostOp::Type::exp;
    case Algorithm::EltwiseGeluErf:
        return ActivationPostOp::Type::gelu_erf;
    case Algorithm::EltwiseGeluTanh:
        return ActivationPostOp::Type::gelu_tanh;
    case Algorithm::EltwiseClamp:
        return ActivationPostOp::Type::clip;
    case Algorithm::EltwiseSwish:
        return ActivationPostOp::Type::swish;
    case Algorithm::EltwiseHswish:
        return ActivationPostOp::Type::hardswish;
    case Algorithm::EltwiseMish:
        return ActivationPostOp::Type::mish;
    case Algorithm::EltwiseHsigmoid:
        return ActivationPostOp::Type::hsigmoid;
    case Algorithm::EltwiseRoundHalfToEven:
        return ActivationPostOp::Type::round_half_to_even;
    case Algorithm::EltwiseRoundHalfAwayFromZero:
        return ActivationPostOp::Type::round_half_away_from_zero;
    default:
        IE_THROW() << "Unexpected eltwise algorithm: " << algToString(alg);
     }
}

static PostOps getPostOps(std::vector<NodePtr> fused) {
    PostOps ops;

    auto makeActivationPostOp = [](const std::shared_ptr<Eltwise> eltwise) {
        return std::make_shared<ActivationPostOp>(convertToActivationPostOpt(eltwise->getAlgorithm()),
                                                  eltwise->getAlpha(),
                                                  eltwise->getBeta(),
                                                  eltwise->getGamma());
    };

    auto makeScaleShiftPostOp = [](const std::shared_ptr<Eltwise> eltwise) {
        return std::make_shared<ScaleShiftPostOp>(convertToScaleShiftOpt(eltwise->getAlgorithm()),
                                                  eltwise->getScales(),
                                                  eltwise->getShifts());
    };

    for (const auto& node : fused) {
        if (const auto eltwise = std::dynamic_pointer_cast<Eltwise>(node)) {
            const auto eltwiseKind = getEltwiseKind(eltwise->getAlgorithm());
            switch (eltwiseKind) {
            case EltwiseKind::Activation:
                ops.push_back(makeActivationPostOp(eltwise));
                break;
            case EltwiseKind::ScaleShift:
                ops.push_back(makeScaleShiftPostOp(eltwise));
                break;
            }
        }

        if (const auto fq = std::dynamic_pointer_cast<FakeQuantize>(node)) {
            ops.push_back(std::make_shared<FakeQuantizePostOp>(fq->getCropLow(),
                                                               fq->getCropHigh(),
                                                               fq->getInputScale(),
                                                               fq->getInputShift(),
                                                               fq->getOutputScale(),
                                                               fq->getOutputShift(),
                                                               fq->getLevels()));
        }
    }

    return ops;
}

ExecutorPtr MatMul::createExecutor() {
    const auto srcMemoryDescs = getSrcMemoryDescs();
    const auto dstMemoryDescs = getDstMemoryDescs();
    const auto postOps = getPostOps(fusedWith);
    const auto executor = factory->make(srcMemoryDescs, dstMemoryDescs, matmulAttrs, postOps);

    return executor;
}

void MatMul::prepareParams() {
    execPtr = createExecutor();
}

void MatMul::execute(dnnl::stream strm) {
    std::vector<MemoryCPtr> srcMemory;
    for (size_t i = 0; i < getOriginalInputsNumber(); i++) {
        srcMemory.push_back(getParentEdgeAt(i)->getMemoryPtr());
    }
    std::vector<MemoryPtr> dstMemory;
    for (size_t i = 0; i < getOriginalOutputsNumber(); i++) {
        dstMemory.push_back(getChildEdgeAt(i)->getMemoryPtr());
    }

    execPtr->exec(srcMemory, dstMemory);
}

void MatMul::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

const std::vector<impl_desc_type>& MatMul::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities = {
        impl_desc_type::unknown,
        impl_desc_type::brgemm_avx512_amx,
        impl_desc_type::brgemm_avx512,
        impl_desc_type::brgemm_avx2,
        impl_desc_type::gemm_acl,
        impl_desc_type::gemm_blas,
        impl_desc_type::gemm_avx512,
        impl_desc_type::gemm_avx2,
        impl_desc_type::gemm_avx,
        impl_desc_type::gemm_sse42,
        impl_desc_type::gemm_any,
        impl_desc_type::gemm,
        impl_desc_type::jit_gemm,
        impl_desc_type::jit_uni_dw,
        impl_desc_type::jit_uni_1x1,
        impl_desc_type::jit_uni,
        impl_desc_type::jit_avx512_dw,
        impl_desc_type::jit_avx512_1x1,
        impl_desc_type::jit_avx512,
        impl_desc_type::jit_avx2_dw,
        impl_desc_type::jit_avx2_1x1,
        impl_desc_type::jit_avx2,
        impl_desc_type::jit_avx_dw,
        impl_desc_type::jit_avx_1x1,
        impl_desc_type::jit_avx,
        impl_desc_type::jit_sse42_dw,
        impl_desc_type::jit_sse42_1x1,
        impl_desc_type::jit_sse42,
        impl_desc_type::ref,
    };

    return priorities;
}

}  // namespace node
}   // namespace intel_cpu
}   // namespace ov
