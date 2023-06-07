// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_utils.hpp"
#include "acl_matmul.hpp"
#include "acl_utils.hpp"
#include "ie_common.h"
#include "nodes/executors/matmul_args.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

// AclMatMulExecutor::AclMatMulExecutor(const ExecutorContext::CPtr context) : MatMulExecutor(context) {}
// static validate()

AclMatMulExecutor::AclMatMulExecutor(const MatMulArgs &args) :
    MatMulExecutor(args.context, args.key.matmulAttrs) {
    auto srcDims = args.key.srcDescs[0]->getShape().getStaticDims();
    auto weiDims = args.key.srcDescs[1]->getShape().getStaticDims();
    auto dstDims = args.key.dstDescs[0]->getShape().getStaticDims();

    auto srcBatch = vectorProduct(srcDims, srcDims.size() - 2);
    auto weiBatch = vectorProduct(weiDims, weiDims.size() - 2);
    auto dstBatch = vectorProduct(dstDims, dstDims.size() - 2);
    auto M = srcDims[srcDims.size() - 2];
    auto K = srcDims[srcDims.size() - 1];
    auto N = weiDims[weiDims.size() - 1];

    // ACL doesn't support cases then both inputs are broadcasted
    // if ((srcBatch > 1 && weiBatch > 1 && srcBatch != weiBatch) ||
    //     (srcBatch != dstBatch && weiBatch != dstBatch)) {
    //     return false;
    // }

    TensorInfo srcTensorInfo = TensorInfo(TensorShape(K, M, 1, srcBatch), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo weiTensorInfo = TensorInfo(TensorShape(N, K, weiBatch), 1, DataType::F32, DataLayout::NCHW);
    TensorInfo dstTensorInfo = TensorInfo(TensorShape(N, M, 1, dstBatch), 1, DataType::F32, DataLayout::NCHW);

    const auto status = arm_compute::NEGEMM::validate(&srcTensorInfo, &weiTensorInfo, nullptr, &dstTensorInfo, 1.0f, 0.0f);
    if (!status)
        IE_THROW() << "AclMatMulExecutor: validate failed with message - " << status.error_description();

    srcTensor.allocator()->init(srcTensorInfo);
    weiTensor.allocator()->init(weiTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    matmul = std::make_unique<arm_compute::NEGEMM>();
    matmul->configure(&srcTensor, &weiTensor, nullptr, &dstTensor, 1.0f, 0.0f);
}

void AclMatMulExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
    srcTensor.allocator()->import_memory(src[0]->getData());
    weiTensor.allocator()->import_memory(src[1]->getData());
    dstTensor.allocator()->import_memory(dst[0]->getData());

    matmul->run();

    srcTensor.allocator()->free();
    weiTensor.allocator()->free();
    dstTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov
