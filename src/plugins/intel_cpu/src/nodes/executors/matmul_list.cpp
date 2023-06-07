// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "matmul_builder.hpp"
#include "matmul_implementation.hpp"
#include "matmul_factory.hpp"
#include "matmul_args.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/dnnl/dnnl_matmul.hpp"
#include "nodes/executors/matmul_key.hpp"
#if defined(OV_CPU_WITH_ACL)
#include "acl/acl_matmul.hpp"
#endif

namespace ov {
namespace intel_cpu {

static std::vector<MatMulImplementation> matmulImplementations = {
    OV_CPU_INSTANCE_X64(
        "matmul_dnnl",                                                       // name
        ExecutorType::Dnnl,                                                  // type
        [](const MatMulArgs &args) { return true; },                         // is supported
        [](const MatMulArgs &args) { return std::make_shared<DnnlMatMulExecutor>(args); }, // instantiate
        )
    OV_CPU_INSTANCE_ACL(
        "matmul",                                                            // name
        ExecutorType::Acl,                                                   // type
        // [](const MatMulArgs &args) { return !(args.key.matmulAttrs.transposeA || args.key.matmulAttrs.transposeB); }, // is supported
        [](const MatMulArgs &args) -> std::pair<bool, MatMulKey>{
            if (args.key.matmulAttrs.transposeB) {
                auto newAttrs = args.key.matmulAttrs;
                newAttrs.transposeB = false;
                return {false, {args.key.srcDescs, args.key.dstDescs, newAttrs, args.key.postOps}};
            }

            return {true, args.key};
        }, // is supported
        [](const MatMulArgs &args) { return std::make_shared<AclMatMulExecutor>(args); }, // instantiate
        )
};

std::vector<MatMulImplementation> getMatMulImplementations() {
    return matmulImplementations;
}

}   // namespace intel_cpu
}   // namespace ov
