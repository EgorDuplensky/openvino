// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpu/x64/cpu_isa_traits.hpp>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include "nodes/eltwise.h"
#include "nodes/executors/matmul_key.hpp"
#include "nodes/executors/matmul_args.hpp"
#include "nodes/executors/matmul.hpp"
#include "dnnl.hpp"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "node.h"

namespace ov {
namespace intel_cpu {

class DnnlMatMulExecutor : public MatMulExecutor, public Node {
public:
    DnnlMatMulExecutor(const MatMulArgs &args);

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    void setPostOpArgs(const std::unordered_map<int, MemoryPtr>& postOpsArgs) {
        this->postOpsArgs = postOpsArgs;
    }

private:
    void fuse(const MatMulKey& key, dnnl::primitive_attr& attr, ExecutorContext::CPtr _context);
    // static std::pair<Shape, Shape> makeDummyInputShapes(const MatMulAttrs& matmulAttrs, const Shape& in0, const Shape& in1);
    dnnl::stream stream;
    // MatMulAttrs matmulAttrs;
    std::shared_ptr<dnnl::matmul> prim;
    MemoryPtr scratchpadMemory;
    std::unordered_map<int, MemoryPtr> postOpsArgs;
    impl_desc_type implType = impl_desc_type::undef;
};

using DnnlMatMulExecutorPtr = std::shared_ptr<DnnlMatMulExecutor>;

}   // namespace intel_cpu
}   // namespace ov
