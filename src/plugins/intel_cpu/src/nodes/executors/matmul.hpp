// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "cpu_memory.h"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "matmul_key.hpp"

namespace ov {
namespace intel_cpu {

class MatMulExecutor : public Executor {
public:
    MatMulExecutor(const ExecutorContext::CPtr context,
                   const MatMulAttrs matmulAttrs);

    virtual ~MatMulExecutor() = default;

    virtual impl_desc_type getImplType() const = 0;

protected:
    const MatMulAttrs matmulAttrs;
    const ExecutorContext::CPtr context;
};

using MatMulExecutorPtr = std::shared_ptr<MatMulExecutor>;
using MatMulExecutorCPtr = std::shared_ptr<const MatMulExecutor>;

}   // namespace intel_cpu
}   // namespace ov
