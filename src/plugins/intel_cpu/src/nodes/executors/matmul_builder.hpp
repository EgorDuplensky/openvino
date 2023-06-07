// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "ie_precision.hpp"
#include "node.h"
#include "onednn/iml_type_mapper.h"
#include "executor.hpp"
#include "nodes/executors/matmul_key.hpp"
#include "nodes/executors/matmul.hpp"

namespace ov {
namespace intel_cpu {

class MatMulExecutorBuilder {
public:
    virtual ~MatMulExecutorBuilder() = default;
    virtual bool isSupported(const MatMulKey& key) const = 0;
    virtual std::vector<std::pair<const std::vector<MemoryDescPtr>, const std::vector<MemoryDescPtr>>>
    getMemoryConfig(const std::vector<InferenceEngine::Precision>& inputPrecisions,
                    const std::vector<Shape>& inputShapes,
                    const std::vector<InferenceEngine::Precision>& outputPrecisions,
                    const std::vector<Shape>& outputShapes) const = 0;
    // virtual MatMulExecutorPtr emit() const = 0;
    virtual MatMulExecutorPtr instantiate(const MatMulKey& key, ExecutorContext::CPtr context) = 0;
    // virtual void fuse(const PostOps& postOps) = 0;
};

using MatMulExecutorBuilderPtr = std::shared_ptr<MatMulExecutorBuilder>;
using MatMulExecutorBuilderCPtr = std::shared_ptr<const MatMulExecutorBuilder>;

}   // namespace intel_cpu
}   // namespace ov
