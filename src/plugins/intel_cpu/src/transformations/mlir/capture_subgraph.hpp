// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include <memory>

namespace ov {
    class Node;
}

namespace ov::intel_cpu {

class CaptureSubGraph: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::CaptureSubGraph");
    CaptureSubGraph();

private:
    // Helper functions for MLIR compatibility checking
    static bool isMLIRCompatible(const std::shared_ptr<ov::Node>& node);
    static bool hasConstantWeights(const std::shared_ptr<ov::Node>& node);
};

} // namespace ov::intel_cpu


