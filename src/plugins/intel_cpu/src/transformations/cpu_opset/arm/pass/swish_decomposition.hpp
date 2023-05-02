// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class SwishDecomposition: public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SwishDecomposition", "0");
    SwishDecomposition();
};

}   // namespace intel_cpu
}   // namespace ov
