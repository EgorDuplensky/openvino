// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     UnsqueezetMatMul transformation detects MatMul operations
 *     and broadcasts one input to another to align the ranks of the inputs.
 *     The transformation is required because oneDNN library
 *     requires inputs to have equal ranks
 */

namespace MKLDNNPlugin {

class UnsqueezeMatMul: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    UnsqueezeMatMul();
};

}  // namespace MKLDNNPlugin
