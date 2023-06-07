// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include "graph.h"

namespace ov {
namespace intel_cpu {

namespace node {
class MatMul;
namespace {
struct MatMulKey;
}
}

class Graph;

using MatMulGraphAligner = std::function<void(const node::MatMul&, const node::MatMulKey&, const node::MatMulKey&)>;

static void alignMatMul(const node::MatMul& mm, Graph::Ptr graph, const node::MatMulKey& requiredKey, const node::MatMulKey& actualKey) {}

class MatMulGraphAlignerClass {
public:
    MatMulGraphAlignerClass(Graph::Ptr graph)
        : graph(graph) {}

    void operator() (const node::MatMul& mm, const node::MatMulKey& requiredKey, const node::MatMulKey& actualKey) {
        alignMatMul(mm, graph, requiredKey, actualKey);
    }
private:
    Graph::Ptr graph;
};


} // namespace intel_cpu
} // namespace ov
