// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "capture_subgraph.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_cpu {

CaptureSubGraph::CaptureSubGraph() {
    MATCHER_SCOPE(CaptureSubGraph);
    // Match MatMul operation with at least one constant input (weight)
    auto X_m = ov::pass::pattern::any_input();
    auto Y_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(); // Second input should be constant (weight)
    auto matmul_m = ov::pass::pattern::wrap_type<ov::op::v0::MatMul>({X_m, Y_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& candidate_out = pattern_map.at(matmul_m);
        const auto& candidate = candidate_out.get_node_shared_ptr();

        if (candidate->get_type_info() == SubModel::get_type_info_static()) {
            return false;  // avoid recursive wrapping submodel into submodel
        }
        
        // Validate MLIR compatibility
        if (!isMLIRCompatible(candidate)) {
            return false;
        }
        
        // Ensure we have at least one constant weight
        if (!hasConstantWeights(candidate)) {
            return false;
        }

        ov::ParameterVector params;
        ov::ResultVector results;
        OutputVector args;

        ov::OutputVector candidate_inputs;
        for (size_t i = 0; i < candidate->inputs().size(); i++) {
            const auto& input = candidate->get_input_node_shared_ptr(i);
            if (op::util::is_on_constant_path(input->output(0))) {
                candidate_inputs.emplace_back(candidate->get_input_source_output(i));
            } else {
                params.emplace_back(std::make_shared<ov::op::v0::Parameter>(candidate->get_element_type(),
                                                                            candidate->get_input_partial_shape(i)));
                candidate_inputs.push_back(params.back()->output(0));
            }
        }

        auto candidate_clone = candidate->clone_with_new_inputs(candidate_inputs);
        candidate_clone->set_friendly_name(candidate->get_friendly_name() + "_clone");

        for (const auto& output : candidate_clone->outputs()) {
            results.emplace_back(std::make_shared<ov::op::v0::Result>(output));
        }

        auto submodel =
            std::make_shared<ov::Model>(results, params, candidate_clone->get_friendly_name() + "_subgraph");
        auto subgraph = std::make_shared<ov::intel_cpu::SubModel>(submodel);
        subgraph->set_friendly_name("Submodel_" + candidate_clone->get_friendly_name());

        for (size_t i = 0; i < params.size(); i++) {
            subgraph->set_invariant_input(params[i], candidate->input_value(i));
        }

        ov::copy_runtime_info(subgraph, candidate);
        ov::replace_node(candidate, subgraph);

        std::cout << "Captured MLIR-compatible MatMul subgraph: " << candidate->get_friendly_name() << std::endl;
        
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, matcher_name);
    this->register_matcher(m, callback);
}

bool CaptureSubGraph::isMLIRCompatible(const std::shared_ptr<ov::Node>& node) {
    // Check if the operation is supported by MLIR
    if (!ov::is_type<ov::op::v0::MatMul>(node)) {
        return false;
    }
    
    auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node);
    
    // Check for supported data types (for now, f32 and f16)
    auto input_type = matmul->get_input_element_type(0);
    if (input_type != ov::element::f32 && input_type != ov::element::f16) {
        return false;
    }
    
    // Check for static shapes (dynamic shapes not supported yet)
    for (size_t i = 0; i < matmul->get_input_size(); i++) {
        if (matmul->get_input_partial_shape(i).is_dynamic()) {
            return false;
        }
    }
    
    // Check tensor rank (2D matrices for simplicity)
    auto input0_shape = matmul->get_input_partial_shape(0);
    auto input1_shape = matmul->get_input_partial_shape(1);
    
    if (input0_shape.rank().get_length() != 2 || input1_shape.rank().get_length() != 2) {
        return false;
    }
    
    return true;
}

bool CaptureSubGraph::hasConstantWeights(const std::shared_ptr<ov::Node>& node) {
    // Check if at least one input is a constant (weight matrix)
    for (size_t i = 0; i < node->get_input_size(); i++) {
        auto input_node = node->get_input_node_shared_ptr(i);
        if (ov::is_type<ov::op::v0::Constant>(input_node)) {
            // Additional check: ensure the constant is reasonably sized
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
            auto shape = constant->get_shape();
            
            // Avoid very large constants that might not benefit from MLIR
            size_t total_elements = 1;
            for (auto dim : shape) {
                total_elements *= dim;
            }
            
            // Skip if weight matrix is too small (< 64 elements) or too large (> 1M elements)
            if (total_elements >= 64 && total_elements <= 1048576) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace ov::intel_cpu
