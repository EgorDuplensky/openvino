// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_eltwise_up_data_movement.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::MoveEltwiseUpThroughDataMov, "MoveEltwiseUpThroughDataMov", 0);

namespace {
    bool is_data_movement_operation(const std::shared_ptr<ngraph::Node>& node) {
        return std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v0::ShuffleChannels>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v7::Roll>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v0::ReverseSequence>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v0::DepthToSpace>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v1::BatchToSpace>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v1::Broadcast>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v3::Broadcast>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v1::Gather>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v7::Gather>(node) != nullptr ||
               std::dynamic_pointer_cast<ngraph::op::v8::Gather>(node) != nullptr;
    }

    bool is_scalar_like(const std::shared_ptr<ngraph::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
        auto shape = constantNode->get_shape();
        if (constantNode == nullptr || shape_size(shape) > 1) {
            return false;
        }
        return true;
    }
} // namespace

MKLDNNPlugin::MoveEltwiseUpThroughDataMov::MoveEltwiseUpThroughDataMov() {
    auto eltwise_pattern = ngraph::pattern::wrap_type<ngraph::op::util::UnaryElementwiseArithmetic, ngraph::op::util::BinaryElementwiseArithmetic>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        bool is_binary_op = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(eltwise) != nullptr;
        if (is_binary_op &&
            (std::dynamic_pointer_cast<ngraph::opset8::Constant>(eltwise->get_input_node_shared_ptr(1)) == nullptr ||
            !is_scalar_like(eltwise->get_input_node_shared_ptr(1)))) {
            return false;
        }

        auto current = eltwise->get_input_node_shared_ptr(0);
        auto child = eltwise;

        while (true) {
            if (current->get_output_size() != 1) {
                break;
            }

            if (!is_data_movement_operation(current)) {
                break;
            }
            child = current;
            current = current->get_input_node_shared_ptr(0);
        }

        if (child == eltwise) {
            return false;
        }

        if (is_binary_op) {
            auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(eltwise->get_input_node_shared_ptr(1));
            auto scalarConstantNode = ngraph::opset8::Constant::create(constantNode->get_element_type(), {1}, constantNode->get_data_ptr());
            ngraph::replace_node(constantNode, scalarConstantNode);
        }

        ngraph::replace_output_update_name(eltwise->output(0), eltwise->input_value(0));

        ngraph::OutputVector eltwiseInputs{child->input_value(0)};

        if (is_binary_op) {
            eltwiseInputs.emplace_back(eltwise->input_value(1));
        }

        auto newEltwise = eltwise->clone_with_new_inputs(eltwiseInputs);
        ngraph::copy_runtime_info(eltwise, newEltwise);
        newEltwise->set_friendly_name(eltwise->get_friendly_name());

        ngraph::OutputVector childInputs{newEltwise};
        for (size_t index = 1; index < child->get_input_size(); ++index) {
            childInputs.emplace_back(child->get_input_node_shared_ptr(index));
        }

        auto newChild = child->clone_with_new_inputs(childInputs);

        ngraph::copy_runtime_info(child, newChild);
        newChild->set_friendly_name(child->get_friendly_name());

        ngraph::replace_node(child, newChild);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise_pattern, "MoveEltwiseUpThroughDataMov");
    register_matcher(m, callback);
}
