// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_eltwise_up_data_movement.hpp"

#include <memory>
#include <vector>

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
} // namespace

MKLDNNPlugin::MoveEltwiseUpThroughDataMov::MoveEltwiseUpThroughDataMov() {
    auto eltwise_pattern = ngraph::pattern::wrap_type<ngraph::opset8::Add>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();

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

        ngraph::replace_output_update_name(eltwise->output(0), eltwise->input_value(0));

        auto newEltwise = eltwise->clone_with_new_inputs({current, eltwise->get_input_node_shared_ptr(1)});
        ngraph::copy_runtime_info(eltwise, newEltwise);
        newEltwise->set_friendly_name(eltwise->get_friendly_name());

        auto newChild = child->clone_with_new_inputs({newEltwise});
        ngraph::copy_runtime_info(child, newChild);
        newChild->set_friendly_name(child->get_friendly_name());

        ngraph::replace_node(child, newChild);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise_pattern, "MoveEltwiseUpThroughDataMov");
    register_matcher(m, callback);
}
