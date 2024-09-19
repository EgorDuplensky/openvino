// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_desc/dnnl_memory_desc.h"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include <memory>
#include <transformations/utils/utils.hpp>
#include <unordered_map>
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/fully_connected.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/placeholder.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"

#include "split_fc.hpp"

#include "itt.hpp"

static size_t weightsThreshold() {
    static int result = std::getenv("SPLIT_THRESHOLD") ? std::stoi(std::getenv("SPLIT_THRESHOLD")) : 6600000;
    return static_cast<size_t>(result);
}

ov::intel_cpu::SplitFC::SplitFC(int sub_stream_num) {
    MATCHER_SCOPE(SplitFC);

    auto m_fc_c = ov::pass::pattern::wrap_type<ov::op::internal::FullyConnectedCompressed>({
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
    });

    auto fc_m = ov::pass::pattern::wrap_type<ov::op::internal::FullyConnected>({
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
        ov::pass::pattern::any_input(),
    });

    auto m_fc_or = std::make_shared<pass::pattern::op::Or>(
        OutputVector{
            fc_m,
            m_fc_c
        });

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto fc_node = pattern_map.count(fc_m) ? pattern_map.at(fc_m).get_node_shared_ptr()
            : pattern_map.at(m_fc_c).get_node_shared_ptr();

        auto& rt_info = fc_node->get_rt_info();
        if (rt_info.count("split_part")) {
            return false;
        }

        const auto src_item = fc_node->get_input_node_shared_ptr(0);
        const auto fc_weight_node = fc_node->get_input_node_shared_ptr(1);

        // split happens on the first dimension.
        constexpr size_t split_dim = 0;
        auto split_dim_node = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, split_dim);

        // needn't to split fc when the dim is 0.
        const auto& wgt_shape = fc_weight_node->get_shape();
        // weight shape size 660000 is a trade-off value, which is summarized and verified by LLMs.
        if (wgt_shape[split_dim] <= 1 || ov::shape_size(wgt_shape) < weightsThreshold()) {
            return false;
        }

        // parts will be splited according the sub stream num.
        int split_num = sub_stream_num;

        // auto split_on_parts = [](int len, int n) {
        //     int average = len / n;
        //     std::vector<int> parts(n, average);
        //     parts.back() = len - average * (n - 1);
        //     return parts;
        // };

        // TODO: support transpose
        if (ov::is_type<ov::op::v1::Transpose>(fc_weight_node)) {
            return false;
        }

        std::cout << "Splitting operation: " << fc_node->get_friendly_name() << "\n";

        std::vector<std::shared_ptr<ov::Node>> wgt_node_vec(split_num);
        auto wgt_item = fc_node->get_input_node_shared_ptr(1);

        auto slice_piece_of_input = [](std::shared_ptr<ov::Node> weights, int split_dim, int split_num, int idx) -> std::shared_ptr<ov::Node> {
            // int64_t offset = N * (i + 1);
            // std::cout << weights->get_shape() << "\n";
            size_t N = weights->get_shape()[split_dim];
            if (N < 2)
                return weights;
            int N_piece = N / split_num;
            int K = weights->get_shape()[split_dim + 1];

            std::vector<int64_t> begin{N_piece * idx, 0}; // Start from {0, 0}
            std::vector<int64_t> end{N_piece * (idx + 1), K};
            // end.back() = offset;
            std::vector<int64_t> strides(2, 1); // Stride of 1 for both dimensions
            // Optionally, you can define masks if necessary
            std::vector<int64_t> begin_mask(2, 0); // Include all elements starting from 'begin'
            // begin_mask.back() = 0;
            std::vector<int64_t> end_mask(2, 0);   // Include all elements up to 'end'
            // end_mask.back() = 0;

            // Create the StridedSlice node
            return std::make_shared<ov::op::v1::StridedSlice>(
                weights,
                ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin),
                ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end),
                ov::op::v0::Constant::create(ov::element::i64, {strides.size()}, strides),
                begin_mask,
                end_mask);
        };

        for (int i = 0; i < split_num; i++) {
            wgt_node_vec[i] = slice_piece_of_input(wgt_item, split_dim, split_num, i);
            wgt_node_vec[i]->set_friendly_name("StridedSliceInplaceSplitW_" + std::to_string(i));

            auto disable_consant_folding = [](Node* node) {
                ov::disable_constant_folding(node->shared_from_this());
            };

            std::unordered_set<Node *> visited;
            ov::op::util::visit_constant_path(wgt_node_vec[i].get(), visited, disable_consant_folding);
        }

        std::vector<ov::Output<ov::Node>> bias_node_vec(split_num);
        if (fc_node->get_input_size() >= 3) {
            auto bias_item = fc_node->get_input_node_shared_ptr(2);
            for (int i = 0; i < split_num; i++) {
                if (as_type_ptr<ov::op::internal::Placeholder>(bias_item)) {
                    bias_node_vec[i] = std::make_shared<ov::op::internal::Placeholder>();
                } else {
                    bias_node_vec[i] = slice_piece_of_input(bias_item, split_dim, split_num, i);
                    bias_node_vec[i].get_node_shared_ptr()->set_friendly_name("StridedSliceInplaceSplitB_" + std::to_string(i));
                }
            }
        }

        std::vector<ov::Output<ov::Node>> decompression_multiply_node_vec(split_num);
        if (fc_node->get_input_size() >= 4) {
            auto multiply_item = fc_node->get_input_node_shared_ptr(3);
            for (int i = 0; i < split_num; i++) {
                if (as_type_ptr<ov::op::internal::Placeholder>(multiply_item)) {
                    decompression_multiply_node_vec[i] = std::make_shared<ov::op::internal::Placeholder>();
                } else {
                    decompression_multiply_node_vec[i] = slice_piece_of_input(multiply_item, split_dim, split_num, i);
                    decompression_multiply_node_vec[i].get_node_shared_ptr()->set_friendly_name(
                        "StridedSliceInplaceSplitWS_" + std::to_string(i));
                }
            }
        }

        std::vector<ov::Output<ov::Node>> decompression_subtract_node_vec(split_num);
        if (fc_node->get_input_size() >= 5) {
            auto subtract_item = fc_node->get_input_node_shared_ptr(4);
            for (int i = 0; i < split_num; i++) {
                if (as_type_ptr<ov::op::internal::Placeholder>(subtract_item)) {
                    decompression_subtract_node_vec[i] = std::make_shared<ov::op::internal::Placeholder>();
                } else {
                    decompression_subtract_node_vec[i] = slice_piece_of_input(subtract_item, split_dim, split_num, i);
                    decompression_subtract_node_vec[i].get_node_shared_ptr()->set_friendly_name(
                        "StridedSliceInplaceSplitWZ_" + std::to_string(i));
                }

                auto disable_consant_folding = [](Node* node) {
                    ov::disable_constant_folding(node->shared_from_this());
                };

                std::unordered_set<Node*> visited;
                ov::op::util::visit_constant_path(decompression_subtract_node_vec[i].get_node(), visited, disable_consant_folding);
            }
        }

        // create fc Nodes according to the splited weight or splited pattern.
        std::vector<std::shared_ptr<Node>> fc_node_vec(split_num);
        for (int i = 0; i < split_num; ++i) {
            if (fc_node->get_input_size() == 3)
                fc_node_vec[i] =
                    fc_node->clone_with_new_inputs(ov::OutputVector{src_item, wgt_node_vec[i], bias_node_vec[i]});
            else if (fc_node->get_input_size() == 4)
                fc_node_vec[i] = fc_node->clone_with_new_inputs(
                    ov::OutputVector{src_item, wgt_node_vec[i], bias_node_vec[i], decompression_multiply_node_vec[i]});
            else if (fc_node->get_input_size() == 5)
                fc_node_vec[i] = fc_node->clone_with_new_inputs(ov::OutputVector{src_item,
                                                                                 wgt_node_vec[i],
                                                                                 bias_node_vec[i],
                                                                                 decompression_multiply_node_vec[i],
                                                                                 decompression_subtract_node_vec[i]});
            fc_node_vec[i]->set_friendly_name(fc_node->get_friendly_name() + "_split_" + std::to_string(i));
            // mark every split node as "split_part"
            fc_node_vec[i]->get_rt_info()["split_part"] = true;
            fc_node_vec[i]->get_rt_info()["piece_idx"] = i;
            fc_node_vec[i]->get_rt_info()["num_pieces"] = split_num;

            if (i > 0)
                // mark every non-first split node as "other_split"
                fc_node_vec[i]->get_rt_info()["other_split"] = true;
        }

        // mark first split node as a "main_split_root"
        fc_node_vec[0]->get_rt_info()["main_split_root"] = true;
        std::vector<std::shared_ptr<ov::Node>> split_parts;
        split_parts.reserve(fc_node_vec.size());
        for (const auto& fc : fc_node_vec) {
            split_parts.push_back(fc);
        }
        fc_node_vec[0]->get_rt_info()["split_parts"] = split_parts;

        // concat all small fc for result.
        ov::NodeVector concat_args = fc_node_vec;
        // concat happens on the latest dimension.
        constexpr size_t concat_dim = -1;
        auto concat_node = std::make_shared<ov::op::v0::Concat>(concat_args, concat_dim);
        concat_node->get_rt_info()["sync_point"] = true;

        // check the shape after transformation.
        const auto& out_shape = fc_node->get_output_partial_shape(0);
        const auto& concat_shape = concat_node->get_output_partial_shape(0);
        if (concat_shape != out_shape) {
            return false;
        }
        ov::replace_node_update_name(fc_node, concat_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_fc_or, matcher_name);
    this->register_matcher(m, callback);
}
