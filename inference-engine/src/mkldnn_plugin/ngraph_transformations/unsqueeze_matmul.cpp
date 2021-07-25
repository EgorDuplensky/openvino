// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unsqueeze_matmul.hpp"

#include "ngraph/op/matmul.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pattern/op/or.hpp>

#include <algorithm>

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::UnsqueezeMatMul, "UnsqueezeMatMul", 0);

MKLDNNPlugin::UnsqueezeMatMul::UnsqueezeMatMul() {
    ngraph::OutputVector twoInputs = {
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
        ngraph::pattern::any_input(ngraph::pattern::has_static_shape())
    };

    auto matmulPattern = ngraph::pattern::wrap_type<ngraph::op::MatMul>(twoInputs, ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher& m) {
        auto matmul = std::dynamic_pointer_cast<ngraph::op::MatMul> (m.get_match_root());

        if (!matmul || transformation_callback(matmul))
            return false;

        const auto& input0 = matmul->input_value(0);
        const auto& input1 = matmul->input_value(1);
        const auto& input0shape = input0.get_shape();
        const auto& input1shape = input1.get_shape();

        if (input0shape.size() == input1shape.size())
            return false;

        auto getUnsqueeze = [](const ngraph::Output<ngraph::Node>& nodeFrom, const ngraph::Output<ngraph::Node>& nodeTo) {
            auto rankFrom = nodeFrom.get_partial_shape().rank().get_length();
            auto rankTo = nodeTo.get_partial_shape().rank().get_length();

            std::vector<int64_t> unsqueeze_axes;
            for (int64_t j = 0; j < rankTo - rankFrom; ++j)
                unsqueeze_axes.push_back(j);

            auto unsqueeze = std::make_shared<ngraph::opset1::Unsqueeze>(
                nodeFrom,
                ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, unsqueeze_axes));

            unsqueeze->set_friendly_name(nodeFrom.get_node()->get_friendly_name() + "/Unsqueeze");

            return unsqueeze;
        };

        std::cout << "#################### Unsqueezing one of Matmul inputs ###################" << "\n";

        auto matmul_new_inputs = matmul->input_values();
        ngraph::NodeVector new_ops;

        if (input0shape.size() < input1shape.size()) {
            std::shared_ptr<ngraph::Node> unsqueezeInput0 = getUnsqueeze(input0, input1);
            matmul_new_inputs[0] = unsqueezeInput0;
            new_ops.push_back(unsqueezeInput0);
        } else if (input0shape.size() > input1shape.size()) {
            std::shared_ptr<ngraph::Node> unsqueezeInput1 = getUnsqueeze(input1, input0);
            matmul_new_inputs[1] = unsqueezeInput1;
            new_ops.push_back(unsqueezeInput1);
        }

        std::shared_ptr<ngraph::Node> matmul_new = matmul->clone_with_new_inputs(matmul_new_inputs);

        new_ops.push_back(matmul_new);
        matmul_new->set_friendly_name(matmul->get_friendly_name());
        ngraph::copy_runtime_info(matmul, new_ops);
        ngraph::replace_node(matmul, matmul_new);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matmulPattern, "UnsqueezeMatMul");
    this->register_matcher(m, callback);
}
