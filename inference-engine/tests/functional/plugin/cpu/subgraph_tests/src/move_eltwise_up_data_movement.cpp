// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/pass/visualize_tree.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_transformations/move_eltwise_up_data_movement.hpp"

using namespace testing;
using namespace ngraph;

namespace SubgraphTestsDefinitions {

class MoveEltwiseUpThroughDataMovFixture : public ::testing::TestWithParam<int64_t> {
};

TEST(MoveEltwiseUpThroughDataMovFixture, SingleUnaryEltwise) {
    const Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto transpose = std::make_shared<opset8::Transpose>(input, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        auto sigmoid = std::make_shared<opset8::Sigmoid>(unsqueeze);

        f = std::make_shared<Function>(NodeVector{sigmoid}, ParameterVector{input});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    std::shared_ptr<Function> f_ref(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto sigmoid = std::make_shared<opset8::Sigmoid>(input);

        auto transpose = std::make_shared<opset8::Transpose>(sigmoid, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        f_ref = std::make_shared<Function>(NodeVector{unsqueeze}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

/* replace_output_update_name doesn't remove previous output shared node*/
TEST(DISABLED_MoveEltwiseUpThroughDataMovFixture, EltwiseSequence) {
    const Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;
    std::shared_ptr<Function> f(nullptr);
    {
        auto input_left = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto input_right = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto matmul = std::make_shared<opset8::MatMul>(input_left, input_right);

        auto transpose = std::make_shared<opset8::Transpose>(matmul, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto relu = std::make_shared<opset8::Relu>(transpose);

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(relu, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        auto sigmoid = std::make_shared<opset8::Sigmoid>(unsqueeze);

        f = std::make_shared<Function>(NodeVector{sigmoid}, ParameterVector{input_left, input_right});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    std::shared_ptr<Function> f_ref(nullptr);
    {
        auto input_left = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto input_right = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto matmul = std::make_shared<opset8::MatMul>(input_left, input_right);

        auto relu = std::make_shared<opset8::Relu>(matmul);

        auto sigmoid = std::make_shared<opset8::Sigmoid>(relu);

        auto transpose = std::make_shared<opset8::Transpose>(sigmoid, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        f_ref = std::make_shared<Function>(NodeVector{unsqueeze}, ParameterVector{input_left, input_right});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMovFixture, DataMovementTwoConsumers) {
    /* In this case transformation shouldn't apply */
    auto create_graph = [] () -> std::shared_ptr<ngraph::Function> {
        const Shape shape{1, 3, 224, 224};
        const std::vector<int64_t> input_order = {1, 2, 0, 3};
        const int64_t unsqueeze_axis = 1;

        auto input_left = std::make_shared<opset8::Parameter>(element::f32, shape);
        auto input_right = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto matmul = std::make_shared<opset8::MatMul>(input_left, input_right);

        auto transpose = std::make_shared<opset8::Transpose>(matmul, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        auto sigmoid = std::make_shared<opset8::Sigmoid>(unsqueeze);

        auto relu = std::make_shared<opset8::Relu>(transpose);

        return std::make_shared<Function>(NodeVector{sigmoid, relu}, ParameterVector{input_left, input_right});
    };

    std::shared_ptr<Function> f = create_graph();

    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<Function> f_ref = create_graph();

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMovFixture, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5;
    std::shared_ptr<Function> f(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto transpose = std::make_shared<opset8::Transpose>(input, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        auto add = std::make_shared<opset8::Add>(unsqueeze, opset8::Constant::create(element::f32, {}, {scalar_value}));

        f = std::make_shared<Function>(NodeVector{add}, ParameterVector{input});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    std::shared_ptr<Function> f_ref(nullptr);
    {
        auto input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto add = std::make_shared<opset8::Add>(input, opset8::Constant::create(element::f32, {}, {scalar_value}));

        auto transpose = std::make_shared<opset8::Transpose>(add, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        f_ref = std::make_shared<Function>(NodeVector{unsqueeze}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

TEST(MoveEltwiseUpThroughDataMovFixture, SingleBinaryEltwiseWithNotScalarOnSecondBranch) {
    auto create_graph = [] () -> std::shared_ptr<ngraph::Function> {
        const Shape shape{1, 3, 224, 224};
        const std::vector<int64_t> input_order = {3, 2, 1, 0};
        const int64_t unsqueeze_axis = 2;
        std::shared_ptr<Function> f(nullptr);
        auto input = std::make_shared<opset8::Parameter>(element::f32, shape);

        auto transpose = std::make_shared<opset8::Transpose>(input, opset8::Constant::create(element::i64, Shape{input_order.size()}, input_order));

        auto unsqueeze = std::make_shared<opset8::Unsqueeze>(transpose, opset8::Constant::create(element::i64, Shape{}, {unsqueeze_axis}));

        auto add_scalar = opset8::Constant::create(element::f32, {1, 1, 1, 3}, {0.5, 0.2, 0.3});
        auto add = std::make_shared<opset8::Add>(unsqueeze, add_scalar);

        return std::make_shared<Function>(NodeVector{add}, ParameterVector{input});
    };
    std::shared_ptr<ngraph::Function> f = create_graph();
    pass::Manager m;
    m.register_pass<pass::InitNodeInfo>();
    m.register_pass<MKLDNNPlugin::MoveEltwiseUpThroughDataMov>();
    m.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));

    std::shared_ptr<ngraph::Function> f_ref = create_graph();
    auto res = compare_functions(f, f_ref);

    ASSERT_TRUE(res.first) << res.second;
}

} // namespace SubgraphTestsDefinitions
