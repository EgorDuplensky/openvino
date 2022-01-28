// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/core_integration.hpp"
#include <openvino/runtime/properties.hpp>
#include "ie_system_conf.h"
#include "openvino/runtime/core.hpp"
#include "openvino/core/type/element_type.hpp"

using namespace ov::test::behavior;
using namespace InferenceEngine::PluginConfigParams;

namespace {
//
// IE Class Common tests with <pluginName, deviceName params>
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassCommon, OVClassBasicTestP,
        ::testing::Values(std::make_pair("openvino_intel_cpu_plugin", "CPU")));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassNetworkTestP, OVClassNetworkTestP,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassImportExportTestP, OVClassImportExportTestP,
        ::testing::Values("HETERO:CPU"));

//
// IE Class GetMetric
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_CONFIG_KEYS,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_SUPPORTED_METRICS,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_AVAILABLE_DEVICES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_FULL_DEVICE_NAME,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_OPTIMIZATION_CAPABILITIES,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_ASYNC_INFER_REQUESTS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_RANGE_FOR_STREAMS,
        ::testing::Values("CPU"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetMetricTest, OVClassGetMetricTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest_ThrowUnsupported,
        ::testing::Values("CPU", "MULTI", "HETERO", "AUTO"));

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetAvailableDevices, OVClassGetAvailableDevices,
        ::testing::Values("CPU"));

//
// IE Class GetConfig
//

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassGetConfigTest, OVClassGetConfigTest,
        ::testing::Values("CPU"));

//////////////////////////////////////////////////////////////////////////////////////////

TEST(OVClassBasicTest, smoke_SetConfigInferenceNumThreads) {
    ov::Core ie;
    int32_t value = 0;
    int32_t num_threads = 1;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);

    num_threads = 4;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::inference_num_threads(num_threads)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::inference_num_threads));
    ASSERT_EQ(num_threads, value);
}

TEST(OVClassBasicTest, smoke_SetConfigStreamsNum) {
    ov::Core ie;
    int32_t value = 0;
    const int32_t num_streams = 1;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::streams::num(num_streams)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::streams::num));
    ASSERT_EQ(num_streams, value);
}

TEST(OVClassBasicTest, smoke_SetConfigAffinity) {
    ov::Core ie;
    ov::Affinity value = ov::Affinity::NONE;

#if (defined(__APPLE__) || defined(_WIN32))
    auto numaNodes = InferenceEngine::getAvailableNUMANodes();
    auto defaultBindThreadParameter = numaNodes.size() > 1 ? ov::Affinity::NUMA : ov::Affinity::NONE;
#else
    auto defaultBindThreadParameter = ov::Affinity::CORE;
#endif
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(defaultBindThreadParameter, value);

    const ov::Affinity affinity = ov::Affinity::HYBRID_AWARE;
    ASSERT_NO_THROW(ie.set_property("CPU", ov::affinity(affinity)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::affinity));
    ASSERT_EQ(affinity, value);
}

TEST(OVClassBasicTest, smoke_SetConfigHintInferencePrecision) {
    ov::Core ie;
    auto value = ov::element::f32;
    const auto precision = InferenceEngine::with_cpu_x86_bfloat16() ? ov::element::bf16 : ov::element::f32;

    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(precision, value);

    const auto forcedPrecision = ov::element::f32;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::hint::inference_precision(forcedPrecision)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::hint::inference_precision));
    ASSERT_EQ(precision, forcedPrecision);
}

TEST(OVClassBasicTest, smoke_SetConfigEnableProfiling) {
    ov::Core ie;
    bool value;
    const bool enableProfilingDefault = false;

    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfilingDefault, value);

    const bool enableProfiling = true;

    ASSERT_NO_THROW(ie.set_property("CPU", ov::enable_profiling(enableProfiling)));
    ASSERT_NO_THROW(value = ie.get_property("CPU", ov::enable_profiling));
    ASSERT_EQ(enableProfiling, value);
}

// IE Class Query network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassQueryNetworkTest, OVClassQueryNetworkTest,
        ::testing::Values("CPU"));

// IE Class Load network

INSTANTIATE_TEST_SUITE_P(
        smoke_OVClassLoadNetworkTest, OVClassLoadNetworkTest,
        ::testing::Values("CPU"));
} // namespace

