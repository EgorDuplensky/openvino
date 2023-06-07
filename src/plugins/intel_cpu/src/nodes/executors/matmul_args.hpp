// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matmul_key.hpp"
#include "executor.hpp"

#pragma once

namespace ov {
namespace intel_cpu {

struct MatMulArgs {
    MatMulKey key;
    ExecutorContext::CPtr context;
};

}  // namespace intel_cpu
}  // namespace ov
