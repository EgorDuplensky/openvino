// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "matmul_args.hpp"
#include "nodes/executors/matmul.hpp"
#include "nodes/executors/matmul_key.hpp"

#include <functional>
#include <string>

namespace ov {
namespace intel_cpu {

class MatMulImplementation {
public:
    MatMulImplementation(const std::string& name,
                         const ExecutorType type,
                         std::function<std::pair<bool, MatMulKey>(const MatMulArgs &)> isSupported,
                         std::function<MatMulExecutorPtr(const MatMulArgs &)> instantiate)
        : m_name(name),
          m_type(type),
          isSupported(isSupported),
          instantiate(instantiate)
    {}

    std::pair<bool, MatMulKey> doIsSupported(const MatMulArgs &args) const {
        // Check supplied is_supported() function first.
        // if (isSupported && !isSupported(args)) {
        //     return false;
        // }

        // return true;
        if (isSupported) {
            return isSupported(args);
        }

        return {false, args.key};
    }

    MatMulExecutorPtr doInstantiate(const MatMulArgs &args) const {
        return instantiate(args);
    }

    const std::string& name() const {
        return m_name;
    }

private:
    const std::string& m_name;
    const ExecutorType m_type;
    std::function<std::pair<bool, MatMulKey>(const MatMulArgs &)> isSupported = {};
    std::function<MatMulExecutorPtr(const MatMulArgs &)> instantiate = {};
};

}   // namespace intel_cpu
}   // namespace ov
