// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <type_traits>
namespace ov {
namespace intel_cpu {
namespace hash {

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
template <typename T, typename std::enable_if<!std::is_enum<T>::value , int>::type = 0>
static size_t combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T, typename std::enable_if<std::is_enum<T>::value , int>::type = 0>
static size_t combine(size_t seed, const T &v) {
    using underlying_t = typename std::underlying_type<T>::type;
    return combine(seed, static_cast<underlying_t>(v));
}

struct Builder {
    Builder(size_t seed)
        : _seed(seed) {}

    // todo add specializations / sfinae
    template <typename T>
    Builder& combine(T v) {
        _seed = ov::intel_cpu::hash::combine(_seed, v);
        return *this;
    }

    size_t generate() {
        return _seed;
    }
private:
    size_t _seed;
};

} // namespace hash
} // namespace intel_cpu
} // namespace ov
