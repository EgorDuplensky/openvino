// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "hash_builder.hpp"

namespace ov {
namespace intel_cpu {

struct PostOp {
    virtual ~PostOp() = default;
    virtual size_t hash(size_t seed) const = 0;
};

struct ActivationPostOp : PostOp {
    enum class Type : size_t {
        relu,
        tanh,
        elu,
        square,
        abs,
        sqrt,
        soft_relu,
        logistic,
        exp,
        gelu_erf,
        gelu_tanh,
        clip,
        swish,
        hardswish,
        mish,
        hsigmoid,
        round_half_to_even,
        round_half_away_from_zero,
        linear,
    };

    ActivationPostOp(const Type _type,
                     const float _alpha,
                     const float _beta,
                     const float _gamma)
        : type(_type),
          alpha(_alpha),
          beta(_beta),
          gamma(_gamma) {}

    size_t hash(size_t seed) const override {
        hash::Builder hb(seed);
        return hb
            .combine(type)
            .combine(alpha)
            .combine(beta)
            .combine(gamma)
            .generate();
    }

    const Type type;

// private:
    const float alpha;
    const float beta;
    const float gamma;
};

struct ScaleShiftPostOp : PostOp {
    enum Type {
        add,
        subtract,
        divide,
        multiply,
        muladd,
        powerstatic,
        prelu,
    };

    ScaleShiftPostOp(const Type _type,
                     std::vector<float> _scales,
                     std::vector<float> _shifts)
        : type(_type),
          scales(std::move(_scales)),
          shifts(std::move(_shifts)) {}

    // @todo Can we do better hashing?
    size_t hash(size_t seed) const override {
                hash::Builder hb(seed);
        return hb
            .combine(type)
            .combine(scales.size())
            .combine(scales.front())
            .combine(shifts.size())
            .combine(shifts.front())
            .generate();
    }

    Type type;

// private:
    std::vector<float> scales;
    std::vector<float> shifts;
};

struct FakeQuantizePostOp : PostOp {
    // enum Type {
    //     add,
    //     subtract,
    //     divide,
    //     multiply,
    //     muladd,
    //     powerstatic,
    //     prelu,
    // };

    FakeQuantizePostOp(std::vector<float> _cropLow,
                       std::vector<float> _cropHigh,
                       std::vector<float> _inputScale,
                       std::vector<float> _inputShift,
                       std::vector<float> _outputScale,
                       std::vector<float> _outputShift,
                       const size_t levels) :
        cropLow(std::move(_cropLow)),
        cropHigh(std::move(_cropHigh)),
        inputScale(std::move(_inputScale)),
        inputShift(std::move(_inputShift)),
        outputScale(std::move(_outputScale)),
        outputShift(std::move(_outputShift)),
        levels(levels) {}

    // Type type;
    // @todo Can we do better hashing?
    // Perhaps memdesc would be better (no data involved) but
    // memdesc would introduce additional overhead
    size_t hash(size_t seed) const override {
        hash::Builder hb(seed);
        return hb
            .combine(cropLow.size()).combine(cropLow.front())
            .combine(cropHigh.size()).combine(cropHigh.front())
            .combine(inputScale.size()).combine(inputScale.front())
            .combine(inputShift.size()).combine(inputShift.front())
            .combine(outputScale.size()).combine(outputScale.front())
            .combine(outputShift.size()).combine(outputShift.front())
            .generate();
    }

// private:
    std::vector<float> cropLow;
    std::vector<float> cropHigh;
    std::vector<float> inputScale;
    std::vector<float> inputShift;
    std::vector<float> outputScale;
    std::vector<float> outputShift;
    const size_t levels;
};

using PostOps = std::vector<std::shared_ptr<PostOp>>;

} // namespace intel_cpu
} // namespace ov
