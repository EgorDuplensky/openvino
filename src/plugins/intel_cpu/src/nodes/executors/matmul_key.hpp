// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "post_ops.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "memory_desc/dnnl_memory_desc.h"
#include <dnnl_extension_utils.h>
#include <common/primitive_hashing_utils.hpp>

namespace ov {
namespace intel_cpu {

struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
    bool withBias;
};

struct MatMulKey {
    // @todo replace with src and dst vectors?
    std::vector<MemoryDescPtr> srcDescs;
    std::vector<MemoryDescPtr> dstDescs;
    MatMulAttrs matmulAttrs;
    PostOps postOps;

    MatMulKey(const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const MatMulAttrs& matmulAttrs,
              const PostOps& postOps)
        : srcDescs(srcDescs),
          dstDescs(dstDescs),
          matmulAttrs(matmulAttrs),
          postOps(postOps) {}

    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash::combine(seed, matmulAttrs.transposeA);
        seed = hash::combine(seed, matmulAttrs.transposeB);
        for (const auto& ptr : srcDescs) {
            if (ptr) {
                // TODO cpu memory desc hash
                seed = hash::combine(seed, get_md_hash(*MemoryDescUtils::convertToDnnlMemoryDesc(ptr)->getDnnlDesc().get()));
            }
        }

        for (const auto& ptr : dstDescs) {
            if (ptr) {
                // TODO cpu memory desc hash
                seed = hash::combine(seed, get_md_hash(*MemoryDescUtils::convertToDnnlMemoryDesc(ptr)->getDnnlDesc().get()));
            }
        }

        for (const auto& op : postOps) {
            seed = op->hash(seed);
        }

        return seed;
    }

    bool operator==(const MatMulKey& rhs) const {
        bool retVal = true;
        retVal = retVal && matmulAttrs.transposeA == rhs.matmulAttrs.transposeA;
        retVal = retVal && matmulAttrs.transposeB == rhs.matmulAttrs.transposeB;

        for (size_t i = 0; i < srcDescs.size(); i++) {
            const auto& desc = srcDescs[i];
            const auto& rhs_desc = rhs.srcDescs[i];
            if (desc != rhs_desc) {
                retVal = retVal && desc && rhs_desc && desc->isCompatible(*rhs_desc);
            }
        }

        for (size_t i = 0; i < dstDescs.size(); i++) {
            const auto& desc = dstDescs[i];
            const auto& rhs_desc = rhs.dstDescs[i];
            if (desc != rhs_desc) {
                retVal = retVal && desc && rhs_desc && desc->isCompatible(*rhs_desc);
            }
        }

        retVal = retVal && postOps == rhs.postOps;
        return retVal;
    }
};

}  // namespace intel_cpu
}  // namespace ov
