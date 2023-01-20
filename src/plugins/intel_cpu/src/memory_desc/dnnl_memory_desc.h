// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_blocked_memory_desc.h"
#include <dnnl_extension_utils.h>
#include <common/memory_desc.hpp>
#include <oneapi/dnnl/dnnl.hpp>

namespace ov {
namespace intel_cpu {

class DnnlMemoryDesc;

using DnnlMemoryDescPtr = std::shared_ptr<DnnlMemoryDesc>;
using DnnlMemoryDescCPtr = std::shared_ptr<const DnnlMemoryDesc>;

class DnnlMemoryDesc : public virtual MemoryDesc {
public:
    InferenceEngine::Precision getPrecision() const override;

    MemoryDescPtr clone() const override;

    MemoryDescPtr cloneWithNewPrecision(const InferenceEngine::Precision prec) const override;

    bool isCompatible(const MemoryDesc& rhs) const override;

    bool hasLayoutType(LayoutType layoutType) const override { return false; }

    std::string serializeFormat() const override;

    size_t getMaxMemSize() const override;

    virtual bool isSame(dnnl::memory::format_tag fmt) const { return false; }

    const dnnl::memory::desc& getDnnlDesc() const {
        return desc;
    }

    dnnl::memory::data_type getDataType() const;

    dnnl::memory::format_kind getFormatKind() const;

    bool hasEmptyExtraData() const;

    void reset(dnnl_memory_desc_t md) {
        desc.reset(md);
    }

protected:
    DnnlMemoryDesc() {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    dnnl::memory::desc desc;

    void setPrecision(InferenceEngine::Precision prc) override {
        // @ TODO ONEDNN_3_0 direct access to internal elements should be avoided
        desc.get()->data_type = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(prc));
    }

private:
    explicit DnnlMemoryDesc(const dnnl::memory::desc& desc);

    size_t getElementOffset(size_t elemNumber) const override;

    bool canComputeMemSizeZeroDims() const override;
    size_t getCurrentMemSizeImp() const override;
    bool isDefinedImp() const override;
    MemoryDescPtr cloneWithNewDimsImp(const VectorDims& dims) const override;

    friend DnnlMemoryDescPtr DnnlExtensionUtils::makeDescriptor(const dnnl::memory::desc &desc);
};

}   // namespace intel_cpu
}   // namespace ov

