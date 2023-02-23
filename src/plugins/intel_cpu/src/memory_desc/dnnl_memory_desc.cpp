// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_memory_desc.h"
#include <dnnl_extension_utils.h>
#include <common/memory_desc.hpp>
#include <common/memory_desc_wrapper.hpp>
#include <onednn/dnnl.h>

namespace ov {
namespace intel_cpu {

DnnlMemoryDesc::DnnlMemoryDesc(const dnnl::memory::desc& desc) :
    MemoryDesc(Shape(DnnlExtensionUtils::convertToVectorDims(desc.get_dims())), Dnnl), desc(desc) {
    if (getFormatKind() == dnnl::memory::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
}

InferenceEngine::Precision DnnlMemoryDesc::getPrecision() const {
    return DnnlExtensionUtils::DataTypeToIEPrecision(getDataType());
}

MemoryDescPtr DnnlMemoryDesc::clone() const {
    return std::make_shared<DnnlMemoryDesc>(*this);
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewPrecision(const InferenceEngine::Precision prec) const {
    auto newDesc = std::make_shared<DnnlMemoryDesc>(*this);
    newDesc->setPrecision(prec);
    return newDesc;
}

bool DnnlMemoryDesc::isCompatible(const MemoryDesc &rhs) const {
    if (MemoryDescType::Dnnl == rhs.getType()) {
        return this->desc == rhs.as<DnnlMemoryDesc>()->desc;
    } else {
        return false;
    }
}

std::string DnnlMemoryDesc::serializeFormat() const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    if (wrapped.is_wino_desc()) {
        switch (desc.get()->format_desc.wino_desc.wino_format) {
            case dnnl::impl::wino_memory_format_t::wino_wei_aaOio: return "wino_aaOio";
            case dnnl::impl::wino_memory_format_t::wino_wei_aaOBiOo: return "wino_aaOBiOo";
            case dnnl::impl::wino_memory_format_t::wino_wei_OBaaIBOIio: return "wino_OBaaIBOIio";
            default: return "wino_undef";
        }
    } else if (wrapped.is_rnn_packed_desc()) {
        switch (desc.get()->format_desc.rnn_packed_desc.format) {
            case dnnl::impl::rnn_packed_format::ldigo_p: return "packed_ldigo";
            case dnnl::impl::rnn_packed_format::ldgoi_p: return "packed_ldgoi";
            case dnnl::impl::rnn_packed_format::ldio_p: return "packed_ldio";
            default: return "packed_undef";
        }
    }
    return "undef";
}

size_t DnnlMemoryDesc::getMaxMemSize() const {
    if (shape.isDynamic()) {
        IE_THROW() << "Can't compute max mem size for DnnlMemoryDesc with dynaimc shape";
    }

    return getCurrentMemSize();
}

dnnl::memory::data_type DnnlMemoryDesc::getDataType() const {
    return desc.get_data_type();
}

dnnl::memory::format_kind DnnlMemoryDesc::getFormatKind() const {
    return desc.get_format_kind();
}

bool DnnlMemoryDesc::hasEmptyExtraData() const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return wrapped.extra().flags == dnnl_memory_extra_flag_none;
}

bool DnnlMemoryDesc::canComputeMemSizeZeroDims() const {
    if (!getShape().hasZeroDims())
        return false;

    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return getShape().hasZeroDims() && wrapped.offset0() != DNNL_RUNTIME_DIM_VAL;
}

size_t DnnlMemoryDesc::getCurrentMemSizeImp() const {
    return DnnlExtensionUtils::getMemSizeForDnnlDesc(desc);
}

size_t DnnlMemoryDesc::getElementOffset(size_t elemNumber) const {
    dnnl::impl::memory_desc_wrapper wrapped(desc.get());
    return wrapped.off_l(elemNumber);
}

bool DnnlMemoryDesc::isDefinedImp() const {
    dnnl::impl::memory_desc_wrapper wrappedThis(desc.get());

    if (wrappedThis.has_runtime_dims_or_strides()) {
        return false;
    }

    return wrappedThis.offset0() != DNNL_RUNTIME_DIM_VAL;
}

MemoryDescPtr DnnlMemoryDesc::cloneWithNewDimsImp(const VectorDims &dims) const {
    IE_THROW(Unexpected) << "Cannot clone non blocked oneDNN desc with new dims";
}

}   // namespace intel_cpu
}   // namespace ov
