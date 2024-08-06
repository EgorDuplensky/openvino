// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <openvino/op/constant.hpp>

namespace ov {
namespace intel_cpu {
namespace node {

class Input : public Node {
public:
    struct InputConfig {
        MemoryDescPtr desc;
        bool inPlace;
    };

    struct OutputConfig {
        // @todo better to use memory desc with any layout and undefined precision
        bool useParentMemoryDescForOutput;
        bool inPlace;
    };

    Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    Input(const Shape& shape,
          const ov::element::Type& prc,
          const std::string& name,
          const std::string& type,
          const GraphContext::CPtr context);

    Input(MemoryDescPtr memDesc, const std::string& name, const std::string& type, const GraphContext::CPtr context);

    Input(const std::shared_ptr<ov::Node>& op,
          const GraphContext::CPtr context,
          InputConfig config);

    Input(const std::shared_ptr<ov::Node>& op,
          const GraphContext::CPtr context,
          OutputConfig config);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void initOptimalPrimitiveDescriptor() override;
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    bool created() const override;
    void withMeanImage();

    void setMemoryPtr(MemoryCPtr memory) {
        memoryPtr = memory;
    }

    MemoryCPtr getMemoryPtr() const;

    void execute(dnnl::stream strm) override {}
    void executeDynamicImpl(dnnl::stream strm) override {}
    bool isExecutable() const override {
        return false;
    }

    bool needShapeInfer() const override { return false; }
    bool needPrepareParams() const override { return false; }

private:
    void cloneBlobIfRequired();
    void initSupportedPdDefault();
    void initSupportedPdFromMemDesc();

private:
    std::shared_ptr<ov::op::v0::Constant> m_constOp;
    MemoryCPtr memoryPtr;
    bool isMeanImage = false;
    MemoryDescPtr extMemDesc = nullptr;
    bool m_useParentMemoryDescForOutput = false;
    bool m_isInPlace = false;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
