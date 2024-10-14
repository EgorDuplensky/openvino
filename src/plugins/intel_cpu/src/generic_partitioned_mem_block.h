// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {

/**
 * This is a memory block that represents a view on a subblock inside another continuous dynamic memory block
 *
 */
class GenericPartitionedMemoryBlock : public IMemoryBlockObserver {
public:
    GenericPartitionedMemoryBlock(MemoryBlockPtr pBlock,
                                  VectorDims chunks,
                                  VectorDims offsets,
                                  VectorDims strides,
                                  ov::element::Type type)
        : m_pBlock(pBlock),
          m_chunks(chunks),
          m_offsets(offsets),
          m_strides(strides),
          m_type(type) {
        OPENVINO_ASSERT(m_pBlock, "Memory block is uninitialized");
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;
    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

private:
    MemoryBlockPtr m_pBlock;
    VectorDims m_chunks;
    VectorDims m_offsets;
    VectorDims m_strides;
    ov::element::Type m_type;
    size_t m_size = 0; // size of the viewed partition in bytes
    size_t m_memoryOffset;
};

}  // namespace intel_cpu
}  // namespace ov
