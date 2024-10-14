// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_partitioned_mem_block.h"
#include <cstddef>
#include <functional>
#include <numeric>

using namespace ov::intel_cpu;

void* GenericPartitionedMemoryBlock::getRawPtr() const noexcept {
    // std::cout << "total chunks: " << m_total_chunks << "\n";
    // std::cout << "offset chunks: " << m_offset_chunks << "\n";
    // std::cout << "num chunks: " << m_size_chunks << "\n";
    // std::cout << "m_size: " << m_size << "\n";
    // std::cout << "getting partitioned memory with offset: " << m_offset_chunks * m_size / m_size_chunks << "\n";
    const auto offset = m_offsets.back() * m_strides[m_strides.size() - 2] * m_type.size() / m_chunks.back();
    // std::cout << "GenericPartitionedMemoryBlock: memory: " << m_pBlock->getRawPtr() << " offset: " << offset << "\n";

    return static_cast<uint8_t*>(m_pBlock->getRawPtr()) + offset;
}

void GenericPartitionedMemoryBlock::setExtBuff(void* ptr, size_t size) {
    m_pBlock->setExtBuff(ptr, size);
}

bool GenericPartitionedMemoryBlock::resize(size_t size) {
    m_size = size;
    // auto parentBlockSize = std::accumulate(m_chunks.begin(), m_chunks.end(), Dim{1}, std::multiplies<Dim>()) * m_size;

    // std::cout << "Resizing pmb with size: " << size << " parent size:" << parentBlockSize << "\n";

    // return m_pBlock->resize(parentBlockSize);
    return m_pBlock->resize_safe(m_size);
    // return m_pBlock->resize_safe(parentBlockSize);
}

bool GenericPartitionedMemoryBlock::hasExtBuffer() const noexcept {
    return m_pBlock->hasExtBuffer();
}

void GenericPartitionedMemoryBlock::registerMemory(Memory* memPtr) {
    m_pBlock->registerMemory(memPtr);
}

void GenericPartitionedMemoryBlock::unregisterMemory(Memory* memPtr) {
    m_pBlock->unregisterMemory(memPtr);
}
