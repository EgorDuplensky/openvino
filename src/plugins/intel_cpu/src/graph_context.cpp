// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "graph_context.h"
#include <memory>
#include <vector>
#include "cache/multi_cache.h"
#include "dnnl_scratch_pad.h"
#include "nodes/memory.hpp"
#include "memory_control.hpp"

namespace ov {
namespace intel_cpu {

static std::shared_ptr<std::vector<DnnlScratchPadPtr>> createScratchPads(int numNumaNodes) {
    std::shared_ptr<std::vector<DnnlScratchPadPtr>> rtScratchPads = std::make_shared<std::vector<DnnlScratchPadPtr>>();
    for (int i = 0; i < numNumaNodes; i++) {
        rtScratchPads->push_back(std::make_shared<DnnlScratchPad>(GraphContext::getEngine(), i));
    }

    return rtScratchPads;
}

static std::shared_ptr<std::vector<MultiCachePtr>> createParamCaches(int numNumaNodes, size_t cacheCapacity) {
    std::shared_ptr<std::vector<MultiCachePtr>> paramCaches = std::make_shared<std::vector<MultiCachePtr>>();
    for (int i = 0; i < numNumaNodes; i++) {
        paramCaches->push_back(std::make_shared<MultiCache>(cacheCapacity));
    }

    return paramCaches;
}

static std::shared_ptr<std::vector<std::shared_ptr<node::MemoryStatesRegister>>> createMemoryStateRegister(
    int numNumaNodes) {
    std::shared_ptr<std::vector<std::shared_ptr<node::MemoryStatesRegister>>> memoryStateRegisters =
        std::make_shared<std::vector<std::shared_ptr<node::MemoryStatesRegister>>>();

    // for (int i = 0; i < numNumaNodes; i++) {
    //     memoryStateRegisters->push_back(std::make_shared<node::MemoryStatesRegister>());
    // }

    auto msr = std::make_shared<node::MemoryStatesRegister>();
    for (int i = 0; i < numNumaNodes; i++) {
        memoryStateRegisters->push_back(msr);
    }

    return memoryStateRegisters;
}

GraphGlobalContext::GraphGlobalContext(const Config& config,
                                       std::shared_ptr<SocketsWeights> w_cache,
                                       bool isGraphQuantized,
                                       int numNumaNodes,
                                       std::shared_ptr<ov::threading::CPUStreamsExecutor> streamExecutor,
                                       std::shared_ptr<SubMemoryManager> sub_memory_manager,
                                       std::shared_ptr<std::vector<std::shared_ptr<node::MemoryStatesRegister>>> memoryStatesRegister,
                                       std::shared_ptr<NetworkMemoryControl> networkMemoryControl,
                                       std::shared_ptr<std::vector<MultiCachePtr>> rtParamsCaches,
                                       std::shared_ptr<std::vector<DnnlScratchPadPtr>> rtScratchPads)
    : config(config),
      weightsCache(std::move(w_cache)),
      isGraphQuantizedFlag(isGraphQuantized),
      streamExecutor(streamExecutor),
      subMemoryManager(sub_memory_manager),
      // memoryStatesRegister(memoryStatesRegister ? memoryStatesRegister
      //                      : std::make_shared<std::vector<std::shared_ptr<node::MemoryStatesRegister>>>(
      //                          numNumaNodes,
      //                          std::make_shared<node::MemoryStatesRegister>())),
      memoryStatesRegister(memoryStatesRegister ? memoryStatesRegister
                           : createMemoryStateRegister(numNumaNodes)),
      networkMemoryControl(networkMemoryControl ? networkMemoryControl : std::make_shared<NetworkMemoryControl>()),
      // rtParamsCaches(std::make_shared<std::vector<MultiCachePtr>>(
      //                    numNumaNodes,
      //                    std::make_shared<MultiCache>(config.rtCacheCapacity))),
      rtParamsCaches(rtParamsCaches && !std::getenv("RTC_PER_GRAPH") ? rtParamsCaches
                     : createParamCaches(numNumaNodes, config.rtCacheCapacity)),
      rtScratchPads(rtScratchPads && !std::getenv("SP_PER_GRAPH") ? rtScratchPads : createScratchPads(numNumaNodes))
{}

GraphContext::GraphContext(const Config& config,
                           std::shared_ptr<SocketsWeights> w_cache,
                           bool isGraphQuantized,
                           std::shared_ptr<ov::threading::CPUStreamsExecutor> streamExecutor,
                           std::vector<std::shared_ptr<ov::threading::CPUStreamsExecutor>> subStreamExecutors,
                           std::shared_ptr<SubMemoryManager> sub_memory_manager,
                           int level,
                           int numaId)
    : local{level, numaId, subStreamExecutors},
      global(std::make_shared<GraphGlobalContext>(config,
                                                  w_cache,
                                                  isGraphQuantized,
                                                  std::max(1, static_cast<int>(subStreamExecutors.size())),
                                                  streamExecutor,
                                                  sub_memory_manager)) {}

const dnnl::engine& GraphContext::getEngine() {
    static const dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    return eng;
}

}   // namespace intel_cpu
}   // namespace ov
