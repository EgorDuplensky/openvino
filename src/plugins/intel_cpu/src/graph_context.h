// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "sub_memory_manager.hpp"
#include "cache/multi_cache.h"
#include "config.h"
#include "dnnl_scratch_pad.h"
#include "weights_cache.hpp"

namespace ov {
namespace intel_cpu {

namespace node {
class MemoryStatesRegister;
} // namespace node

class NetworkMemoryControl;

class GraphGlobalContext {
public:
    using Ptr  = std::shared_ptr<GraphGlobalContext>;
    using CPtr = std::shared_ptr<const GraphGlobalContext>;

    GraphGlobalContext(const Config& config,
                       std::shared_ptr<SocketsWeights> w_cache,
                       bool isGraphQuantized,
                       int numNumaNodes = 1,
                       std::shared_ptr<ov::threading::CPUStreamsExecutor> streamExecutor = nullptr,
                       std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr,
                       std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister = nullptr,
                       std::shared_ptr<NetworkMemoryControl> networkMemoryControl = nullptr,
                       std::shared_ptr<std::vector<MultiCachePtr>> rtParamsCaches = nullptr,
                       std::shared_ptr<std::vector<DnnlScratchPadPtr>> rtScratchPads = nullptr);

    GraphGlobalContext::CPtr useSubStreamExecutor(std::shared_ptr<ov::threading::CPUStreamsExecutor> streamExecutor) const {
        return std::make_shared<const GraphGlobalContext>(config,
                                                          weightsCache,
                                                          isGraphQuantizedFlag,
                                                          rtParamsCaches->size(),
                                                          streamExecutor,
                                                          subMemoryManager,
                                                          memoryStatesRegister,
                                                          networkMemoryControl,
                                                          rtParamsCaches,
                                                          rtScratchPads);
    }

    Config config;  // network-level config

    std::shared_ptr<SocketsWeights> weightsCache;         // per NUMA node caches for sharing weights data

    bool isGraphQuantizedFlag = false;

    ov::threading::CPUStreamsExecutor::Ptr streamExecutor;   // cpu stream executor for current graph

    std::shared_ptr<SubMemoryManager> subMemoryManager;
    // MultiCachePtr rtParamsCache;     // primitive cache
    // DnnlScratchPadPtr rtScratchPad;  // scratch pad
    std::shared_ptr<node::MemoryStatesRegister> memoryStatesRegister;
    std::shared_ptr<NetworkMemoryControl> networkMemoryControl;
    std::shared_ptr<std::vector<MultiCachePtr>> rtParamsCaches;     // primitive cache
    std::shared_ptr<std::vector<DnnlScratchPadPtr>> rtScratchPads;  // scratch pad (each sub-stream has its own copy)
};

// context which is specific to the current graph
struct GraphLocalContext {
    int level;
    int numaId;
    std::vector<std::shared_ptr<ov::threading::CPUStreamsExecutor>> subStreamExecutors;   // stream executor for current graph
};

class GraphContext {
public:
    typedef std::shared_ptr<GraphContext> Ptr;
    typedef std::shared_ptr<const GraphContext> CPtr;

    GraphContext(const Config& config,
                 std::shared_ptr<SocketsWeights> w_cache,
                 bool isGraphQuantized,
                 std::shared_ptr<ov::threading::CPUStreamsExecutor> streamExecutor = nullptr,
                 std::vector<std::shared_ptr<ov::threading::CPUStreamsExecutor>> subStreamExecutors = {},
                 std::shared_ptr<SubMemoryManager> sub_memory_manager = nullptr,
                 int level = -1,
                 int numaId = 0);

    GraphContext(GraphLocalContext local, GraphGlobalContext::CPtr global) : local(local), global(global) {}

    static const dnnl::engine& getEngine();

    const Config& getConfig() const {
        return global->config;
    }

    WeightsSharing::Ptr getWeightsCache() const {
        return (*global->weightsCache)[local.numaId];
    }

    MultiCachePtr getParamsCache() const {
        // std::cout << "Using runtime cache from numaId: " << local.numaId << "\n";
        return global->rtParamsCaches->at(local.numaId);
    }

    DnnlScratchPadPtr getScratchPad(int subStreamID = 0) const {
        // if (subStreamID < 0)
        //     subStreamID = 0;
        // if (subStreamID >= global->numNumaNodes - 1)
        //     subStreamID = global->numNumaNodes - 1;
        // return global->rtScratchPads[subStreamID];
        return global->rtScratchPads->at(local.numaId);
    }

    const std::vector<DnnlScratchPadPtr>& getScratchPads() const {
        return *global->rtScratchPads;
    }

    bool isGraphQuantized() const {
        return global->isGraphQuantizedFlag;
    }

    const std::vector<std::shared_ptr<ov::threading::CPUStreamsExecutor>>& getCPUStreamExecutors() const {
        return local.subStreamExecutors;
    }

    ov::threading::CPUStreamsExecutor::Ptr getCPUStreamExecutor() const {
        return global->streamExecutor;
    }

    std::shared_ptr<SubMemoryManager> getSubMemory() const {
        return global->subMemoryManager;
    }

    int getNumNumaNodes() const {
        return std::max(1, get_num_numa_nodes());
    }

    const std::shared_ptr<node::MemoryStatesRegister>& getMemoryStatesRegister() const {
        return global->memoryStatesRegister;
    }

    const std::shared_ptr<NetworkMemoryControl>& getNetworkMemoryControl() const {
        return global->networkMemoryControl;
    }

    int level() const {
        return local.level;
    }

    // go one level deeper into the context
    GraphContext::Ptr down() const {
        return std::make_shared<GraphContext>(GraphLocalContext{local.level + 1, local.numaId, local.subStreamExecutors}, global);
    }

    GraphContext::Ptr moveToNuma(int numaId) const {
        return std::make_shared<GraphContext>(GraphLocalContext{local.level, numaId, {}},
                                              global->useSubStreamExecutor(local.subStreamExecutors[numaId]));
    }

private:
    GraphLocalContext local;
    GraphGlobalContext::CPtr global;
};

}  // namespace intel_cpu
}  // namespace ov
