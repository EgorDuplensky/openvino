// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "executor.hpp"
#include "graph_context.h"
#include "matmul_builder.hpp"
#include "matmul_implementation.hpp"

#include "matmul.hpp"
#include "node.h"
#include "nodes/executors/transpose_list.hpp"
#include "nodes/executor_node.hpp"
#include "graph.h"

// #include "dnnl/dnnl_matmul.hpp"

namespace ov {
namespace intel_cpu {

struct MatMulExecutorDesc {
    ExecutorType executorType;
    MatMulExecutorBuilderPtr builder;
};

std::vector<MatMulImplementation> getMatMulImplementations();

// @todo why do we need ExecutorFactory interface
// It seems we do not plan to have any storage of the factories
class MatMulExecutorFactory {
public:
    MatMulExecutorFactory(const ExecutorContext::CPtr context)
        : context(context) {}
        // _MatMulAttrs(MatMulAttrs),
        // postOps(postOps) {
        // (void)_MatMulAttrs;
        // for (auto& desc : getMatMulExecutorsList()) {
            // if (desc.builder->isSupported(MatMulAttrs, srcDescs, dstDescs)) {
            // supportedDescs.push_back(desc);
            // }
        // }
    // }

    ~MatMulExecutorFactory() = default;


    NodeConfig getNodeConfig(std::vector<MemoryDescPtr> srcDescs, std::vector<MemoryDescPtr> dstDescs) {
        NodeConfig config;

        for (size_t i = 0; i < srcDescs.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            portConfig.setMemDesc(srcDescs[i]);

            config.inConfs.push_back(portConfig);
        }

        for (size_t i = 0; i < dstDescs.size(); i++) {
            PortConfig portConfig;
            portConfig.inPlace(-1);
            portConfig.constant(false);
            portConfig.setMemDesc(dstDescs[i]);

            config.outConfs.push_back(portConfig);
        }

        return config;
    }

    NodePtr createTranspose(MemoryDescPtr desc, const ExecutorContext::CPtr context, const std::string& name) {
        TransposeParams transposeParams;
        const auto& srcDesc = desc;
        auto transposedDims = srcDesc->getShape().getDims();
        const auto n = transposedDims.size();
        std::swap(transposedDims[n - 1], transposedDims[n - 2]);
        auto transposedDesc = std::make_shared<CpuBlockedMemoryDesc>(srcDesc->getPrecision(), Shape{transposedDims});
        // const auto& transposed = key.srcDescs[1];
        transposeParams.permuteParams.src_block_dims = srcDesc->as<BlockedMemoryDesc>()->getBlockDims();
        transposeParams.permuteParams.src_block_order = srcDesc->as<BlockedMemoryDesc>()->getOrder();
        transposeParams.permuteParams.dst_block_dims = transposedDesc->as<BlockedMemoryDesc>()->getBlockDims();
        transposeParams.permuteParams.dst_block_order = transposedDesc->as<BlockedMemoryDesc>()->getOrder();
        std::vector<size_t> transposeOrder(srcDesc->getShape().getRank());
        std::iota(transposeOrder.begin(), transposeOrder.end(), 0);
        const auto m = transposeOrder.size();
        std::swap(transposeOrder[m - 1], transposeOrder[m - 2]);
        transposeParams.permuteParams.order = transposeOrder;
        transposeParams.permuteParams.data_size = srcDesc->getPrecision().size();

        std::vector<impl_desc_type> fakeImplPriority;
        // auto transpose_context = std::make_shared<ExecutorContext>(context, fakeImplPriority);
        auto factory = std::make_shared<TransposeExecutorFactory>(transposeParams,
                                                                  std::vector<MemoryDescPtr>{srcDesc},
                                                                  std::vector<MemoryDescPtr>{transposedDesc},
                                                                  context);
        dnnl::primitive_attr attr;
        auto transposeExecutor = factory->makeExecutor(transposeParams, {srcDesc}, {transposedDesc}, attr);
        auto nodeConfig = getNodeConfig({srcDesc}, {transposedDesc});
        auto transposeNode = std::make_shared<node::Executor>(transposeExecutor, nodeConfig, name, context->getGraphContext());

        return transposeNode;
    }

    void insertNode(Graph& graph, NodePtr node) {
        auto& graphNodes = graph.GetNodes();

        auto isSutableParentNode = [](const NodePtr& node) {
            return node->getType() == Type::Executor;
        };

        auto current = graphNodes.begin();
        while (current != graphNodes.end()) {
            auto currentNode = *current;
            if (!isSutableParentNode(currentNode)) {
                current++;
                continue;
            }
            // CPU_GRAPH_OPTIMIZER_SCOPE(FuseMatMulAndSimpleOperation_ParentNode);
            auto edge = currentNode->getParentEdgeAt(1);
            graph.InsertNode(edge, node, false);
        }
    }

    ExecutorPtr select(const MatMulKey& key) {
        for (const auto& impl : getMatMulImplementations()) {
            auto res = impl.doIsSupported({key, context});

            auto executor = impl.doInstantiate({key, context});

            if (res.first) {
                return executor;
            }

            auto newKey = res.second;
            // create node holding executor
            std::vector<NodePtr> graphNodes;
            std::vector<EdgePtr> graphEdges;

            auto nodeConfig = getNodeConfig(key.srcDescs, key.dstDescs);
            auto node = std::make_shared<node::Executor>(executor, nodeConfig, impl.name(), context->getGraphContext());
            std::vector<NodePtr> inputs;
            std::vector<NodePtr> outputs;
            graphNodes.push_back(node);

            for (size_t i = 0; i < key.srcDescs.size(); i++) {
                const auto inputName = node->getName() + "_in_" + std::to_string(i);
                inputs.push_back(std::make_shared<node::Input>(key.srcDescs[i]->getShape(),
                                                               key.srcDescs[i]->getPrecision(),
                                                               inputName,
                                                               "Parameter",
                                                               context->getGraphContext()));
                auto edge = std::make_shared<Edge>(inputs[i], node, 0, static_cast<int>(i));
                node->addEdge(edge);
                graphNodes.push_back(inputs[i]);
                graphEdges.push_back(edge);
            }

            for (size_t i = 0; i < key.dstDescs.size(); i++) {
                const auto outputName = node->getName() + "_out_" + std::to_string(i);
                outputs.push_back(std::make_shared<node::Input>(key.dstDescs[i]->getShape(),
                                                               key.dstDescs[i]->getPrecision(),
                                                               outputName,
                                                               "Result",
                                                               context->getGraphContext()));
                auto edge = std::make_shared<Edge>(node, outputs[i], static_cast<int>(i), 0);
                node->addEdge(edge);
                graphNodes.push_back(outputs[i]);
                graphEdges.push_back(edge);
            }

            auto graph = std::make_shared<Graph>();
            graph->CreateGraph(graphNodes, graphEdges, context->getGraphContext(), node->getName() + "_graph");

            if (key.matmulAttrs.transposeB && !newKey.matmulAttrs.transposeB) {
                auto transposeNode = createTranspose(key.srcDescs[1], context, impl.name());
                insertNode(*graph, transposeNode);

                return graph;
            }

            return impl.doInstantiate({key, context});
        }

        return nullptr;
    }

    ExecutorPtr make(const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const MatMulAttrs& matmulAttrs,
                     const PostOps& postOps) {
        /* list
         * of
         * heuristics
         * to
         * choose
         * the
         * executor
         * based on Key (src/dst descs, attributes, postops)
         */
        MatMulKey key{srcDescs, dstDescs, matmulAttrs, postOps};
        auto executor = select(key);
        // auto actualKey = executor_and_key.second;
        // if (actualKey == requestedKey) {
        //     executor = align(requestedKey, actualKey);
        // }

        if (!executor)
            IE_THROW() << "MatMulExecutorFactory: Failed to create executor";

        return executor;
    }

    // MatMulExecutorBuilderPtr builder() {
    //     if (chosenDesc) {
    //         return chosenDesc->builder;
    //     }

    //     for (const auto& sd : supportedDescs) {
    //         chosenDesc = &sd;
    //         return chosenDesc->builder;
    //     }

    //     IE_THROW() << "Supported executor is not found";
    // }

    // void setEngine(const dnnl::engine& engine) {
    //     this->engine = engine;
    // }

    // void setScratchPad(const DnnlScratchPadPtr& scratchPad) {
    //     this->scratchPad = scratchPad;
    // }

private:
    // TODO: remove dnnl dependency
    // dnnl::engine engine;

    // DnnlScratchPadPtr scratchPad = nullptr;

    // std::vector<MatMulExecutorDesc> supportedDescs;
    // const MatMulExecutorDesc* chosenDesc = nullptr;
    const ExecutorContext::CPtr context;
    // MatMulAttrs _MatMulAttrs;
    // const PostOps& postOps;
};

using MatMulExecutorFactoryPtr = std::shared_ptr<MatMulExecutorFactory>;
using MatMulExecutorFactoryCPtr = std::shared_ptr<const MatMulExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov
