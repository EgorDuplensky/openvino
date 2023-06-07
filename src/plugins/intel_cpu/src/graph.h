// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "config.h"
#include "cpu_memory.h"
#include "nodes/executors/executor.hpp"
#include "normalize_preprocess.h"
#include "node.h"
#include "nodes/input.h"
#include "edge.h"
#include "cache/multi_cache.h"
#include "dnnl_scratch_pad.h"
#include "graph_context.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

#include "proxy_mem_mgr.h"

namespace ov {
namespace intel_cpu {

class InferRequestBase;
class InferRequest;

class Graph : public Executor {
public:
    typedef std::shared_ptr<Graph> Ptr;

    enum class Status {
        NotReady = 0,
        ReadyStatic = 1,
        ReadyDynamic = 2
    };

    Graph() = default;

    Graph(std::string name,
          const GraphContext::CPtr ctx)
        : reuse_io_tensors(false),
          _name(name),
          context(ctx) {}

    ~Graph();

    bool IsReady() {
        return (status != Status::NotReady);
    }

    const Config & getConfig() const {
        return context->getConfig();
    }

    template<typename NET>
    void CreateGraph(NET &network, const GraphContext::CPtr ctx);

    void CreateGraph(const std::vector<NodePtr> &graphNodes,
                     const std::vector<EdgePtr> &graphEdges,
                     const GraphContext::CPtr ctx,
                     std::string name);


    void CreateGraphNoInit(const GraphContext::CPtr ctx,
                           std::string name) {
        std::cout << "Creating graph: " << name << "\n";

        context = ctx;

        this->_name = std::move(name);
        this->reuse_io_tensors = false;
    }

    bool hasMeanImageFor(const std::string& name) {
        return _normalizePreprocMap.find(name) != _normalizePreprocMap.end();
    }

    void PushInputData(const std::string& name, const InferenceEngine::Blob::Ptr &in);
    void PullOutputData(InferenceEngine::BlobMap &out);

    void Infer(InferRequestBase* request = nullptr);

    static std::shared_ptr<Graph> createGraph(const NodePtr& node) {
        const auto graphName = node->getName() + "_graph";
        const auto& context = node->getContext();
        auto _graph = std::make_shared<Graph>(graphName, context);

        //Make inputs
        for (size_t i = 0, port = 0; i < node->getParentEdges().size(); i++) {
            if (node->getParentEdgeAt(i)->getParent()->isConstant())
                continue;

            const auto inputName = node->getName() + "_in_" + std::to_string(port);
            auto input = std::make_shared<node::Input>(node->getInputShapeAtPort(port),
                                                       node->getOriginalInputPrecisionAtPort(port),
                                                       inputName,
                                                       "Parameter",
                                                       context);
            _graph->AddEdge(input, node, port, port);
            _graph->AddNode(input);
            _graph->GetInputNodesMap()[input->getName()] = input;
            _graph->inputs.push_back(input);
            port++;
        }

        _graph->AddNode(node);

        for (size_t i = 0, port = 0; i < node->getChildEdges().size(); i++) {
            if (node->getParentEdgeAt(i)->getParent()->isConstant())
                continue;

            const auto outputName = node->getName() + "_out_" + std::to_string(port);
            auto output = std::make_shared<node::Input>(node->getOutputShapeAtPort(port),
                                                        node->getOriginalOutputPrecisionAtPort(port),
                                                        outputName,
                                                        "Result",
                                                        context);

            _graph->AddEdge(output, node, port, port);
            _graph->AddNode(output);
            _graph->GetOutputNodesMap()[output->getName()] = output;
            _graph->outputs.push_back(output);
            port++;
        }

        return _graph;
    }

    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override {
        assert(src.size() == inputs.size());
        assert(dst.size() == outputs.size());

        for (size_t i = 0; i != src.size(); i++)
            inputs[i]->getChildEdgeAt(0)->getMemoryPtr()->getMemoryMngr()->setExtBuff(src[i]->getData(), src[i]->getSize());

        for (size_t i = 0; i != dst.size(); i++)
            outputs[i]->getParentEdgeAt(0)->getMemoryPtr()->getMemoryMngr()->setExtBuff(dst[i]->getData(), src[i]->getSize());

        Infer();
    }

    void InferShapes() {
        for (auto& node : executableGraphNodes) {
            if (node->isDynamicNode()) {
                node->updateShapes();
            }
        }
    }

    const std::vector<NodePtr>& GetNodes() const {
        return graphNodes;
    }

    std::vector<NodePtr>& GetNodes() {
        return graphNodes;
    }

    std::string GetName() const {
        return _name;
    }

    std::vector<EdgePtr>& GetEdges() {
        return graphEdges;
    }

    std::map<std::string, NodePtr>& GetInputNodesMap() {
        return inputNodesMap;
    }

    std::map<std::string, NodePtr>& GetOutputNodesMap() {
        return outputNodesMap;
    }

    NodePtr getInputNodeByName(const std::string &name) {
        auto input = inputNodesMap.find(name);
        if (input == inputNodesMap.end())
            IE_THROW() << "CPU execution graph doesn't contain input node with name: " << name;
        return input->second;
    }

    NodePtr getOutputNodeByName(const std::string &name) {
        auto output = outputNodesMap.find(name);
        if (output == outputNodesMap.end())
            IE_THROW() << "CPU execution graph doesn't contain output node with name: " << name;
        return output->second;
    }

    bool hasOutputWithName(const std::string& name) const {
        return outputNodesMap.count(name);
    }

    dnnl::engine getEngine() const {
        return context->getEngine();
    }

    GraphContext::CPtr getGraphContext() const {
        return context;
    }

    void GetPerfData(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> &perfMap) const;

    void RemoveDroppedNodes();
    void RemoveDroppedEdges();
    void RemoveEdge(EdgePtr& edge);
    void AddEdge(const NodePtr& parent,
                 const NodePtr& child,
                 int parentPort = 0,
                 int childPort = 0) {
        auto edge = std::make_shared<Edge>(parent, child, parentPort, childPort);
        parent->childEdges.push_back(edge);
        child->parentEdges.push_back(edge);
        graphEdges.push_back(edge);
    }

    void AddNode(NodePtr node) {
        assert(std::find(graphNodes.begin(), graphNodes.end(), node) == graphNodes.end());
        graphNodes.push_back(node);
    }

    void AddEdge(EdgePtr edge) {
        assert(std::find(graphEdges.begin(), graphEdges.end(), edge) == graphEdges.end());
        graphEdges.push_back(edge);
    }

    void DropNodeRelaxed(const NodePtr &node) {
        auto children = node->childEdges;
        auto parents = node->parentEdges;

        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (!p_edge) continue;
            auto parent = p_edge->getParent();
            if (!parent) continue;
            if (parent->isConstant()) continue;

            const int inNum = p_edge->getInputNum();
            p_edge->drop();
            RemoveEdge(p_edge);

            for (size_t j = 0; j < children.size(); j++) {
                auto c_edge = children[j].lock();
                if (!c_edge) continue;
                auto child = c_edge->getChild();
                if (!child) continue;

                const int outNum = c_edge->getOutputNum();
                c_edge->drop();
                RemoveEdge(c_edge);

                std::cout << "Adding edge between: " << parent->getName() << " and: " << child->getName() << "\n";

                EdgePtr newEdge(new Edge(parent, child, inNum, outNum));
                graphEdges.push_back(newEdge);
                parent->addEdge(newEdge);
            }
        }
    }

    void ReplaceNode(const NodePtr &origin, const NodePtr &node) {
        std::cout << "Replacing node: " << origin->getName() << " with: " << node->getName() << "\n";

        auto children = origin->childEdges;
        auto parents = origin->parentEdges;

        for (size_t i = 0; i < parents.size(); i++) {
            auto p_edge = parents[i].lock();
            if (!p_edge) continue;
            auto parent = p_edge->getParent();
            if (!parent) continue;
            // if (parent->isConstant()) continue;

            const int inNum  = p_edge->getInputNum();
            const int outNum = p_edge->getOutputNum();
            // p_edge->drop();
            RemoveEdge(p_edge);
            std::cout << "Creatin edge between: " << parent->getName() << " and " << node->getName() << "\n";
            EdgePtr newEdge(new Edge(parent, node, inNum, outNum));
            parent->addEdge(newEdge);
            graphEdges.push_back(newEdge);
        }

        for (size_t j = 0; j < children.size(); j++) {
            auto c_edge = children[j].lock();
            if (!c_edge) continue;
            auto child = c_edge->getChild();
            if (!child) continue;

            const int inNum  = c_edge->getInputNum();
            const int outNum = c_edge->getOutputNum();
            RemoveEdge(c_edge);

            std::cout << "Creatin edge between: " << child->getName() << " and " << node->getName() << "\n";
            EdgePtr newEdge(new Edge(node, child, inNum, outNum));
            child->addEdge(newEdge);
            graphEdges.push_back(newEdge);
        }

        auto pos = std::find(std::begin(graphNodes), std::end(graphNodes), origin);
        if (pos != std::end(graphNodes)) {
            *pos = node;
        }
    }

    void RemoveEdgeNoDrop(const EdgePtr& edge) {
        auto pos = std::find(std::begin(graphEdges), std::end(graphEdges), edge);

        if (pos != std::end(graphEdges)) {
            std::cout << "Removing node: " << (*pos)->name() << "\n";
            graphEdges.erase(pos);
        }
    }

    void RemoveNode(const NodePtr& node) {
        // auto pos = std::remove_if(std::begin(graphNodes), std::end(graphNodes), [&name](const NodePtr& node) {
        //     return name == node->getName();
        // });
        auto pos = std::find(std::begin(graphNodes), std::end(graphNodes), node);

        if (pos != std::end(graphNodes)) {
            std::cout << "Removing node: " << (*pos)->getName() << "\n";
            graphNodes.erase(pos);
        }
    }

    void SetInputs(std::vector<MemoryDescPtr> memDescs) {
        for (size_t i = 0; i < memDescs.size(); i++) {
            inputs[i];
        }
    }

    void DropNode(const NodePtr& node);
    void DropDWConvNode(const NodePtr& node);

    /**
     * @brief Insert Reorder node at the edge-specified location.
     * The Reorder node must be inserted in case when there are inplace conflicts or the input and output tensor descriptors do not match.
     * The Reorder node rearranges the elements in memory according to inDesc and outDesc, or reinterprets memory descriptor without
     * rearrangement of elements if isOptimized is true.
     * @param edge
     * pointer to the edge in the graph where Reorder node will be inserted
     * @param layerName
     * Reorder layer name
     * @param inDesc
     * input memory descriptor
     * @param outDesc
     * output memory descriptor
     * @param isOptimized
     * optimization flag; if isOptimized is true then Reorder node does nothing
     * @param src_perm
     * optimization flag; permutation applied to input desc before passing to reorder primitive
     * @param scales
     * pointer to the blob containing scales
     * @return pointer to the new Reorder node.
     */
    NodePtr InsertReorder(EdgePtr edge, std::string layerName, const MemoryDesc& inDesc,
            const MemoryDesc& outDesc, bool isOptimized = false, const std::vector<int> & src_perm = {});

    /**
     * @brief Insert Node at the edge-specified location.
     * This method supports two regimes. First, the node is inserted without initialization (i.e. supported descriptors initialization,
     * supported primitive descriptors selection, etc.), which can be useful after the InitEdges() completes. The second is just inserting the
     * node without initialization.
     * @param edge
     * pointer to the edge in the graph where the node will be inserted
     * @param node
     * pointer to the inserted node
     * @param initNode
     * parameter that determines whether the node needs to be initialized
     * @return true in case of success, false otherwise.
     */
    bool InsertNode(EdgePtr edge, NodePtr node, bool initNode = false);

    /**
     * @brief Insert Node between two specified nodes.
     * This procedure creates two edges that link the parent and child nodes to the inserted one and adds all created objects to the graph.
     * This method supports two regimes. First, the node is inserted without initialization (i.e. supported descriptors initialization,
     * supported primitive descriptors selection, etc.), which can be useful after the InitEdges() completes. The second is just inserting the
     * node without initialization.
     * @param parent
     * pointer to the parent node
     * @param child
     * pointer to the child node
     * @param parentPort
     * port number of the parent node to which the inserted node should be connected
     * @param childPort
     * port number of the child node to which the inserted node should be connected
     * @param initNode
     * parameter that determines whether the node needs to be initialized
     * @return true in case of success, false otherwise.
     */
    bool InsertNode(NodePtr parent, NodePtr child, NodePtr node, int parentPort, int childPort, bool initNode = false);

    std::shared_ptr<ngraph::Function> dump() const;

    void ResetInferCount() { infer_count = 0; }

    void SortTopologically();

    bool hasDynamicInput() const {
        return graphHasDynamicInput;
    }

    Status getStatus() const {return status;}
    std::vector<NodePtr> inputs;
    std::vector<NodePtr> outputs;
    void InitGraph(bool optimize = true);

protected:
    void VisitNode(NodePtr node, std::vector<NodePtr>& sortedNodes);

    void ForgetGraphData() {
        status = Status::NotReady;

        inputNodesMap.clear();
        outputNodesMap.clear();
        graphNodes.clear();
        graphEdges.clear();
        _normalizePreprocMap.clear();
        syncNodesInds.clear();
    }
    Status status { Status::NotReady };

    // For dumping purposes. -1 - no counting, all other positive
    // values mean increment it within each Infer() call
    int infer_count = -1;

    bool reuse_io_tensors = true;

    MemoryPtr memWorkspace;

    std::vector<NodePtr> graphNodes;
    std::vector<EdgePtr> graphEdges;

    std::map<std::string, NormalizePreprocess> _normalizePreprocMap;
    std::string _name;

    bool graphHasDynamicInput = false;

    void Replicate(const InferenceEngine::CNNNetwork &network);
    void Replicate(const std::shared_ptr<const ov::Model> &subgraph);
    void InitNodes();
    void InitDescriptors();
    void ResolveInplaceDirections();
    void InitOptimalPrimitiveDescriptors();
    void InitEdges();
    bool ProcessDynNodes();
    void Allocate();
    void AllocateWithReuse();
    void ExtractExecutableNodes();
    void ExecuteNode(const NodePtr& node, const dnnl::stream& stream) const;
    void CreatePrimitivesAndExecConstants() const;
    void InferStatic(InferRequestBase* request);
    void InferDynamic(InferRequestBase* request);
    bool SyncNodes();
    void CleanUpNodes();
    void RegisterOptimization();

    friend class LegacyInferRequest;
    friend class intel_cpu::InferRequest;
    friend class intel_cpu::InferRequestBase;
    friend std::shared_ptr<ngraph::Function> dump_graph_as_ie_ngraph_net(const Graph &graph);

private:
    // TODO: change std::map to std::unordered_map
    std::map<std::string, NodePtr> inputNodesMap;
    std::map<std::string, NodePtr> outputNodesMap;

    std::unordered_map<std::string, ProxyMemoryMngrPtr> outputNodesMemMngrMap;

    // these node pointers (from graphNodes) are to avoid regular checking for
    // constantness of nodes in Infer methods and calls of
    // non-executable (optimized out) nodes, such as Input, Reshape, etc.
    std::vector<NodePtr> executableGraphNodes;

    std::unordered_map<Node*, size_t> syncNodesInds;

    GraphContext::CPtr context;

    void EnforceInferencePrecision();
    void EnforceBF16();
    void resolveInPlaceDirection(const NodePtr& node) const;
};

}   // namespace intel_cpu
}   // namespace ov
