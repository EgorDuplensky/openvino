// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNEmbeddingBagSumNode {
public:
    MKLDNNEmbeddingBagSumNode(const std::shared_ptr<ngraph::Node>&,
                              size_t requiredInputsNum,
                              size_t indicesIdx,
                              size_t perSampleWeightsIdx,
                              size_t defaultIndexIdx);

    void execute(const uint8_t* srcData,
                 const uint8_t* weightsData,
                 uint8_t* dstData,
                 const InferenceEngine::Precision& srcPrc,
                 const InferenceEngine::SizeVector& inDims,
                 const InferenceEngine::SizeVector& outDims);

    ~MKLDNNEmbeddingBagSumNode() = default;

protected:
    virtual void initFromInputs() = 0;
    virtual void getIndices(int embIndex, const int*& indicesRef, size_t& size, int& weightsIdx, bool& withWeights) = 0;

    template <typename T>
    void processData(const T* srcData,
                     const T* weightsData,
                     T* dstData,
                     const InferenceEngine::SizeVector& inDataDims,
                     const InferenceEngine::SizeVector& outDataDims);

    const size_t EMB_TABLE_IDX = 0lu;
    const size_t INDICES_IDX;
    const size_t PER_SAMPLE_WEIGHTS_IDX;
    const size_t DEFAULT_INDEX_IDX;

    bool _withWeights = false;
    size_t _embDepth = 0;
    std::string _layerName;
};

}  // namespace MKLDNNPlugin
