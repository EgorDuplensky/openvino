// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "common/pass/align_matmul_input_ranks.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"
#include "common/pass/convert_tile_to_seq_tiles.hpp"
#include "common/pass/convert_matmul_to_fc.hpp"
#include "common/pass/convert_to_power_static.hpp"
#include "common/pass/convert_to_leaky_relu.hpp"
#include "common/pass/convert_to_swish_cpu.hpp"
#include "common/pass/move_fc_reshape_to_weights.hpp"
#include "common/pass/fc_bias_fusion.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "transformations/cpu_opset/common/pass/split_fc.hpp"
#include "transformations/cpu_opset/common/pass/duplicate_model.hpp"
#include "transformations/cpu_opset/common/pass/move_convert_through_variadic_split.hpp"
#include "transformations/op_conversions/convert_fc_to_compressed.hpp"
#include "transformations/op_conversions/convert_fc_to_quantized_legacy.hpp"
#include "common/pass/rnn_sequences_optimization.hpp"
#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"
#include "transformations/defs.hpp"

#include "itt.hpp"

namespace ov {
namespace intel_cpu {

static std::string extraDump() {
    static auto env = std::getenv("EXTRA_DUMP");
    static std::string result = env ? std::string(env) : std::string{};

    return result;
}

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ov::Model> &model, int numSubStreams) {
    RUN_ON_FUNCTION_SCOPE(ConvertToCPUSpecificOpset);

    ov::pass::Manager manager("CPU:ConvertToCPUSpecificOpset");
    manager.set_per_pass_validation(false);

    // CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertMatMulToFC);
    if (std::getenv("EXTRA_DUMP")) {
        manager.run_passes(model);
        ov::pass::Serialize("after_fc.xml", "/dev/null").run_on_model(model);
        CPU_DISABLE_PASS_COMMON(manager, ConvertMatMulToFC);
    }

    std::vector<ov::element::Type> supported_compression_types {
        ov::element::u8,
        ov::element::i8,
        ov::element::u4,
        ov::element::i4,
        ov::element::nf4,
        ov::element::f4e2m1,
    };

    CPU_REGISTER_PASS_X64(manager, pass::ConvertFullyConnectedToFullyConnectedCompressed,
                          supported_compression_types,
                          [](size_t IC, size_t OC, size_t G) {
                              if (IC % G != 0 || IC / G < 4 || OC == 1) {
                                  return false;
                              }
                              return true;
                          });

    CPU_REGISTER_PASS_X64(manager, pass::ConvertFCToFCQuantizedLegacy);
    if (!extraDump().empty()) {
        manager.run_passes(model);
        ov::pass::Serialize("after_fc_quantized.xml", "/dev/null").run_on_model(model);
        CPU_DISABLE_PASS_COMMON(manager, ConvertMatMulToFC);
    }
    CPU_REGISTER_PASS_COMMON(manager, FullyConnectedBiasFusion);
    CPU_REGISTER_PASS_X64(manager, MoveFCReshapeToWeights);
    CPU_REGISTER_PASS_X64(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_X64(manager, ov::pass::Serialize, "before_split.xml", "/dev/null");
    // ov::pass::Serialize("before_split.xml", "/dev/null").run_on_model(model);

    if ((numSubStreams >= 1 || std::getenv("FORCE_SPLIT")) && !std::getenv("DISABLE_SPLIT")) {
        std::cout << "numSubStreams: " << numSubStreams << "\n";

        if (std::getenv("FORCE_SPLIT")) {
            numSubStreams = 2;
        }

        CPU_REGISTER_PASS_COMMON(manager, SplitFC, numSubStreams);
        if (std::getenv("EXTRA_DUMP")) {
            manager.run_passes(model);
            std::cout << "### Dumping graph after SplitFC" << "\n";
            ov::pass::Serialize("after_split.xml", "/dev/null").run_on_model(model);
            CPU_DISABLE_PASS_COMMON(manager, SplitFC);
        }

        // else {
        //     CPU_REGISTER_PASS_COMMON(manager, SplitFCbyK, numSubStreams);
        //     if (std::getenv("EXTRA_DUMP")) {
        //         manager.run_passes(model);
        //         std::cout << "### Dumping graph after SplitFCbyK" << "\n";
        //         ov::pass::Serialize("after_split.xml", "/dev/null").run_on_model(model);
        //         CPU_DISABLE_PASS_COMMON(manager, SplitFCbyK);
        //     }
        // }

        CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

        if (!std::getenv("DISABLE_DUP") && !std::getenv("ENABLE_SUBGRAPH")) {
            // CPU_REGISTER_PASS_COMMON(manager, DuplicateModelNew);
            CPU_REGISTER_PASS_COMMON(manager, DuplicateModel);
            // manager.register_pass<pass::PrintModel>("print_model.txt");
            CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);

            if (std::getenv("EXTRA_DUMP")) {
                manager.run_passes(model);
                std::cout << "### Dumping graph after DuplicateModel" << "\n";
                ov::pass::Serialize("after_duplicate.xml", "/dev/null").run_on_model(model);
                CPU_DISABLE_PASS_COMMON(manager, DuplicateModel);
                // CPU_DISABLE_PASS_COMMON(manager, DuplicateModelNew);
            }
        }

        if (!extraDump().empty()) {
            std::cout << "EXTRA_DUMP = " << extraDump() << "\n";

            for (const auto& op : model->get_ordered_ops()) {
                if (const auto submodel = ov::as_type_ptr<SubModel>(op)) {
                    if (extraDump() == "all" || extraDump() == submodel->get_friendly_name())
                        ov::pass::Serialize(submodel->get_friendly_name() + ".xml", "/dev/null").run_on_model(submodel->get_function());
                }
            }
        }

        CPU_REGISTER_PASS_COMMON(manager, MoveConvertThroughVariadicSplit);
        CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    }

    CPU_REGISTER_PASS_COMMON(manager, AlignMatMulInputRanks);
    CPU_REGISTER_PASS_COMMON(manager, ConvertTileToSeqTiles);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToPowerStatic);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToLeakyRelu);
    CPU_REGISTER_PASS_COMMON(manager, ConvertToSwishCPU);
    CPU_REGISTER_PASS_COMMON(manager, OptimizeSequenceTransposes);
    // after transformation "MoveEltwiseUpThroughDataMov" there can be reshaped sequences that should be eliminated or fused
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ReshapeSequenceFusion);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::ConstantFolding);
    CPU_REGISTER_PASS_COMMON(manager,
                             ov::pass::ConvertPrecision,
                             precisions_map{{ov::element::i64, ov::element::i32}},
                             type_to_fuse_map{{}},
                             false,
                             false);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::Validate);
    CPU_REGISTER_PASS_COMMON(manager, ov::pass::EliminateConvert); // Need to clean up after the ConvertPrecision.

    manager.run_passes(model);
}

}   // namespace intel_cpu
}   // namespace ov
