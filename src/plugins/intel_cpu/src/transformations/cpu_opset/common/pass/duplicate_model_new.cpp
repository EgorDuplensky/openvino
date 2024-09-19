// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "duplicate_model_new.hpp"
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"
#include "transformations/cpu_opset/common/op/fully_connected.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "transformations/cpu_opset/common/op/submodel.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"

static std::string v(const std::shared_ptr<ov::Node>& node) {
    return node->get_friendly_name() + ":" + node->get_type_name();
}

bool ov::intel_cpu::DuplicateModelNew::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(DuplicateModelNew);

    int sub_stream_num = 2;  // @todo pass as argument

    // find last concat node. There is no reason to duplicate graph after last concat node.
    const auto& ops = f->get_ordered_ops();
    auto r_last_cc = std::find_if(ops.rbegin(), ops.crend(), [](const std::shared_ptr<ov::Node>& node) {
        return node->get_rt_info().count("sync_point");
    });
    auto last_cc = r_last_cc == ops.rend() ? ops.end() : --r_last_cc.base();

    // no split node found
    if (last_cc == ops.end()) {
        return false;  // model has not been changed
    }

    (*last_cc)->get_rt_info()["final_sync_point"] = true;

    // std::cout << "### Running DuplicateModelNew transformation" << "\n";

    std::unordered_set<Node*> nodes_done;
    std::unordered_set<Node*> split_fc_nodes;

    for (const auto& op : ops) {
        if (op->get_rt_info().count("split_part")) {
            nodes_done.insert(op.get());
        }
    }

    std::vector<std::shared_ptr<ov::Node>> ordered_split_nodes;
    std::vector<std::vector<std::shared_ptr<ov::Node>>> ordered_nodes_for_node;

    for (const auto& op : ops) {
        if (ov::is_type<ov::op::v0::Parameter>(op))
            nodes_done.insert(op.get());
    }

    // collect all the nodes from the root (the node we split, i.e. FC) until the stop points (i.e. Parameters or other root nodes)
    for (const auto& op : ops) {
        if (op->get_rt_info().count("main_split_root")) {
            nodes_done.erase(op.get());
            ordered_split_nodes.push_back(op);
            ordered_nodes_for_node.emplace_back(ov::topological_sort_new<std::vector<std::shared_ptr<ov::Node>>>({op}, nodes_done));
            nodes_done.insert(op.get());
        }
    }

    class SubgraphBuilder {
    public:
        SubgraphBuilder(std::shared_ptr<ov::Model> model)
            : m_model(model)
        {}

        std::shared_ptr<ov::Node> add_duplicate(
            const std::shared_ptr<ov::Node>& node,
            size_t idx,
            std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
            std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones) {
            OutputVector new_inputs;
            std::cout << "Duplicating node: " << v(node) << "\n";

            for (const auto& input : node->inputs()) {
                auto input_source_output = input.get_source_output();  // @todo is there a another way?
                auto input_node = input_source_output.get_node_shared_ptr();

                if (m_output_clones.count(input_source_output)) {
                    // input is already a part of the subgraph
                    auto output_clone = m_output_clones[input_source_output];
                    new_inputs.emplace_back(output_clone);
                    // std::cout << "Input node: " << v(input_node) << " is already a part of SubModel" << "\n";
                } else {
                    // input is on constant path, basically do nothing
                    new_inputs.emplace_back(input.get_source_output());
                    // std::cout << "Input node: " << v(input_node) << " is constant" << "\n";
                }

                std::cout << "New input: " << v(new_inputs.back().get_node_shared_ptr()) << "\n";
            }

            auto clone = node->clone_with_new_inputs(new_inputs);

            ov::copy_runtime_info(node, clone);
            // @todo make sure all the extra runtime info is cleaned up after the transformation
            // if (clone->get_rt_info().count("main_split")) {
            //     clone->get_rt_info().erase("main_split");
            // }
            clone->get_rt_info()["numa_id"] = idx;

            clone->set_friendly_name(node->get_friendly_name() + "_clone_" + std::to_string(idx));
            std::cout << "Cloned: " << v(clone) << "\n";

            for (size_t i = 0; i < node->get_output_size(); i++) {
                m_output_clones[node->output(i)] = clone->output(i);
            }

            if (auto read_value = ov::as_type_ptr<ov::op::v6::ReadValue>(clone)) {
                auto variable = read_value->get_variable();
                auto new_variable_id = variable->get_info().variable_id + "_clone_" + std::to_string(idx);
                auto variable_clone =
                    std::make_shared<op::util::Variable>(op::util::VariableInfo{variable->get_info().data_shape,
                                                                                variable->get_info().data_type,
                                                                                new_variable_id});
                read_value->set_variable(variable_clone);
                variable_clones[variable->get_info().variable_id][idx] = variable_clone;
            } else if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(clone)) {
                // special handling for Assign (state). Basically assign node needs to be duplicated as well
                // @todo Can we somehow generalize this to have Assign as a part of collected nodes to be included into
                // subgraph? currently we only go up when collecting the nodes. For Assign we would need to go down auto
                // assign_clone = as_type_ptr<ov::op::v6::Assign>(
                //     assign->clone_with_new_inputs({clone_output}));
                auto variable_id = assign->get_variable_id();
                auto variable_clone = variable_clones[variable_id][idx];
                assign->set_variable(variable_clone);
                m_model->add_sinks({assign});
                std::cout << "Adding variable: " << variable_clone->get_info().variable_id << "\n";
                m_model->add_variables({variable_clone});
            }

            return clone;
        }

        std::shared_ptr<ov::Node> add_sync_point(
            const std::shared_ptr<ov::Node>& node,
            size_t idx,
            std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
            std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones) {
            std::cout << "Cloning sync point: " << v(node) << "\n";

            OutputVector new_inputs = node->input_values();
            auto clone = node->clone_with_new_inputs(new_inputs);

            ov::copy_runtime_info(node, clone);

            clone->set_friendly_name(node->get_friendly_name() + "_clone_" + std::to_string(idx));
            std::cout << "Cloned: " << v(clone) << "\n";

            for (size_t i = 0; i < node->get_output_size(); i++) {
                m_output_clones[node->output(i)] = clone->output(i);
            }

            clone->get_rt_info()["numa_id"] = idx;
            clone->get_rt_info()["sync_point"] = true;

            return clone;
        }


        void finalize(const std::shared_ptr<ov::Node>& node, const int idx) {
            auto input = node->input(idx);
            std::cout << "Replacing input: " << v(input.get_node()->shared_from_this()) << " of node: " << v(node)
                      << "with new input: " << v(m_output_clones[input.get_source_output()].get_node_shared_ptr())
                      << "\n";

            input.replace_source_output(m_output_clones[input.get_source_output()]);
        }

        void build(const std::shared_ptr<ov::Node>& node,
                   const std::unordered_set<std::shared_ptr<ov::Node>>& split_nodes,
                   const int idx) {
            std::set<Output<Node>> split_clones;
            for (const auto& split_node : split_nodes) {
                for (size_t i = 0; i < split_node->get_output_size(); i++) {
                    split_clones.insert(split_node->output(i));
                }
            }

            ov::ResultVector results;
            auto clone = m_output_clones[node->get_default_output()].get_node_shared_ptr();
            m_submodel = std::make_shared<ov::Model>(results, m_parameters, node->get_friendly_name() + "_model");
            m_subgraph = std::make_shared<ov::intel_cpu::SubModel>(m_submodel);

            m_subgraph->set_friendly_name("SubModel_" + node->get_friendly_name() + "_clone");
            m_subgraph->get_rt_info()["sub_stream_id"] = idx - 1;
            m_subgraph->get_rt_info()["numa_id"] = idx;
            m_numa_id = idx;

            for (size_t i = 0; i < m_parameters.size(); i++) {
                m_subgraph->set_invariant_input(m_parameters[i], m_invariant_input[i]);
            }

            // std::cout << "Created subgraph: " << m_subgraph->get_friendly_name() << " Parameters: ";
            // for (const auto& parameter : m_parameters) {
            //     std::cout << v(parameter) << ", ";
            // }
            // std::cout << "\n";
        }

        void resolve_outside_connections(const std::unordered_set<std::shared_ptr<ov::Node>>& split_nodes,
                                      std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
                                      size_t idx) {
            std::set<Output<Node>> split_clones;
            for (const auto& split_node : split_nodes) {
                for (size_t i = 0; i < split_node->get_output_size(); i++) {
                    split_clones.insert(split_node->output(i));
                }
            }

            for (const auto& entry : m_output_clones) {
                const auto& orig_output = entry.first;
                const auto& clone_output = entry.second;

                auto target_inputs = orig_output.get_target_inputs();

                if (target_inputs.empty())
                    continue;

                bool has_outside_target_node =
                    std::any_of(target_inputs.begin(),
                                target_inputs.end(),
                                [this, &split_clones](const Input<Node>& input) {
                                    // for (const auto& output : input.get_node()->outputs()) {
                                    //     if (!m_output_clones.count(output) && !split_clones.count(output))
                                    //         return true;
                                    // }
                                    // return false;
                                    return !m_output_clones.count(input.get_node()->output(0)) &&
                                           !split_clones.count(input.get_node()->output(0));
                                });
                if (has_outside_target_node) {
                    bool output_added = false;

                    for (auto& target_input : target_inputs) {
                        auto target_node = target_input.get_node();
                        // special handling of the connections between SubModel nodes to avoid cross numa connections
                        if (ov::is_type<ov::intel_cpu::SubModel>(target_node)) {
                            auto numa_id = target_node->get_rt_info()["numa_id"].as<int>();
                            if (numa_id == m_numa_id || split_clones.count(orig_output)) {
                                auto target_node_output = target_input.get_node()->output(0);
                                if (!m_output_clones.count(target_node_output) && !split_clones.count(target_node_output)) {
                                    if (!output_added) {
                                        m_subgraph->set_output_size(m_subgraph->get_output_size() + 1);
                                        m_submodel->add_results({std::make_shared<ov::op::v0::Result>(clone_output)});
                                    }
                                    output_added = true;
                                    target_input.replace_source_output(m_subgraph->output(m_subgraph->get_output_size() - 1));
                                }
                                // for (const auto& output : target_input.get_node()->outputs()) {
                                //     if (!m_output_clones.count(output) && !split_clones.count(output)) {
                                //         if (!output_added) {
                                //             m_subgraph->set_output_size(m_subgraph->get_output_size() + 1);
                                //             m_submodel->add_results({std::make_shared<ov::op::v0::Result>(clone_output)});
                                //         }
                                //         output_added = true;
                                //         target_input.replace_source_output(m_subgraph->output(m_subgraph->get_output_size() - 1));
                                //     }
                                // }
                            }
                        } else if (auto* assign = dynamic_cast<ov::op::v6::Assign*>(target_node)) {
                                 // special handling for Assign (state). Basically assign node needs to be duplicated as well
                                 // @todo Can we somehow generalize this to have Assign as a part of collected nodes to be included into subgraph?
                                 // currently we only go up when collecting the nodes. For Assign we would need to go down
                                 auto assign_clone = as_type_ptr<ov::op::v6::Assign>(
                                     assign->clone_with_new_inputs({clone_output}));
                                 auto variable_id = assign->get_variable_id();
                                 auto variable_clone = variable_clones[variable_id][idx];
                                 assign_clone->set_variable(variable_clone);
                                 m_submodel->add_sinks({assign_clone});
                                 m_submodel->add_variables({variable_clone});
                        } else {
                            auto target_node_output = target_input.get_node()->output(0);
                            if (!m_output_clones.count(target_node_output) && !split_clones.count(target_node_output)) {
                                if (!output_added) {
                                    m_subgraph->set_output_size(m_subgraph->get_output_size() + 1);
                                    m_submodel->add_results({std::make_shared<ov::op::v0::Result>(clone_output)});
                                }
                                output_added = true;
                                target_input.replace_source_output(m_subgraph->output(m_subgraph->get_output_size() - 1));
                            }
                        }
                    }
                }
            }
        }

    private:
        std::map<Output<Node>, Output<Node>> m_output_clones;
        ov::ParameterVector m_parameters;
        ov::OutputVector m_invariant_input;
        ov::NodeVector m_nodes_to_replace;
        std::shared_ptr<ov::Model> m_submodel;
        std::shared_ptr<ov::Model> m_model;
        std::shared_ptr<ov::intel_cpu::SubModel> m_subgraph;
        int m_numa_id;
    };

    class DuplicateSubgraphBuilder {
    public:
        DuplicateSubgraphBuilder(size_t duplicates_num, std::shared_ptr<ov::Model> model)
            : m_subgraphBuilders(std::vector<SubgraphBuilder>(duplicates_num, model)) {}

        void add_duplicate(const std::shared_ptr<ov::Node>& node,
                           std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
                           std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones) {
            for (size_t i = 0; i < m_subgraphBuilders.size(); i++) {
                m_subgraphBuilders[i].add_duplicate(node, i, variable_clones, split_clones);
            }
        }

        void add_sync_point(const std::shared_ptr<ov::Node>& node,
                           std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
                           std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones) {
            for (size_t i = 0; i < m_subgraphBuilders.size(); i++) {
                m_subgraphBuilders[i].add_sync_point(node, i, variable_clones, split_clones);
            }
        }

        void finalize(const std::shared_ptr<ov::Node>& node) {
            for (size_t i = 0; i < m_subgraphBuilders.size(); i++) {
                m_subgraphBuilders[i].finalize(node, i);
            }
        }

        std::shared_ptr<ov::Node> add_split(const std::shared_ptr<ov::Node>& node,
                                            const int idx,
                                            std::unordered_map<std::string, op::util::VariableVector>& variable_clones,
                                            std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones) {
            auto sb = std::next(m_subgraphBuilders.begin(), idx);
            return sb->add_duplicate(node, idx, variable_clones, split_clones);
        }

        void build(const std::shared_ptr<ov::Node>& node,
                   const int idx,
                   const std::unordered_set<std::shared_ptr<ov::Node>>& split_nodes) {
            auto sb = std::next(m_subgraphBuilders.begin(), idx);
            sb->build(node, split_nodes, idx);
        }

        void resolve_outside_connections(const std::unordered_set<std::shared_ptr<ov::Node>>& split_nodes,
                                      std::unordered_map<std::string, op::util::VariableVector>& variable_clones) {
            for (size_t i = 0; i < m_subgraphBuilders.size(); i++) {
                m_subgraphBuilders[i].resolve_outside_connections(split_nodes, variable_clones, i);
            }
        }

    private:
        std::vector<SubgraphBuilder> m_subgraphBuilders;
    };

    std::vector<DuplicateSubgraphBuilder> builders(ordered_split_nodes.size(), DuplicateSubgraphBuilder(sub_stream_num, f));
    std::vector<std::unordered_set<std::shared_ptr<ov::Node>>> all_split_nodes(ordered_split_nodes.size());
    std::unordered_map<std::string, op::util::VariableVector> variable_clones;

    for (const auto& op : ops) {
        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(op)) {
            variable_clones[assign->get_variable_id()] = op::util::VariableVector(sub_stream_num);
        }
    }

    // for (size_t i = 0; i < ordered_split_nodes.size(); i++) {
    //     const auto& nodes = ordered_nodes_for_node[i];
    std::map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> split_clones;

    for (const auto& node : ops) {
        if (ov::is_type<ov::op::v0::Parameter>(node) || ov::is_type<ov::op::v0::Result>(node))
            continue;

        // @todo cache is_on_constant_path before the loop
        if (ov::op::util::is_on_constant_path(node->output(0))) {
            continue;
        }

        if (node->get_rt_info().count("main_split_root") || node->get_rt_info().count("main_split")) {
            auto split_nodes = node->get_rt_info()["split_parts"].as<std::vector<std::shared_ptr<ov::Node>>>();
            // track all split nodes to avoid creating extra connections
            // for (const auto& _node : split_nodes) {
            //     all_split_nodes[i].insert(_node);
            // }

            for (size_t split_idx = 0; split_idx < split_nodes.size(); split_idx++) {
                const auto& split_node = split_nodes[split_idx];
                auto split_clone = builders[0].add_split(split_node, split_idx, variable_clones, split_clones);
                split_clones[node] = split_clone;
                // if this is a root split (i.e. FC node), finish subgraph building
                // if (node->get_rt_info().count("main_split_root")) {
                //     builders[i].build(split_node, split_idx, all_split_nodes[i]);
                // }
            }
        } else if (node->get_rt_info().count("other_split")) {
            // @todo can we handle other_splits the same way we handle the main one?
            continue;
        } else if (node->get_rt_info().count("sync_point")) {
            builders[0].finalize(node);
            if (!node->get_rt_info().count("final_sync_point")) {
                builders[0].add_sync_point(node, variable_clones, split_clones);
            }
        } else {
            builders[0].add_duplicate(node, variable_clones, split_clones);
        }
    }

    // if (!ordered_split_nodes.empty()) {
    //     for (size_t i = 0; i < builders.size(); i++) {
    //         builders[i].resolve_outside_connections(all_split_nodes[i], variable_clones);
    //     }
    // }

    // for (auto& variables_pair : variable_clones) {
    //     auto& variable_id = variables_pair.first;
    //     auto orig_variable = f->get_variable_by_id(variable_id);
    //     auto orig_info = orig_variable->get_info();
    //     auto& variables = variables_pair.second;
    //     // use original variable info must be preserved, use it in 0 numa clone
    //     variables[0]->update(orig_info);
    // }

    ov::SinkVector sinks = f->get_sinks();
    for (auto sink : sinks) {
        if (auto assign = ov::as_type_ptr<ov::op::v6::Assign>(sink)) {
            if (variable_clones.count(assign->get_variable_id())) {
                auto variable_id = assign->get_variable_id();
                auto orig_variable = f->get_variable_by_id(variable_id);
                auto orig_info = orig_variable->get_info();
                auto& variables = variable_clones[variable_id];
                // use original variable info must be preserved, use it in 0 numa clone
                auto variable_clone_0_info = variables[0]->get_info();
                variables[0]->update(orig_info);
                orig_variable->update(variable_clone_0_info);

                std::cout << "Removing variable: " << orig_variable->get_info().variable_id << "\n";
                f->remove_variable(orig_variable);
                f->remove_sink(sink);
                variable_clones.erase(variable_id);
            }
        }
    }

    return true;
}
