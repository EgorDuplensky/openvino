// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend_visibility.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Batched convolution operation, with optional window dilation and stride.
            ///
            class BACKEND_API Convolution : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Convolution", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched convolution operation.
                Convolution() = default;
                /// \brief Constructs a batched convolution operation.
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                /// \param padding_below The padding-below sizes.<br>
                /// `[f]`
                /// \param padding_above The padding-above sizes.<br>
                /// `[f]`
                /// \param data_dilation_strides The data dilation strides.<br>
                /// `[f]`
                /// \param pad_type The pad type for automatically computing padding sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above,
                            const Strides& data_dilation_strides,
                            const PadType& pad_type = PadType::EXPLICIT);

                /// \brief Constructs a batched convolution operation with no data dilation (i.e.,
                /// all
                ///        data dilation strides are 1).
                /// ngraph/test/runtime/interpreter/unit_test.manifest
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                /// \param padding_below The padding-below sizes.<br>
                /// `[f]`
                /// \param padding_above The padding-above sizes.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above);

                /// \brief Constructs a batched convolution operation with no padding or data
                /// dilation
                ///        (i.e., padding above and below are 0 everywhere, and all data dilation
                ///        strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                /// \param window_dilation_strides The window dilation strides.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides);

                /// \brief Constructs a batched convolution operation with no window dilation,
                /// padding,
                ///        or data dilation (i.e., padding above and below are 0 everywhere, and all
                ///        window/data dilation strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                /// \param window_movement_strides The window movement strides.<br>
                /// `[f]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch,
                            const Output<Node>& filters,
                            const Strides& window_movement_strides);

                /// \brief Constructs a batched convolution operation with no window dilation or
                ///        movement stride (i.e., padding above and below are 0 everywhere, and all
                ///        window/data dilation strides and window movement strides are 1).
                ///
                /// \param data_batch The node producing the input data batch tensor.<br>
                /// `[N, C_IN, D1, ... Df]`
                /// \param filters The node producing the filters tensor.<br>
                /// `[C_OUT, C_IN, F1, ... Ff]`
                ///
                /// Output `[N, C_OUT, R1, ... Rf]`
                ///
                Convolution(const Output<Node>& data_batch, const Output<Node>& filters);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The window movement strides.
                const Strides& get_window_movement_strides() const
                {
                    return m_window_movement_strides;
                }
                void set_window_movement_strides(const Strides& window_movement_strides)
                {
                    m_window_movement_strides = window_movement_strides;
                }
                /// \return The window dilation strides.
                const Strides& get_window_dilation_strides() const
                {
                    return m_window_dilation_strides;
                }
                void set_window_dilation_strides(const Strides& window_dilation_strides)
                {
                    m_window_dilation_strides = window_dilation_strides;
                }
                /// \return The padding-below sizes (possibly negative).
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const CoordinateDiff& padding_below)
                {
                    m_padding_below = padding_below;
                }
                /// \return The padding-above sizes (possibly negative).
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                void set_adding_above(const CoordinateDiff& padding_above)
                {
                    m_padding_above = padding_above;
                }
                /// \return The input data dilation strides.
                const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
                void set_data_dilation_strides(const Strides& data_dilation_strides)
                {
                    m_data_dilation_strides = data_dilation_strides;
                }
                /// \return The pad type for convolution.
                const PadType& get_pad_type() const { return m_pad_type; }
                void set_pad_type(const PadType& pad_type) { m_pad_type = pad_type; }
                /// \return The default value for Convolution.
                NGRAPH_SUPPRESS_DEPRECATED_START
                virtual std::shared_ptr<Node> get_default_value() const override;
                NGRAPH_SUPPRESS_DEPRECATED_END

            protected:
                Strides m_window_movement_strides;
                Strides m_window_dilation_strides;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Strides m_data_dilation_strides;
                PadType m_pad_type;
            };

            /// \brief Data batch backprop for batched convolution operation.
            class BACKEND_API ConvolutionBackpropData : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ConvolutionBackpropData", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a batched-convolution data batch-backprop operation.
                ConvolutionBackpropData() = default;
                ///
                /// \brief      Constructs a batched-convolution data batch-backprop operation.
                ///
                /// \param      data_batch_shape                 The shape of the data batch from
                ///                                              forward-prop.
                /// \param      filters                          The node producing the filters from
                ///                                              forward-prop.
                /// \param      data                             The node producing output delta.
                /// \param      window_movement_strides_forward  The window movement strides from
                ///                                              forward-prop.
                /// \param      window_dilation_strides_forward  The window dilation strides from
                ///                                              forward-prop.
                /// \param      padding_below_forward            The padding-below sizes from
                ///                                              forward-prop.
                /// \param      padding_above_forward            The padding-above sizes from
                ///                                              forward-prop.
                /// \param      data_dilation_strides_forward    The data dilation strides from
                ///                                              forward-prop.
                ///
                ConvolutionBackpropData(const Shape& data_batch_shape,
                                        const Output<Node>& filters,
                                        const Output<Node>& data,
                                        const Strides& window_movement_strides_forward,
                                        const Strides& window_dilation_strides_forward,
                                        const CoordinateDiff& padding_below_forward,
                                        const CoordinateDiff& padding_above_forward,
                                        const Strides& data_dilation_strides_forward);

                void validate_and_infer_types() override;
                bool visit_attributes(AttributeVisitor& visitor) override;

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

                /// \return The data batch shape.
                const Shape& get_data_batch_shape() const { return m_data_batch_shape; }
                void set_data_batch_shape(const Shape& data_batch_shape)
                {
                    m_data_batch_shape = data_batch_shape;
                }
                /// \return The window movement strides from the forward prop.
                const Strides& get_window_movement_strides_forward() const
                {
                    return m_window_movement_strides_forward;
                }
                void set_window_movement_strides_forward(
                    const Strides& window_movement_strides_forward)
                {
                    m_window_movement_strides_forward = window_movement_strides_forward;
                }
                /// \return The window dilation strides from the forward prop.
                const Strides& get_window_dilation_strides_forward() const
                {
                    return m_window_dilation_strides_forward;
                }
                void set_window_dilation_strides_forward(
                    const Strides& window_dilation_strides_forward)
                {
                    m_window_dilation_strides_forward = window_dilation_strides_forward;
                }
                /// \return The padding-below sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_padding_below_forward() const
                {
                    return m_padding_below_forward;
                }
                void set_padding_below_forward(const CoordinateDiff& padding_below_forward)
                {
                    m_padding_below_forward = padding_below_forward;
                }
                /// \return The padding-above sizes (possibly negative) from the forward prop.
                const CoordinateDiff& get_padding_above_forward() const
                {
                    return m_padding_above_forward;
                }
                void set_padding_above_forward(const CoordinateDiff& padding_above_forward)
                {
                    m_padding_above_forward = padding_above_forward;
                }
                /// \return The input data dilation strides from the forward prop.
                const Strides& get_data_dilation_strides_forward() const
                {
                    return m_data_dilation_strides_forward;
                }
                void set_data_dilation_strides_forward(const Strides& data_dilation_strides_forward)
                {
                    m_data_dilation_strides_forward = data_dilation_strides_forward;
                }

                // Compute the pad_above values to be used if in a convolution
                CoordinateDiff compute_backward_delta_out_pad_above() const;
                CoordinateDiff compute_backward_delta_out_pad_below() const;

            protected:
                Shape m_data_batch_shape;
                Strides m_window_movement_strides_forward;
                Strides m_window_dilation_strides_forward;
                CoordinateDiff m_padding_below_forward;
                CoordinateDiff m_padding_above_forward;
                Strides m_data_dilation_strides_forward;
            };
        } // namespace v0
    }     // namespace op
} // namespace ngraph
