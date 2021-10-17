/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/qnn/op/global_maxpool.cc
 * \brief Property def of qnn global_maxpool operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/tir/data_layout.h>

#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.conv2d
TVM_REGISTER_NODE_TYPE(QnnCSIGlobalMaxPoolAttrs);

bool QnnCSIGlobalMaxPoolRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto dshape = data->shape;
  CHECK_GE(dshape.size(), 2U)
      << "Pool2D only support input >= 2-D: input must have height and width";
  const auto param = attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  CHECK(param != nullptr);

  Layout layout(param->layout);
  CHECK(layout.Contains(LayoutAxis::Get('H')) && layout.Contains(LayoutAxis::Get('W')) &&
        !layout.Contains(LayoutAxis::Get('h')) && !layout.Contains(LayoutAxis::Get('w')))
      << "Invalid layout " << layout << ". Pool2D layout must have H and W, which cannot be split";

  const auto hidx = layout.IndexOf(LayoutAxis::Get('H'));
  const auto widx = layout.IndexOf(LayoutAxis::Get('W'));
  Array<IndexExpr> oshape(dshape);
  oshape.Set(hidx, 1);
  oshape.Set(widx, 1);

  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeQnnCSIGlobalMaxPool(Expr data, std::string layout, double input_scale,
                             int32_t input_zero_point, double output_scale,
                             int32_t output_zero_point, DataType out_dtype,
                             Array<IndexExpr> max_values, Array<IndexExpr> min_values,
                             String layer_name) {
  auto attrs = make_object<QnnCSIGlobalMaxPoolAttrs>();
  attrs->layout = std::move(layout);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->input_scale = std::move(input_scale);
  attrs->output_scale = std::move(output_scale);
  attrs->out_dtype = std::move(out_dtype);
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.global_maxpool");
  return Call(op, {data}, Attrs(attrs), {});
}

// GlobalAvgPool
RELAY_REGISTER_OP("qnn.csi.global_maxpool")
    .describe(R"code(Global max pooling operation for 2D data.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, channels, height, width) if `layout` is `NCHW`.
- **out**: This depends on the `layout` parameter. Output is 4D array of shape
           (batch_size, channels, 1, 1)  if `layout` is `NCHW`.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIGlobalMaxPoolAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized input data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIGlobalMaxPoolAttrs", QnnCSIGlobalMaxPoolRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIGlobalMaxPool").set_body_typed(MakeQnnCSIGlobalMaxPool);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
