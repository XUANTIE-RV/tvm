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
 * \file pad.cc
 * \brief Implementation of operator pad
 */
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/op.h>
#include <tvm/topi/nn.h>

#include <vector>

#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.nn.pad
TVM_REGISTER_NODE_TYPE(QnnCSIPadAttrs);

bool QnnCSIPadRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const QnnCSIPadAttrs* param = attrs.as<QnnCSIPadAttrs>();
  CHECK(param != nullptr);

  // check that pad widths match lengths
  CHECK(data->shape.size() == param->pad_width.size())
      << "There should be as many pad width pairs as shape dimensions "
      << "but the shape has " << data->shape.size() << " dimensions "
      << "and there are " << param->pad_width.size() << " pad width pairs.";

  // each pad width element should be a pair of positive integers
  std::vector<IndexExpr> oshape;
  for (size_t i = 0; i < param->pad_width.size(); i++) {
    CHECK(param->pad_width[i].size() == 2)
        << "Each pad width element should be a pair but at index " << i << " there are "
        << param->pad_width[i].size() << " elements.";

    auto width1 = tir::as_const_int(param->pad_width[i][0]);
    auto width2 = tir::as_const_int(param->pad_width[i][1]);
    CHECK(width1 != nullptr);
    CHECK(width2 != nullptr);

    CHECK(*width1 >= 0) << "Param width elements should be positive but first pad width at "
                        << "index " << i << " is " << *width1 << ".";
    CHECK(*width2 >= 0) << "Param width elements should be positive but first pad width at "
                        << "index " << i << " is " << *width2 << ".";

    if (!data->shape[i].as<tir::AnyNode>()) {
      auto padding = tir::make_const(data->shape[i].dtype(), *width1 + *width2);
      oshape.push_back(data->shape[i] + padding);
    } else {
      oshape.push_back(data->shape[i]);
    }
  }

  reporter->Assign(types[1], TensorType(Array<IndexExpr>(oshape), data->dtype));
  return true;
}

// Handler to create a call to the padding op used by front-end FFI
Expr MakeQnnCSIPad(Expr data, Array<Array<IndexExpr> > pad_width, double pad_value,
                   std::string pad_mode, double input_scale, int32_t input_zero_point,
                   double output_scale, int32_t output_zero_point, DataType out_dtype,
                   Array<IndexExpr> max_values, Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIPadAttrs>();
  attrs->pad_value = pad_value;
  attrs->pad_width = std::move(pad_width);
  attrs->pad_mode = std::move(pad_mode);
  attrs->input_scale = std::move(input_scale);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.pad");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.pad")
    .describe(R"code(Pad for n-D tensor.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIPadAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIPad", QnnCSIPadRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIPad").set_body_typed(MakeQnnCSIPad);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
