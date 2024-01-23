# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""riscv transpose operator"""
from tvm import te
from .utils import get_simd_32bit_lanes
from .utils import intrin_layout_transform


def _schedule_transpose(s, out, axes):
    """Schedule for transpose.
    Parameters
    ----------
    s: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the transpose op.
    Returns
    -------
    s: Schedule
         The updated schedule.
    """

    def my_prod(shape):
        result = 1
        for num in shape:
            result *= num
        return result

    length = len(s[out].op.axis)
    input_shape = s[out].op.input_tensors[0].shape
    dtype = s[out].op.input_tensors[0].dtype
    fused = s[out].fuse(*s[out].op.axis[0:length])

    simd_width = get_simd_32bit_lanes()
    factor = 1
    for tmp in range(simd_width, 0, -1):
        if out.shape[-1] % tmp == 0:
            factor = tmp
            break

    index = (int)(axes[-1]) if axes else 0
    stride = 1 if index == length - 1 else my_prod(input_shape[index + 1 :])
    num = length - 1 - index

    lo, li = s[out].split(fused, factor)
    s[out].parallel(lo)
    load_intrin = intrin_layout_transform(factor, dtype, stride, num)
    s[out].tensorize(li, load_intrin)

    return s


def schedule_transpose(outs, attrs):
    """Schedule for transpose.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of transpose
          in the format of an array of tensors.
    attrs: Attrs of the transpose.

    Returns
    -------
    s: Schedule
        The computation schedule for transpose.
    """
    axes = attrs.axes
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    _schedule_transpose(s, outs[0], axes)

    return s
