# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Daniele Jahier Pagliari <daniele.jahier@polito.it>                *
# *----------------------------------------------------------------------------*
from plinio.cost import CostSpec
from plinio.cost.pattern import Conv1dGeneric, Conv2dGeneric, LinearGeneric

ROGUE_MAC_CYCLE = 1.8
ROGUE_FREQ = 50e06

def _lat_conv1d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    macs_per_t = cout * (cin * k[0] + (1 if spec['_parameters']['bias'] is not None else 0))
    macs = macs_per_t * out_shape[2]
    latency = macs / (ROGUE_MAC_CYCLE * ROGUE_FREQ)
    return latency


def _lat_conv2d_generic(spec):
    cin = spec['in_channels']
    cout = spec['out_channels']
    k = spec['kernel_size']
    out_shape = spec['output_shape']
    macs_per_pixel = cout * (cin * k[0] * k[1] + (1 if spec['_parameters']['bias'] is not None else 0))
    macs = macs_per_pixel * out_shape[2] * out_shape[3]
    latency = macs / (ROGUE_MAC_CYCLE * ROGUE_FREQ)
    return latency


def _lat_linear_generic(spec):
    cin = spec['in_features']
    cout = spec['out_features']
    macs = cout * (cin + (1 if spec['_parameters']['bias'] is not None else 0))
    latency = macs / (ROGUE_MAC_CYCLE * ROGUE_FREQ)
    return latency


rogue_latency = CostSpec(shared=False, default_behavior='zero')
rogue_latency[Conv1dGeneric] = _lat_conv1d_generic
rogue_latency[Conv2dGeneric] = _lat_conv2d_generic
rogue_latency[LinearGeneric] = _lat_linear_generic
