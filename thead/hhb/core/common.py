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
"""Some common utils"""
import os
import importlib
import functools
import argparse
import re
import yaml
import logging
import collections
import sys

import numpy as np


ArgInfo = collections.namedtuple("_ArgInfo", ["name", "choices", "default_value", "help"])

HHB_REGISTERED_PARSER = []
ALL_ARGUMENTS_INFO = {}
HHB_IR = []
ALL_ARGUMENTS_DESC = {}
ARGS_DEST_TO_OPTIONS_STRING = {}
logger = logging.getLogger("HHB")


def hhb_version():
    """Version information"""
    __version__ = "1.12.0"
    __build_time__ = "20220426"
    return "HHB version: " + __version__ + ", build " + __build_time__


def hhb_exit(message):
    logger.error(message)
    sys.exit()


class HHBException(Exception):
    """HHB Exception"""


class AttributeDict(dict):
    def __init__(self, **kwargs):
        super(AttributeDict, self).__init__()
        for name in kwargs:
            setattr(self, name, kwargs[name])

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def hhb_register_parse(make_subparser):
    """
    Utility function to register a subparser for HHB.

    Functions decorated with `hhb_register_parse` will be invoked
    with a parameter containing the subparser instance they need to add itself to,
    as a parser.

    Example
    -------

        @hhb_register_parse
        def _example_parser(main_subparser):
            subparser = main_subparser.add_parser('example', help='...')
            ...

    """
    HHB_REGISTERED_PARSER.append(make_subparser)
    return make_subparser


def hhb_ir_helper(hhb_ir):
    """
    Utility function to register a HHB_IR.

    Classes decorated with `hhb_register_parse` will be put into HHB_IR

    Example
    -------

        @hhb_ir_helper
        class HHBIR_E(HHBIRBase):
            pass

    """
    HHB_IR.append(hhb_ir)
    return hhb_ir


def get_parameters_info(params_name="unknown"):
    def decorate(func):
        @functools.wraps(func)
        def inner_wrapper(parser):
            if not isinstance(parser, argparse.ArgumentParser):
                raise HHBException("invalid parser:{}".format(parser))
            before_args = vars(parser.parse_known_args()[0])
            value = func(parser)
            after_args = vars(parser.parse_known_args()[0])
            ALL_ARGUMENTS_INFO[params_name] = {
                key: value for key, value in after_args.items() if key not in before_args
            }
            return value

        return inner_wrapper

    return decorate


def argument_filter_helper(func):
    @functools.wraps(func)
    def inner_wrapper(filtered_args, extra=None):
        if not isinstance(filtered_args, AttributeDict):
            raise HHBException("invalid filtered_args:{}".format(filtered_args))
        if extra is not None and not isinstance(extra, AttributeDict):
            raise HHBException("invalid extra: {}".format(extra))
        value = func(filtered_args, extra)
        return value

    return inner_wrapper


def import_module_for_register(all_modules):
    """Dynamic importing libraries"""
    if not all_modules:
        return
    for m in all_modules:
        importlib.import_module(m)


def ensure_dir(directory):
    """Create a directory if not exists

    Parameters
    ----------

    directory : str
        File path to create
    """
    if directory is None:
        directory = "hhb_out"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_target(board):
    """Get the target info accorrding to the board type."""
    if board == "anole":
        target = "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard -device=anole"
    elif board == "light":
        target = "llvm -device=light"
    elif board == "hlight":
        target = "llvm -device=hlight"
    elif board == "asp":
        target = "llvm -device=asp"
    elif board == "i805":
        target = "llvm -mtriple=csky -mcpu=i805"
    elif board == "c860":
        target = "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard -device=c860"
    elif board == "c906":
        target = "llvm -mtriple=riscv -mcpu=c906 -mfloat-abi=hard -device=c906"
    elif board == "c908":
        target = "llvm -mtriple=riscv -mcpu=c908 -mfloat-abi=hard -device=c908"
    elif board == "x86_ref":
        target = "llvm"
    elif board == "ch8601":
        target = "llvm -device=ch8601"
    elif board == "dp1k":
        target = "llvm -device=dp1k"
    return target


def print_top5(value, output_name, shape):
    """Print the top5 infomation"""
    if not isinstance(value, np.ndarray):
        raise HHBException("Unsupport for {}, please input ndarray".format(type(value)))
    len_t = np.prod(value.size)
    pre = np.reshape(value, [len_t])
    ind = np.argsort(pre)
    if len_t > 5:
        ind = ind[len_t - 5 :]
    value = pre[ind]
    ind = ind[::-1]
    value = value[::-1]
    print(f"====== index:{output_name}, shape:{shape}, top5: ======")
    for (i, v) in zip(ind, value):
        print("{}:{}".format(i, v))


def convert_invalid_symbol(input_str):
    """Convert invalid symbols in string into '_' """
    invalid_symbol = r"[/:\s\.]"
    new_str = re.sub(invalid_symbol, "_", input_str)
    return new_str


def find_index(data, item):
    """Find item index in data

    Parameters
    ----------
    data: list or tuple
        The data that will be searched

    item: object
        To be searched item

    Returns
    -------
    res: int
        The index that item in data. If item is not in data, return -1.

    """
    if not isinstance(data, (list, tuple)):
        raise HHBException("Unsupport for type: {}".format(type(data)))

    res = -1
    if item in data:
        res = data.index(item)
    return res


def generate_config_file(config_file):
    logger.debug("save the cmd parameters info into: %s", config_file)
    with open(config_file, "w") as f:
        yaml.safe_dump(ALL_ARGUMENTS_INFO, f, default_flow_style=False)


def collect_arguments_info(actions):
    """Convert argparse.Actions into ArgInfo."""
    res = list()
    for a in actions:
        if a.dest not in ARGS_DEST_TO_OPTIONS_STRING:
            ARGS_DEST_TO_OPTIONS_STRING[a.dest] = a.option_strings
        # ignore hidden arguments.
        if a.help == "==SUPPRESS==":
            continue
        default_value = a.default
        if default_value == "==SUPPRESS==":
            default_value = None

        if not a.option_strings:
            name = [a.dest]
        else:
            name = a.option_strings

        tmp = ArgInfo(
            name=", ".join(name), choices=a.choices, default_value=default_value, help=a.help
        )
        res.append(tmp)
    return res


def generate_readme_file(output_dir="."):
    """Automatically README.md file for command line tools."""

    def _gen_table(key):
        res = list()
        res.append("\n| Arguments | choices | default | Note |\n")
        res.append("| ------------|---------|---------| ---- |\n")
        if key in ALL_ARGUMENTS_DESC:
            for arg in ALL_ARGUMENTS_DESC[key]:
                new_help = arg.help.replace("\n", " ")
                res.append(f"| {arg.name} | {arg.choices} | {arg.default_value} | {new_help} |\n")
        return res

    content = list(
        """<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

## HHB Command line tools
### 1. Introduction
HHB is a set of function related command line tools, its usage is similar
to gcc. There are two ways to use it: single-stage mode and multi-stages mode.

### 1.1 Single-stage command

> Usage: python hhb.py [OPTION]...

All arguments are listed as follows:
"""
    )

    # generate main command arguments.
    content.extend(_gen_table("main_command"))

    # generate import subcommand arguments.
    content.append(
        """
### 1.2 Multi-stages command

#### 1.2.1 import subcommand

> Usage: python hhb.py import [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("import"))

    # generate quantize subcommand arguments.
    content.append(
        """
#### 1.2.2 quantize subcommand

> Usage: python hhb.py quantize [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("quantize"))

    # generate codegen subcommand arguments.
    content.append(
        """
#### 1.2.3 codegen subcommand

> Usage: python hhb.py codegen [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("codegen"))

    # generate simulate subcommand arguments.
    content.append(
        """
#### 1.2.4 simulate subcommand

> Usage: python hhb.py simulate [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("simulate"))

    # generate profiler subcommand arguments.
    content.append(
        """
#### 1.2.5 profiler subcommand

> Usage: python hhb.py profiler [OPTION]...

All arguments are listed as follows:
"""
    )
    content.extend(_gen_table("profiler"))

    # generate examples
    content.append(
        r"""
### 2. How to use

HHB support two kind of modes: single-stage mode and multi-stages mode. And both modes can parser all command parameters from specify file.

#### 2.1 single-stage mode

For example, you can import, quantize and simulate the specified model by a single command as follows:

```Python
python hhb.py --simulate \
	-v -v -v \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 0.017 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
	--simulate-data n01440764_188.JPEG \
	--opt-level 3 \
	--board x86_ref \
	--model-file mobilenetv1.prototxt \
	mobilenetv1.caffemodel \
	--postprocess top5 \
```



#### 2.2 multi-stages mode

In this mode, you can compile model by executing multiply sub command.

##### 2.2.1 import model

```Python
python hhb.py import alexnet.prototxt alexnet.caffemodel -o model.relay --opt-level -1
```

##### 2.2.2 quantize model

```Python
python hhb.py quantize \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
    -o model_qnn \
	model.relay
```

##### 2.2.3 codegen

```Python
python hhb.py codegen \
	--board x86_ref \
	-o quant_codegen \
	model_qnn \
```

##### 2.2.4 simulate

```Python
python hhb.py simulate \
	--simulate-data /lhome/fern/aone/hhb/tests/thead/images/n01440764_188.JPEG \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--postprocess top5 \
	-o output \
	quant_codegen \
```


#### 2.3 using config file

You can generate a template config file by:

```bash
python hhb.py --generate-config -o config.yaml
```

change the part parameters...

The use the config file by:

```Bash
python hhb.py --config-file config.yaml --file mobilenetv1.prototxt mobilenetv1.caffemodel
```

#### 2.4 generate preprocessed dataset

You can generate dataset by provided preprocess parameters

```bash
python hhb.py --generate-dataset \
	-v -v -v \
	-sd /lhome/fern/aone/hhb/tests/thead/images \
	--input-shape "1 3 224 224" \
	-o gen_dataset_o \
	--data-resize 256 \
	--data-mean "103.94 116.78 123.68" \
	--data-scale 0.017 \
```

#### 2.5 benchmark test

You can do accuracy testing by 'benchmark' subcommand. Currently, we support classfication model only.

```bash
python hhb.py benchmark \
	-v -v -v \
	--board x86_ref \
	-cd quant.txt \
	-sd ILSVRC2012_img_val \
	--reference-label val.txt \
	--print-interval 50 \
	--save-temps \
	--data-scale 1 \
	--data-mean "103.94 116.98 123.68" \
	--data-resize 256 \
	-o alexnet_caffe_benchmark_o \
	--no-quantize \
	alexnet.prototxt \
	alexnet.caffemodel \
```

#### 2.6 profiler

You can profile model by 'profiler' subcommand.

```bash
python hhb.py profiler \
	-v -v -v \
	--ir-type relay \
	--indicator cal \
	--output-type binary json \
	--model-file alexnet.caffemodel alexnet.prototxt \
```
"""
    )

    if not os.path.exists(output_dir):
        logger.error("%s dose not exist." % output_dir)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.writelines(content)
