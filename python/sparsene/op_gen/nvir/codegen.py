from __future__ import annotations

from enum import Enum
from typing import List, Dict, Tuple, Optional, Sequence
from dataclasses import dataclass
from sparsene.op_gen.nvir.nvop import (
    ConstantNvOp,
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpImpl,
    OpMemType,
    IoMemType,
    ShapeEle,
    Shape,
    VarlenShape,
    IntShape,
    ParamShape,
    MnkShape,
    NvOpSequence,
    NvOpPipeline,
    NvOpProgram,
    NvOpTensor,
    ForLoopNvOp,
    GmemTensorNvOp,
)
import sparsene.op_gen.nvir.templates as templates
from itertools import chain
from sparsene.logging import get_logger

import re

logger = get_logger(__name__)


class NvIrCodeGenerator:
    def __init__(self, indent_size: int = 4, debug_indent: bool = False):
        self.indent_size = indent_size
        self.debug_indent = debug_indent
        self.debug_chars = "."

    def _get_indent(self, indent_level: int) -> str:
        return (
            (self.debug_chars if self.debug_indent else " ")
            * indent_level
            * self.indent_size
        )

    def _indent_lines(self, text: str, indent_level: int = 1) -> str:
        """Helper method to indent all lines in a text string."""
        indent = self._get_indent(indent_level)
        return "\n".join(f"{indent}{line}" for line in text.split("\n"))

    def dump_nvop_class_def(self, op: NvOp) -> str:
        def dump_template_head() -> str:
            parameter_types_strs = [
                f"class {param} = Int<{op.parameters[param]}>"
                for param in op.parameters
            ]
            parameter_types_str = ", ".join(parameter_types_strs)
            parameter_types_str = (
                ", " + parameter_types_str if parameter_types_str else ""
            )

            in_types_strs = [
                f"class TI{idx} = {inp.tensor.dtype}"
                for idx, inp in enumerate(op.inputs)
            ]
            out_types_strs = [
                f"class TO{idx} = {out.tensor.dtype}"
                for idx, out in enumerate(op.outputs)
            ]
            io_types_str = ", ".join(chain(in_types_strs, out_types_strs))
            io_types_str = ", " + io_types_str if io_types_str else ""

            non_owning_tensor_types_strs = []
            for out in op.outputs:
                if not out.owning:
                    non_owning_tensor_types_strs.append(f"class Tensor_o{out.idx}")
            non_owning_tensor_types_str = (
                ", " + ", ".join(non_owning_tensor_types_strs)
                if non_owning_tensor_types_strs
                else ""
            )

            if op.pipelined:
                nbuf_type_str = "class NBUF, "
            else:
                nbuf_type_str = ""

            return f"template<{nbuf_type_str}class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK{parameter_types_str}{non_owning_tensor_types_str}{io_types_str}>"

        template_head_str = dump_template_head()

        class_head_str = f"class {op.name} {{\npublic:"

        inputs_str = f"// Inputs\n" + "\n".join(
            map(lambda inp: self.dump_input_decl(inp), op.inputs)
        )
        outputs_str = f"// Outputs\n" + "\n".join(
            map(lambda out: self.dump_output_decl(out), op.outputs)
        )

        hw_param_str = f"// Hardware parameters\nint tid, lid, wid;"

        def dump_constructor_head() -> str:
            hw_constructor_params = ["int tid", "int lid", "int wid"]
            output_constructor_params = []
            for out in op.outputs:
                if not out.owning:
                    output_constructor_params.append(f"Tensor_o{out.idx} {out.name}")
                else:
                    output_constructor_params.append(f"TO{out.idx} *{out.name}")
            varlen_modes_constructor_params = []
            for out in op.outputs:
                i: int = 0
                for s in out.tensor.shape:
                    if isinstance(s, VarlenShape):
                        varlen_modes_constructor_params.append(
                            f"int mode_o{out.idx}_{i}"
                        )
                        i += 1
            for inp in op.inputs:
                i: int = 0
                for s in inp.tensor.shape:
                    if isinstance(s, VarlenShape):
                        varlen_modes_constructor_params.append(
                            f"int mode_i{inp.idx}_{i}"
                        )
                        i += 1
            constructor_params = ", ".join(
                chain(
                    hw_constructor_params,
                    output_constructor_params,
                    varlen_modes_constructor_params,
                )
            )

            return f"CUTE_DEVICE {op.name}({constructor_params})"

        constructor_head_str = dump_constructor_head()
        hw_param_constructor_init_list = ["tid(tid)", "lid(lid)", "wid(wid)"]
        varlen_shapes_constructor_init_list = self.dump_init_input_varlen_shape(
            op
        ) + self.dump_init_output_varlen_shape(op)
        output_constructor_init_list = []
        for out in op.outputs:
            if not out.owning:
                output_constructor_init_list.append(f"{out.name}({out.name})")
            else:
                output_constructor_init_list.append(
                    f"{out.name}(make_tensor(make_{out.tensor.mem}_ptr({out.name}), Layout_o{out.idx}{{}}))"
                )

        constructor_init_list = (
            hw_param_constructor_init_list
            + varlen_shapes_constructor_init_list
            + output_constructor_init_list
        )
        constructor_init_list_str = ", ".join(constructor_init_list)
        constructor_str = f"{constructor_head_str}\n: {constructor_init_list_str} {{}}"

        get_output_method_str = self.dump_get_output_method(op)

        f_method_str = self.dump_f_method(op)

        return (
            f"{template_head_str}\n{class_head_str}\n"
            f"{self._indent_lines(inputs_str)}\n\n"
            f"{self._indent_lines(outputs_str)}\n\n"
            f"{self._indent_lines(hw_param_str)}\n\n"
            f"{self._indent_lines(constructor_str)}\n\n"
            f"{self._indent_lines(get_output_method_str)}\n\n"
            f"{self._indent_lines(f_method_str)}\n"
            "};"
        )

    def dump_gmem_tensor_nvop_class_def(self, op: GmemTensorNvOp) -> str:
        parameter_types_strs = [
            f"class {param} = Int<{op.parameters[param]}>" for param in op.parameters
        ]
        parameter_types_str = ", ".join(parameter_types_strs)
        parameter_types_str = ", " + parameter_types_str if parameter_types_str else ""

        in_dtypes_strs = [
            f"class TI{idx} = {inp.tensor.dtype}" for idx, inp in enumerate(op.inputs)
        ]
        out_dtypes_strs = [
            f"class TO{idx} = {out.tensor.dtype}" for idx, out in enumerate(op.outputs)
        ]
        io_dtypes_str = ", ".join(chain(in_dtypes_strs, out_dtypes_strs))
        io_dtypes_str = ", " + io_dtypes_str if io_dtypes_str else ""
        op_name_str = op.name
        tensor_type_str = op.gmem_inout.tensor_type_str
        tensor_str = op.gmem_inout.tensor_str
        tensor_name = op.gmem_inout.name

        varlen_modes_constructor_params = []
        for out in op.outputs:
            i: int = 0
            for s in out.tensor.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes_constructor_params.append(f"int {s.name}")
                    i += 1
        for inp in op.inputs:
            i: int = 0
            for s in inp.tensor.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes_constructor_params.append(f"int {s.name}")
                    i += 1
        varlen_modes_constructor_params_str = "".join(
            f", {param}" for param in varlen_modes_constructor_params
        )

        get_output_method_str = self.dump_get_output_method(op)

        return templates.GMEM_TENSOR_NVOP_CLASS_DEF_SKELETON.format(
            parameter_types=parameter_types_str,
            io_dtypes=io_dtypes_str,
            op_name=op_name_str,
            tensor_type_str=tensor_type_str,
            tensor_name=tensor_name,
            varlen_modes_constructor_params=varlen_modes_constructor_params_str,
            tensor_str=tensor_str,
            get_output_method=self._indent_lines(get_output_method_str),
        )

    def dump_constant_nvop_class_def(self, op: ConstantNvOp) -> str:
        nbuf_type_str = "class NBUF, " if op.pipelined else ""
        parameter_types_strs = [
            f"class {param} = Int<{op.parameters[param]}>" for param in op.parameters
        ]
        parameter_types_str = ", ".join(parameter_types_strs)
        parameter_types_str = ", " + parameter_types_str if parameter_types_str else ""

        in_dtypes_strs = [
            f"class TI{idx} = {inp.tensor.dtype}" for idx, inp in enumerate(op.inputs)
        ]
        out_dtypes_strs = [
            f"class TO{idx} = {out.tensor.dtype}" for idx, out in enumerate(op.outputs)
        ]
        io_dtypes_str = ", ".join(chain(in_dtypes_strs, out_dtypes_strs))
        io_dtypes_str = ", " + io_dtypes_str if io_dtypes_str else ""
        op_name_str = op.name

        outputs_str = f"// Outputs\n" + "\n".join(
            map(lambda out: self.dump_output_decl(out), op.outputs)
        )

        make_shape_str = self.dump_decl_make_shape(
            op.outputs[0].tensor.shape, nbuf=False
        )

        mem_type_str = op.outputs[0].tensor.mem
        tensor_name = op.outputs[0].name
        if op.value == 0:
            f_body = f"clear({tensor_name});"
        else:
            f_body = f"for (int i = 0; i < {tensor_name}.size(); i++) {{ {tensor_name}(i) = {op.value}; }}"

        get_output_method_str = self.dump_get_output_method(op)

        return templates.CONSTANT_NVOP_CLASS_DEF_SKELETON.format(
            nbuf_type=nbuf_type_str,
            parameter_types=parameter_types_str,
            io_dtypes=io_dtypes_str,
            op_name=op_name_str,
            outputs=self._indent_lines(outputs_str),
            make_shape_str=make_shape_str,
            mem_type_str=mem_type_str,
            tensor_name=tensor_name,
            f_body=f_body,
            get_output_method=self._indent_lines(get_output_method_str),
        )

    def dump_pipelined_for_loop_nvop_class_def(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        visible_op_type_template_params = [f"class T_{op.name}" for op in visible_ops]
        visible_op_type_template_params_str = "".join(
            f", {param}" for param in visible_op_type_template_params
        )

        non_owning_tensor_type_template_params = [
            f"class Tensor_o{out.idx}" for out in op.outputs
        ]
        non_owning_tensor_type_template_params_str = "".join(
            f", {param}" for param in non_owning_tensor_type_template_params
        )

        in_dtypes_strs = [
            f"class TI{idx} = {inp.tensor.dtype}" for idx, inp in enumerate(op.inputs)
        ]
        out_dtypes_strs = [
            f"class TO{idx} = {out.tensor.dtype}" for idx, out in enumerate(op.outputs)
        ]
        io_dtype_template_params_str = "".join(
            f", {dtype}" for dtype in chain(in_dtypes_strs, out_dtypes_strs)
        )

        parameter_types_strs = [
            f"class {param} = Int<{op.parameters[param]}>" for param in op.parameters
        ]
        parameter_template_params_str = "".join(
            f", {param}" for param in parameter_types_strs
        )

        op_name_str = op.name

        inputs_str = f"// Inputs\n" + "\n".join(
            map(lambda inp: self.dump_input_decl(inp), op.inputs)
        )
        outputs_str = f"// Outputs\n" + "\n".join(
            map(lambda out: self.dump_output_decl(out), op.outputs)
        )

        get_output_method_str = self.dump_get_output_method(op)

        visible_op_strs = [f"T_{op.name} &{op.name}_v;" for op in visible_ops]
        visible_ops_str = "\n".join(visible_op_strs)

        visible_op_as_params_strs = [f"T_{op.name} &{op.name}_v" for op in visible_ops]
        visible_op_as_params_str = "".join(
            f", {param}" for param in visible_op_as_params_strs
        )

        visible_op_init_strs = [f"{op.name}_v({op.name}_v)" for op in visible_ops]
        visible_op_inits_str = "".join(
            f", {init_str}" for init_str in visible_op_init_strs
        )

        pipeline_methods = self.dump_pipelined_for_loop_nvop_methods(op)

        get_output_method_str = self.dump_get_output_method(op)

        non_owning_tensor_as_params_strs = [
            f"Tensor_o{out.idx} {out.name}" for out in op.outputs
        ]
        non_owning_tensor_as_params_str = "".join(
            f", {param}" for param in non_owning_tensor_as_params_strs
        )

        non_owning_tensor_inits_strs = [f"{out.name}({out.name})" for out in op.outputs]
        non_owning_tensor_inits_str = "".join(
            f", {init_str}" for init_str in non_owning_tensor_inits_strs
        )

        return templates.PIPELINED_FOR_LOOP_NVOP_DEF_SKELETON.format(
            visible_op_type_template_params=visible_op_type_template_params_str,
            non_owning_tensor_type_template_params=non_owning_tensor_type_template_params_str,
            non_owning_tensor_as_params=non_owning_tensor_as_params_str,
            non_owning_tensor_inits=non_owning_tensor_inits_str,
            io_dtype_template_params=io_dtype_template_params_str,
            parameter_template_params=parameter_template_params_str,
            op_name=op_name_str,
            outputs=self._indent_lines(outputs_str),
            visible_ops=self._indent_lines(visible_ops_str),
            visible_op_as_params=visible_op_as_params_str,
            visible_op_inits=visible_op_inits_str,
            pipeline_methods=self._indent_lines(pipeline_methods),
            get_output_method=self._indent_lines(get_output_method_str),
        )

    def dump_blk_idx_for_loop_nvop_class_def(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        def dump_loop_body() -> str:
            op_calls = []
            for bop in op.body.ops:
                op_calls.append(
                    self.dump_f_method_call(bop, buf_idx_str="", shift_constant="")
                )
            return "\n".join(op_calls)

        visible_op_type_template_params = [f"class T_{op.name}" for op in visible_ops]
        visible_op_type_template_params_str = "".join(
            f", {param}" for param in visible_op_type_template_params
        )

        parameter_types_strs = [
            f"class {param} = Int<{op.parameters[param]}>" for param in op.parameters
        ]
        parameter_template_params_str = "".join(
            f", {param}" for param in parameter_types_strs
        )

        op_name_str = op.name

        visible_op_strs = [f"T_{op.name} &{op.name}_v;" for op in visible_ops]
        visible_ops_str = "\n".join(visible_op_strs)

        visible_op_as_params_strs = [f"T_{op.name} &{op.name}_v" for op in visible_ops]
        visible_op_as_params_str = "".join(
            f", {param}" for param in visible_op_as_params_strs
        )

        visible_op_init_strs = [f"{op.name}_v({op.name}_v)" for op in visible_ops]
        visible_op_inits_str = "".join(
            f", {init_str}" for init_str in visible_op_init_strs
        )

        body_str = dump_loop_body()

        block_idx_dim_str = op.blk_idx_mapping
        assert block_idx_dim_str in ["x", "y", "z"]

        return templates.BLK_IDX_FOR_LOOP_NVOP_DEF_SKELETON.format(
            visible_op_type_template_params=visible_op_type_template_params_str,
            parameter_template_params=parameter_template_params_str,
            op_name=op_name_str,
            visible_ops=self._indent_lines(visible_ops_str),
            visible_op_as_params=visible_op_as_params_str,
            visible_op_inits=visible_op_inits_str,
            body=self._indent_lines(body_str, 2),
            block_idx_dim=block_idx_dim_str,
        )

    def dump_sequential_for_loop_nvop_class_def(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        if op.blk_idx_mapping:
            return self.dump_blk_idx_for_loop_nvop_class_def(op, visible_ops)

        raise NotImplementedError("Sequential for loop is not implemented")

    def dump_f_method(self, op: NvOp) -> str:
        def dump_nested_tuple_access(
            shape: str, shape_indices: List[int], size_outside: bool = False
        ) -> str:
            if len(shape_indices) == 0:
                return shape
            if size_outside:
                return f"size<{shape_indices[-1]}>({dump_nested_tuple_access(shape, shape_indices[:-1])})"
            else:
                return f"get<{shape_indices[-1]}>({dump_nested_tuple_access(shape, shape_indices[:-1])})"

        static_assert_strs = []

        def dump_static_assert() -> None:
            def dump_static_assert(
                shape: Shape, inp: NvOpInput, parent_shape_indices: List[int]
            ) -> None:
                for i, s in enumerate(shape):
                    if isinstance(s, VarlenShape):
                        continue
                    elif isinstance(s, Shape):
                        dump_static_assert(s, inp, parent_shape_indices + [i])
                    else:
                        static_assert_strs.append(
                            f"CUTE_STATIC_ASSERT_V(("
                            f"{dump_nested_tuple_access(f'shape_i{inp.idx}', parent_shape_indices + [i])}"
                            f" == "
                            f"{dump_nested_tuple_access(f'shape({inp.name})', parent_shape_indices + [i], size_outside=True)}"
                            "));"
                        )

            for inp in op.inputs:
                dump_static_assert(inp.tensor.shape, inp, [])
                # f"CUTE_STATIC_ASSERT_V((get<{i}>(shape_i{inp.idx}) == size<{i}>(shape({inp.name}))));"
                # f"CUTE_STATIC_ASSERT((is_static<decltype(size<{i}>(shape({inp.name})))>::value));"

        dump_static_assert()

        if op.pipelined:
            nbuf_template_param_str = "size_t buf_idx, "
        else:
            nbuf_template_param_str = ""

        code_str = "\n".join([*static_assert_strs, op.impl.code_template])
        f_template_head_str = (
            f"template<{nbuf_template_param_str}"
            # + (", size_t wait_num" if op.mem_type[0] == "s" else "")
            + ", ".join(
                f"class Tensor_i{inp.idx}"
                for inp in op.inputs
                if len(inp.tensor.shape) > 0
            )
            + ">"
        )
        f_input_list = []
        for inp in op.inputs:
            # f_input_list.append(f"Tensor_i{inp.idx} {inp.name}")
            if len(inp.tensor.shape) > 0:
                f_input_list.append(f"Tensor_i{inp.idx} {inp.name}")
            else:
                f_input_list.append(f"TI{inp.idx} {inp.name}")
        f_input_list_str = ", ".join(f_input_list)
        return (
            f"{f_template_head_str}\n"
            + f"CUTE_DEVICE void f({f_input_list_str}) {{\n"
            + self._indent_lines(code_str)
            + "\n"
            + "}"
        )

    def dump_f_method_call_arg(
        self,
        inp: NvOpInput,
        buf_idx_str: str,
        shift_constant: Optional[str] = None,
    ) -> str:
        # g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>()
        # l + i + shift1 + shift2
        if isinstance(inp.tensor.source, NvOpOutput):
            assert isinstance(inp.tensor.source.op, NvOp)
            if inp.tensor.source.op.pipelined:
                buf_idx_str = ", " + buf_idx_str
            else:
                buf_idx_str = ""
            arg = f"{inp.tensor.source.op.name}_v.template output<{inp.tensor.source.idx}{buf_idx_str}>()"
            if len(inp.tensor.shape) == 0:
                arg += "(0)"
            return arg
        elif isinstance(inp.tensor.source, str):
            if "{c}" in inp.tensor.source:
                if shift_constant is not None:
                    return inp.tensor.source.format(c=shift_constant)
                else:
                    raise ValueError("Missing value for 'c'")
            return inp.tensor.source
        else:
            raise ValueError(f"Unknown source type: {type(inp.tensor.source)}")

    def dump_f_method_call(
        self,
        op: NvOp,
        buf_idx_str: str,
        shift_constant: Optional[str] = None,
    ) -> str:
        # g2s_coo_atomic_format_load_val_op.template f<(i - shift1) % nbuf_rmem>(val_coo_val,
        #     g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>(),
        #     g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 1>()
        # );
        arg_strs = [
            self.dump_f_method_call_arg(inp, buf_idx_str, shift_constant)
            for inp in op.inputs
        ]
        if not op.pipelined:
            buf_idx_str = ""

        if len(arg_strs) == 0:
            return f"{op.name}_v.template f<{buf_idx_str}>();"
        if isinstance(op, ForLoopNvOp) and op.blk_idx_mapping is not None:
            return f"{op.name}_v.template f<{buf_idx_str}>();"  # 映射到blk idx的for循环本质上只会迭代一次，无需lr

        return (
            f"{op.name}_v.template f<{buf_idx_str}>(\n"
            + self._indent_lines(",\n".join(arg_strs))
            + "\n);"
        )

    # TODO: 调整commit和wait num的发射逻辑：所有以SMEM为输出的Op都添加commit；wait num每个op判断wait num
    # TODO: short pipe 延长到sum shifts + max shift(现在是sum shifts)
    # TODO: 依赖的种类
    # 输入是SMEM
    # 输出是SMEM,但是这个SMEM的owner是其他op(origin带来的依赖)
    # depends_on(op1, op2) -> bool:
    def dump_short_parallel(self, pipeline: NvOpPipeline) -> str:
        stages = pipeline.stages
        shifts = pipeline.shifts

        #! deprecated
        def get_wait_num_old(
            op: NvOp, i: int, pipeline_records: List[List[Tuple[NvOp, int]]]
        ) -> int:
            wait_num = 0
            find_flag = False
            for list_op in reversed(pipeline_records):
                for inp in op.inputs:
                    # print("test source", inp.source, "source name = ", inp.source[0].name)
                    if not isinstance(inp.tensor.source, NvOpOutput):
                        # if not isinstance(inp.source, tuple):
                        continue
                    assert isinstance(inp.tensor.source.op, NvOp)

                    for item in list_op:
                        # print("test item = ", item)
                        # if inp.source[0].name == item[0].name and i == item[1]:
                        if inp.tensor.source.op.name == item[0].name and i == item[1]:
                            find_flag = True
                            break
                    if find_flag:
                        break
                if not find_flag:
                    wait_num += 1
                else:
                    return wait_num
            if not find_flag:
                return -1

        def depends_on(current_op: NvOp, depend_op: NvOp) -> bool:
            for inp in current_op.inputs:
                if not isinstance(inp.tensor.source, NvOpOutput):
                    continue
                assert isinstance(inp.tensor.source.op, NvOp)
                # print("debug inp tensor source = ", inp.tensor.source)
                if inp.tensor.source.op.name == depend_op.name:
                    return True
            return False

        def has_cp_async_smem_input(current_op: NvOp) -> bool:
            for inp in current_op.inputs:
                if inp.tensor.mem == "smem" and inp.tensor.source.op.attrs.get("cp_async", True):
                    return True
            return False

        def get_wait_num(
            current_op: NvOp, i: int, pipeline_records: List[Tuple[NvOp, int]]
        ) -> int:
            wait_num = 0
            if not has_cp_async_smem_input(current_op):
                return -1
            for item in reversed(pipeline_records):
                if i == item[1] and depends_on(current_op, item[0]):
                    return wait_num
                else:
                    wait_num += 1
            return -1

        def launch_commit(current_op: NvOp) -> bool:
            # 输出是否有SMEM的东西
            for opt in current_op.outputs:
                if opt.tensor.mem == "smem" and opt.attrs.get("cp_async", True):
                    return True
            return False

        def dump_short_parallel_i(i, K_total, pipeline_records) -> str:
            output_str = f"// i = {i}\n"
            op_list: List[Tuple[NvOp, int]] = []  # {op, "i"}
            stage_num = len(stages)
            nbuf_num = max(shifts) + 1
            logger.debug("=" * 20 + f"K_total = {K_total}, i = {i}")
            # > 1. 确定op list
            for stage_i in range(stage_num):
                shift_val = sum(
                    shifts[:stage_i]
                )  # sum(shifts[:stage_i + 1]) 从shift0+shift1+shift2+...+shift_i
                logger.debug(f"stage_i = {stage_i}, shift_val = {shift_val}")
                if 0 <= i - shift_val < K_total:
                    for op in stages[stage_i].ops:
                        op_list.append((op, i - shift_val))
            # for item in op_list:
            #     print(f"{{{item[0].name}, {item[1]}}}, ")
            # > 顺序发射op list。每个op: 1. 判断依赖关系，决定是否发射wait; 2. 发射op; 3. 判断是否需要发射commit
            for item in op_list:
                # > i. 判断依赖关系，决定是否发射wait
                wait_num = get_wait_num(item[0], item[1], pipeline_records)
                if wait_num != -1:
                    output_str += (
                        f"__pipeline_wait_prior({wait_num});\n__syncthreads();\n"
                    )
                # > ii. 发射op
                buf_idx_str = str(item[1] % nbuf_num)
                shift_constant = str(i)
                output_str += (
                    self.dump_f_method_call(item[0], buf_idx_str, shift_constant) + "\n"
                )
                # > iii. 判断是否发射commit
                if launch_commit(item[0]):
                    output_str += "__pipeline_commit();\n"
                    pipeline_records.append(item)

            # # > 2. op list 排序，g2s的op在前
            # # return output_str
            # op_list_g2s: List[Tuple[NvOp, int]] = []
            # op_list_other: List[Tuple[NvOp, int]] = []
            # for item in op_list:
            #     if item[0].mem_type == ("g", "s"):
            #         op_list_g2s.append(item)
            #     else:
            #         op_list_other.append(item)
            # logger.debug("op_list_g2s:")
            # for item in op_list_g2s:
            #     logger.debug(f"\t{{{item[0].name}, {item[1]}}}, ")
            # logger.debug("op_list_other:")
            # for item in op_list_other:
            #     logger.debug(f"\t{{{item[0].name}, {item[1]}}}, ")
            # # return output_str

            # # > 生成g2s操作
            # for item in op_list_g2s:
            #     buf_idx_str = str(item[1] % nbuf_num)
            #     shift_constant = str(i)
            #     output_str += (
            #         self.dump_f_method_call(item[0], buf_idx_str, shift_constant) + "\n"
            #     )

            # # > 3. 所有g2s操作做完之后整体添加一个commit
            # if len(op_list_g2s) != 0:
            #     output_str += "__pipeline_commit();\n"
            #     pipeline_records.append(op_list_g2s)
            # # return output_str
            # # > 4. 生成所有其他op
            # for item in op_list_other:
            #     # > 如果是从SMEM到其他位置的op，需要插入一条wait prior指令
            #     if item[0].mem_type[0] == "s":
            #         wait_num = get_wait_num(item[0], item[1], pipeline_records)
            #         if wait_num != -1:
            #             output_str += (
            #                 f"__pipeline_wait_prior({wait_num});\n__syncthreads();\n"
            #             )
            #     buf_idx_str = str(item[1] % nbuf_num)
            #     shift_constant = str(i)
            #     output_str += (
            #         self.dump_f_method_call(item[0], buf_idx_str, shift_constant) + "\n"
            #     )

            return output_str

        def dump_short_parallel_K_total(K_total) -> str:
            pipeline_records = []
            output_strs = []
            for i in range(K_total + sum(shifts)):
                output_strs.append(dump_short_parallel_i(i, K_total, pipeline_records))
            return "\n\n".join(output_strs)

        total_stages = sum(shifts) + max(shifts)

        output_str = ""  #! 函数头参数似乎时固定的？(pipeline_switch + pipeline)
        for K_total in range(total_stages + 1):
            if K_total != 0:
                output_str += "else "
            output_str += f"if (k == {K_total}) {{\n"
            output_str += (
                self._indent_lines(dump_short_parallel_K_total(K_total)) + "\n"
            )
            output_str += "}\n"
            logger.debug("\n" * 2)
        return output_str

    def dump_get_output_method(self, op: NvOp) -> str:
        def dump_output_with_buf_idx_switch(idx: int) -> str:
            if len(op.outputs[idx].tensor.shape) == 0:
                coord_str = "_, buf_idx"
            else:
                coord_str = "_, " * len(op.outputs[idx].tensor.shape) + "buf_idx"
            return (
                f"if constexpr (output_idx == {idx}) {{\n"
                + self._indent_lines(f"return {op.outputs[idx].name}({coord_str});")
                + "\n"
                + "}"
            )

        def dump_output_without_buf_idx_switch(idx: int) -> str:
            return (
                f"if constexpr (output_idx == {idx}) {{\n"
                + self._indent_lines(f"return {op.outputs[idx].name};")
                + "\n"
                + "}"
            )

        output_with_buf_idx_str = (
            "template<size_t output_idx, size_t buf_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + self._indent_lines(
                " else ".join(
                    (
                        dump_output_with_buf_idx_switch(idx)
                        if not op.outputs[idx].unique
                        else dump_output_without_buf_idx_switch(idx)
                    )
                    for idx in range(len(op.outputs))
                )
            )
            + "\n"
            + "}"
        )

        output_without_buf_idx_str = (
            "template<size_t output_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + self._indent_lines(
                " else ".join(
                    dump_output_without_buf_idx_switch(idx)
                    for idx in range(len(op.outputs))
                )
            )
            + "\n"
            + "}"
        )

        if op.pipelined:
            return output_with_buf_idx_str + "\n\n" + output_without_buf_idx_str
        else:
            return output_without_buf_idx_str

    def dump_decl_make_shape(self, shape: Shape, *, nbuf: bool) -> str:
        shape_strs = []
        for s in shape:
            if isinstance(s, IntShape):
                shape_strs.append(f"Int<{s.value}>{{}}")
            elif isinstance(s, ParamShape):
                shape_strs.append(f"{s.param}{{}}")
            elif isinstance(s, VarlenShape):
                shape_strs.append("VARLEN")
            elif isinstance(s, MnkShape):
                shape_strs.append(
                    f"get<{0 if s.mnk == 'm' else 1 if s.mnk == 'n' else 2}>({s.param}{{}})"
                )
            elif isinstance(s, Shape):
                shape_strs.append(self.dump_decl_make_shape(s, nbuf=False))
            else:
                raise ValueError(f"Unknown shape type: {type(s)}")
        if len(shape_strs) == 0:
            shape_strs.append("_1{}")
        if nbuf:
            shape_strs.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shape_strs)
        return f"make_shape({comma_seperated_shapes})"

    def dump_decl_make_stride(
        self, shape: Shape, *, row_major: bool = False, nbuf: bool
    ) -> str:
        def get_product(shape: ShapeEle) -> str:
            if isinstance(shape, IntShape):
                return f"Int<{shape.value}>{{}}"
            elif isinstance(shape, ParamShape):
                return f"{shape.param}{{}}"
            elif isinstance(shape, VarlenShape):
                return "VARLEN"
            elif isinstance(shape, MnkShape):
                return f"get<{0 if shape.mnk == 'm' else 1 if shape.mnk == 'n' else 2}>({shape.param}{{}})"
            elif isinstance(shape, Shape):
                product = "_1{}"
                for s in shape:
                    product = f"{product} * {get_product(s)}"
                return product
            else:
                raise ValueError(f"Unknown shape type: {type(shape)}")

        def get_make_stride_strs(shape: Shape, products: Sequence[str], row_major: bool = False) -> List[str]:
            products = list(products)
            stride_strs = []
            if not row_major:
                for s in shape:
                    if isinstance(s, IntShape) and s.value == 1:
                        stride_strs.append("_0{}")
                        continue
                    if not isinstance(s, Shape):
                        stride_strs.append(" * ".join(products) if products else "_1{}")
                    else: # Shape (list)
                        stride_strs.append(f"make_stride({', '.join(get_make_stride_strs(s, products, row_major))})")
                    products.append(get_product(s))
            else:
                for s in reversed(shape):
                    if isinstance(s, IntShape) and s.value == 1:
                        stride_strs.insert(0, "_0{}")
                        continue
                    if not isinstance(s, Shape):
                        stride_strs.insert(0, " * ".join(products) if products else "_1{}")
                    else: # Shape (list)
                        stride_strs.insert(0, f"make_stride({', '.join(get_make_stride_strs(s, products, row_major))})")
                    products.append(get_product(s))
            return stride_strs

        stride_strs = get_make_stride_strs(shape, [], row_major)
        if len(stride_strs) == 0:
            stride_strs.append("_0{}")
        if nbuf:
            stride_strs.append(" * ".join(get_product(s) for s in shape) if shape else "_1{}")
        return f"make_stride({', '.join(stride_strs)})"
                          

    def dump_input_decl(self, inp: NvOpInput) -> str:
        def dump_decl_input_shape_type() -> str:
            make_shape_str = self.dump_decl_make_shape(inp.tensor.shape, nbuf=False)
            decl_shape_str = f"decltype({make_shape_str})"
            return f"using Shape_i{inp.idx} = {decl_shape_str};"

        def dump_decl_input_shape() -> str:
            return f"Shape_i{inp.idx} shape_i{inp.idx};"

        def dump_decl_input_tensor() -> str:
            return f"// Tensor_i{inp.idx} {inp.name};"

        if len(inp.tensor.shape) > 0:
            shape_type_str = dump_decl_input_shape_type()
            shape_str = dump_decl_input_shape()
            tensor_str = dump_decl_input_tensor()
            return f"{shape_type_str}\n{shape_str}\n{tensor_str}"
        else:
            return f"// TI{inp.idx} {inp.name};"

    def dump_output_decl(self, out: NvOpOutput) -> str:
        def dump_decl_output_shape_type() -> str:
            assert out.op is not None
            if not out.op.pipelined or out.unique:
                make_shape_str = self.dump_decl_make_shape(out.tensor.shape, nbuf=False)
            else:
                make_shape_str = self.dump_decl_make_shape(out.tensor.shape, nbuf=True)
            decl_shape_str = f"decltype({make_shape_str})"
            return f"using Shape_o{out.idx} = {decl_shape_str};"

        def dump_decl_output_layout_type() -> str:
            assert out.op is not None
            if not out.op.pipelined or out.unique:
                make_stride_str = self.dump_decl_make_stride(
                    out.tensor.shape, row_major=out.tensor.row_major, nbuf=False
                )
            else:
                make_stride_str = self.dump_decl_make_stride(
                    out.tensor.shape, row_major=out.tensor.row_major, nbuf=True
                )

            using_stride_str = f"using Stride_o{out.idx} = decltype({make_stride_str});"
            make_layout_str = (
                f"make_layout(Shape_o{out.idx}{{}}, Stride_o{out.idx}{{}})"
            )

            if (swizzle := out.tensor.swizzle) is not None:
                using_layout_str = f"using Layout_o{out.idx} = decltype(composition(Swizzle<{swizzle.b}, {swizzle.m}, {swizzle.s}>{{}}, {make_layout_str}));"
            else:
                using_layout_str = (
                    f"using Layout_o{out.idx} = decltype({make_layout_str});"
                )

            return f"{using_stride_str}\n{using_layout_str}"

        def dump_decl_output_tensor_type() -> str:
            match out.tensor.mem:
                case "gmem":
                    decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr((TO{out.idx}*)nullptr), Layout_o{out.idx}{{}}))"
                case "smem":
                    decl_tensor_str = f"decltype(make_tensor(make_smem_ptr((TO{out.idx}*)nullptr), Layout_o{out.idx}{{}}))"
                case "rmem":
                    decl_tensor_str = f"decltype(make_tensor(make_rmem_ptr((TO{out.idx}*)nullptr), Layout_o{out.idx}{{}}))"
            return f"using Tensor_o{out.idx} = {decl_tensor_str};"

        def dump_decl_output_shape() -> str:
            return f"Shape_o{out.idx} shape_o{out.idx};"

        def dump_decl_output_layout() -> str:
            return f"Layout_o{out.idx} layout_o{out.idx};"

        def dump_decl_output_tensor() -> str:
            return f"Tensor_o{out.idx} {out.name};"

        if out.owning:
            shape_type_str = dump_decl_output_shape_type()
            shape_str = dump_decl_output_shape()
            layout_type_str = dump_decl_output_layout_type()
            layout_str = dump_decl_output_layout()
            tensor_type_str = dump_decl_output_tensor_type()
            tensor_str = dump_decl_output_tensor()
            return f"{shape_type_str}\n{layout_type_str}\n{tensor_type_str}\n{shape_str}\n{layout_str}\n{tensor_str}"
        else:
            shape_type_str = dump_decl_output_shape_type()
            shape_str = dump_decl_output_shape()
            tensor_str = dump_decl_output_tensor()
            return f"{shape_type_str}\n{shape_str}\n{tensor_str}"

    def dump_init_input_varlen_shape(self, op: NvOp) -> List[str]:
        init_strs = []
        for idx, inp in enumerate(op.inputs):
            make_shape_str, varlen_idx = self.dump_init_make_shape(
                idx, inp.tensor.shape, is_output=False
            )
            if varlen_idx > 0:
                init_strs.append(f"shape_i{idx}({make_shape_str})")
        return init_strs

    def dump_init_output_varlen_shape(self, op: NvOp) -> List[str]:
        init_strs = []
        for idx, out in enumerate(op.outputs):
            make_shape_str, varlen_idx = self.dump_init_make_shape(
                idx, out.tensor.shape, is_output=True
            )
            if varlen_idx > 0:
                init_strs.append(f"shape_o{idx}({make_shape_str})")
        return init_strs

    def dump_init_make_shape(
        self, idx: int, shape: Shape, *, is_output: bool
    ) -> Tuple[str, int]:
        shape_strs = []
        varlen_idx: int = 0
        for i, s in enumerate(shape):
            if isinstance(s, IntShape):
                shape_strs.append(f"Int<{s.value}>{{}}")
            elif isinstance(s, ParamShape):
                shape_strs.append(f"{s.param}{{}}")
            elif isinstance(s, VarlenShape):
                shape_strs.append(f"mode_{'o' if is_output else 'i'}{idx}_{varlen_idx}")
                varlen_idx += 1
            elif isinstance(s, MnkShape):
                shape_strs.append(
                    f"get<{0 if s.mnk == 'm' else 1 if s.mnk == 'n' else 2}>({s.param}{{}})"
                )
            elif isinstance(s, Shape):
                shape_strs.append(self.dump_decl_make_shape(s, nbuf=False))
            else:
                raise ValueError(f"Unknown shape type: {type(s)}")
        if is_output:
            shape_strs.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shape_strs)
        return f"make_shape({comma_seperated_shapes})", varlen_idx

    def dump_non_owning_tensor_type_template_arg_strs(self, op: NvOp) -> List[str]:
        non_owning_tensor_type_template_arg_strs = []
        for out in op.outputs:
            if not out.owning:
                assert isinstance(out.tensor.source, NvOpOutput)
                origin = out.tensor.source.origin
                assert origin is not None
                assert origin.op is not None
                origin_out_str = f"{origin.op.name}_v.template output<{origin.idx}>()"
                non_owning_tensor_type_template_arg_strs.append(
                    f"std::decay_t<decltype({origin_out_str})>"
                )
        return non_owning_tensor_type_template_arg_strs

    def dump_non_owning_tensor_as_arg_strs(self, op: NvOp) -> List[str]:
        non_owning_tensor_as_arg_strs = []
        for out in op.outputs:
            if not out.owning:
                assert isinstance(out.tensor.source, NvOpOutput)
                origin = out.tensor.source.origin
                assert origin is not None
                assert origin.op is not None
                origin_out_str = f"{origin.op.name}_v.template output<{origin.idx}>()"
                non_owning_tensor_as_arg_strs.append(origin_out_str)
        return non_owning_tensor_as_arg_strs

    def dump_nvop_create(self, op: NvOp, nbuf: int) -> str:
        rmem_outputs = [out for out in op.outputs if out.tensor.mem == "rmem"]
        smem_outputs = [out for out in op.outputs if out.tensor.mem == "smem"]
        gmem_outputs = [out for out in op.outputs if out.tensor.mem == "gmem"]

        template_args = []
        if op.pipelined:
            template_args.append(f"Int<{nbuf}>")
        template_args.extend(["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"])
        # template_args.extend([parameter for parameter in op.parameters])
        template_args.extend(self.dump_non_owning_tensor_type_template_arg_strs(op))

        template_args_str = ", ".join(template_args)
        using_type_str = f"using {op.name}_t = {op.name}<{template_args_str}>;"

        smem_output_shapes = [
            f"using {op.name}_layout_o{out.idx} = typename {op.name}_t::Layout_o{out.idx};"
            for out in smem_outputs
            if out.owning
        ]
        smem_output_array_defs = [
            f"__shared__ {out.tensor.dtype} {op.name}_tensor_o{out.idx}[cosize_v<{op.name}_layout_o{out.idx}>];"
            for out in smem_outputs
            if out.owning
        ]

        rmem_output_shapes = [
            f"using {op.name}_layout_o{out.idx} = typename {op.name}_t::Layout_o{out.idx};"
            for out in rmem_outputs
            if out.owning
        ]
        rmem_output_array_defs = [
            f"{out.tensor.dtype} {op.name}_tensor_o{out.idx}[cosize_v<{op.name}_layout_o{out.idx}>];"
            for out in rmem_outputs
            if out.owning
        ]

        outputs = []
        for out in op.outputs:
            if out.owning:
                outputs.append(f"{op.name}_tensor_o{out.idx}")
            else:
                if isinstance(out.tensor.source, NvOpOutput):
                    origin = out.tensor.source.origin
                    assert origin is not None
                    assert origin.op is not None
                    outputs.append(
                        f"{origin.op.name}_v.template output<{origin.idx}>()"
                    )
                else:
                    outputs.append(out.tensor.source)

        varlen_modes = []
        for io in chain(op.inputs, op.outputs):
            for s in io.tensor.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes.append(s.name)

        args_0 = ["tid", "lid", "wid"]
        args_1 = outputs
        args_2 = varlen_modes
        if args_0 and args_1 and args_2:
            args_str = (
                ", ".join(args_0)
                + ", \n"
                + ", ".join(args_1)
                + ", \n"
                + ", ".join(args_2)
                + ","
            )
        elif args_0 and args_1:
            args_str = ", ".join(args_0) + ", \n" + ", ".join(args_1) + ","
        elif args_0 and args_2:
            args_str = ", ".join(args_0) + ", \n" + ", ".join(args_2) + ","
        else:
            args_str = ", ".join(args_0) + ","
        op_var_def_str = (
            f"{op.name}_t {op.name}_v{{\n" + self._indent_lines(args_str) + "\n" + "};"
        )

        return (
            using_type_str
            + "\n"
            + (("\n".join(smem_output_shapes) + "\n") if smem_output_shapes else "")
            + (
                ("\n".join(smem_output_array_defs) + "\n")
                if smem_output_array_defs
                else ""
            )
            + (("\n".join(rmem_output_shapes) + "\n") if rmem_output_shapes else "")
            + (
                ("\n".join(rmem_output_array_defs) + "\n")
                if rmem_output_array_defs
                else ""
            )
            + op_var_def_str
        )

    def dump_gmem_tensor_nvop_create(
        self,
        op: GmemTensorNvOp,
    ) -> str:
        op_name = op.name
        tensor_name = op.gmem_inout.name
        varlen_modes_constructor_args = [
            s.name for s in op.gmem_inout.shape if isinstance(s, VarlenShape)
        ]
        varlen_modes_constructor_args_str = "".join(
            ", " + arg for arg in varlen_modes_constructor_args
        )

        return templates.GMEM_TENSOR_NVOP_CREATE_SKELETON.format(
            op_name=op_name,
            tensor_name=tensor_name,
            varlen_modes_constructor_args=varlen_modes_constructor_args_str,
        )

    def dump_constant_nvop_create(
        self,
        op: ConstantNvOp,
    ) -> str:
        op_name = op.name
        dtype = op.outputs[0].tensor.dtype

        return templates.CONSTANT_NVOP_CREATE_SKELETON.format(
            op_name=op_name,
            dtype=dtype,
        )

    def dump_for_loop_nvop_create(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        op_name = op.name
        visible_op_type_template_args_strs = [f"{vop.name}_t" for vop in visible_ops]
        visible_op_as_args_strs = [f"{vop.name}_v" for vop in visible_ops]
        visible_op_type_template_args_str = "".join(
            f", {arg}" for arg in visible_op_type_template_args_strs
        )
        visible_op_as_args_str = "".join(f", {arg}" for arg in visible_op_as_args_strs)

        non_owning_tensor_type_template_arg_strs = (
            self.dump_non_owning_tensor_type_template_arg_strs(op)
        )
        non_owning_tensor_type_template_args_str = "".join(
            f", {arg_str}" for arg_str in non_owning_tensor_type_template_arg_strs
        )

        non_owning_tensor_as_arg_strs = self.dump_non_owning_tensor_as_arg_strs(op)
        non_owning_tensor_as_args_str = "".join(
            f", {arg}" for arg in non_owning_tensor_as_arg_strs
        )

        return templates.FOR_LOOP_NVOP_CREATE_SKELETON.format(
            op_name=op_name,
            visible_op_type_template_args=visible_op_type_template_args_str,
            non_owning_tensor_type_template_args=non_owning_tensor_type_template_args_str,
            visible_op_as_args=visible_op_as_args_str,
            non_owning_tensor_as_args=non_owning_tensor_as_args_str,
        )

    def dump_pipelined_for_loop_nvop_create(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        return self.dump_for_loop_nvop_create(op, visible_ops)

    def dump_sequential_for_loop_nvop_create(
        self, op: ForLoopNvOp, visible_ops: Sequence[NvOp]
    ) -> str:
        return self.dump_for_loop_nvop_create(op, visible_ops)

    def dump_pipelined_for_loop_nvop_methods(self, op: ForLoopNvOp) -> str:
        assert isinstance(op.body, NvOpPipeline)
        pipeline = op.body

        op_stage_idx_mapping = {
            op: stage_idx
            for (stage_idx, stage) in enumerate(pipeline.stages)
            for op in stage.ops
        }

        @dataclass
        class Inst: ...

        @dataclass
        class OpCall(Inst):
            op: NvOp
            buf_idx: int
            shift_constant: str

        @dataclass
        class PipelineWait(Inst):
            wait_num: int

        @dataclass
        class PipelineCommit(Inst): ...

        @dataclass
        class Comment(Inst):
            comment: str

            def __str__(self) -> str:
                return f"// {self.comment}"

        @dataclass
        class SyncThreads(Inst): ...

        def smem_depends_on(current_op: NvOp, depend_op: NvOp) -> bool:
            for inp in current_op.inputs:
                if inp.tensor.mem != "smem":
                    continue
                if not isinstance(inp.tensor.source, NvOpOutput):
                    continue
                assert isinstance(inp.tensor.source.op, NvOp)
                if inp.tensor.source.op.name == depend_op.name:
                    return True
            return False

        def get_wait_num(
            current_op: NvOp, i: int, pipeline_records: List[Tuple[NvOp, int]]
        ) -> int:
            wait_num = 0
            for item in reversed(pipeline_records):
                if not has_cp_async_smem_output(item[0]):
                    continue
                if i == item[1] and smem_depends_on(current_op, item[0]):
                    return wait_num
                else:
                    wait_num += 1
            return -1

        def get_full_pipeline_history(
            stage_buf_offset_global: int,
        ) -> List[Tuple[NvOp, int]]:
            # 每个stage的buf_idx偏移量，假设最后一个stage的偏移量为0 + stage_buf_offset_global
            # 前面各个stage的偏移量由于进行的更提前，所以依次增加
            stage_buf_offsets = [
                sum(pipeline.shifts[i:]) + stage_buf_offset_global
                for i in range(len(pipeline.stages))
            ]
            history = []

            # exactly pipeline.max_shift步
            for step in range(pipeline.max_shift):
                for stage_idx, stage in enumerate(pipeline.stages):
                    stage_buf_idx = (
                        stage_buf_offsets[stage_idx] + step
                    ) % pipeline.nbuf
                    for op in stage.ops:
                        history.append((op, stage_buf_idx))
            return history

        def has_cp_async_smem_output(current_op: NvOp) -> bool:
            # 输出是否有SMEM的东西
            for opt in current_op.outputs:
                if opt.tensor.mem == "smem" and opt.attrs.get("cp_async", True):
                    return True
            return False

        def has_cp_async_smem_input(current_op: NvOp) -> bool:
            for inp in current_op.inputs:
                if inp.tensor.mem == "smem" and inp.tensor.source.op.attrs.get(
                    "cp_async", True
                ):
                    return True
            return False

        def default_schedule(
            op_calls: List[OpCall], pipeline_history: List[Tuple[NvOp, int]]
        ) -> List[Inst]:
            instructions = []

            for op_call in op_calls:
                if has_cp_async_smem_input(op_call.op):
                    wait_num = get_wait_num(
                        op_call.op, op_call.buf_idx, pipeline_history
                    )
                    if wait_num != -1:
                        instructions.append(PipelineWait(wait_num=wait_num))
                        instructions.append(SyncThreads())

                instructions.append(
                    Comment(
                        f"op_call |buf_idx={op_call.buf_idx}| {op_call.shift_constant} {op_call.op.name} "
                    )
                )
                instructions.append(op_call)

                if has_cp_async_smem_output(op_call.op):
                    instructions.append(PipelineCommit())

                pipeline_history.append((op_call.op, op_call.buf_idx))

            return instructions

        def dump_insts(insts: List[Inst]) -> str:
            inst_strs = []
            for inst in insts:
                if isinstance(inst, OpCall):
                    inst_strs.append(
                        self.dump_f_method_call(
                            inst.op,
                            buf_idx_str=str(inst.buf_idx),
                            shift_constant=inst.shift_constant,
                        )
                    )
                elif isinstance(inst, PipelineWait):
                    inst_strs.append(f"__pipeline_wait_prior({inst.wait_num});")
                elif isinstance(inst, PipelineCommit):
                    inst_strs.append("__pipeline_commit();")
                elif isinstance(inst, Comment):
                    inst_strs.append(str(inst))
                elif isinstance(inst, SyncThreads):
                    inst_strs.append("__syncthreads();")
                else:
                    raise ValueError(f"Unknown instruction: {inst}")
            return "\n".join(inst_strs)

        def dump_short_pipe_device_function() -> str:
            short_pipe_body = self.dump_short_parallel(pipeline)
            return templates.SHORT_PIPE_SKELETON.format(
                short_pipe_body=self._indent_lines(short_pipe_body),
            )

        def dump_fill() -> str:
            def dump_fill_body() -> str:
                # All stages start from 0
                stage_counter = [0 for _ in range(len(pipeline.stages))]

                # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
                # The first stage has offset 0
                stage_idx_offsets = [
                    -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
                ]

                pipeline_history = []
                all_insts = []
                total_step = 0
                for i in range(len(pipeline.stages)):
                    all_insts.append(Comment(f"stage 0..{i}"))
                    # 最后一个
                    nsteps = (
                        pipeline.shifts[i]
                        if i < len(pipeline.shifts)
                        else pipeline.max_shift
                    )
                    for _ in range(nsteps):
                        # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 1
                        op_calls = []  # [op0, op1, op2, ...]
                        for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
                            for op in stage.ops:
                                op_calls.append(
                                    OpCall(
                                        op,
                                        buf_idx=stage_counter[stage_idx]
                                        % pipeline.nbuf,
                                        shift_constant=f"{total_step + stage_idx_offsets[stage_idx]}",
                                    )
                                )
                            stage_counter[stage_idx] += 1
                        total_step += 1
                        insts = default_schedule(op_calls, pipeline_history)
                        all_insts.extend(insts)
                return dump_insts(all_insts)

            return templates.FILL_SKELETON.format(
                fill_body=self._indent_lines(dump_fill_body()),
            )

        def dump_loop_step() -> str:
            def dump_loop_step_body() -> str:
                # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
                # The last stage has buf_offset max_shift due to 如Full
                stage_buf_offsets = [
                    sum(pipeline.shifts[i:]) + pipeline.max_shift
                    for i in range(len(pipeline.stages))
                ]

                # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
                # The first stage has offset 0
                stage_idx_offsets = [
                    -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
                ]

                pipeline_history = get_full_pipeline_history(0)
                all_insts = []
                for step in range(pipeline.nbuf):
                    all_insts.append(Comment(f"step {step}"))
                    op_calls = []
                    for stage_idx, stage in enumerate(pipeline.stages):
                        for op in stage.ops:
                            op_calls.append(
                                OpCall(
                                    op,
                                    buf_idx=(step + stage_buf_offsets[stage_idx])
                                    % pipeline.nbuf,
                                    shift_constant=f"i + ({step + stage_idx_offsets[stage_idx]})",
                                )
                            )
                    insts = default_schedule(op_calls, pipeline_history)
                    all_insts.extend(insts)

                return dump_insts(all_insts)

            return templates.LOOP_STEP_SKELETON.format(
                loop_step_body=self._indent_lines(dump_loop_step_body()),
            )

        def dump_remainder() -> str:
            def dump_remains_r(r: int) -> str:
                def dump_remains_r_body() -> str:
                    # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
                    # The last stage has buf_offset 0
                    stage_buf_offsets = [
                        sum(pipeline.shifts[i:]) + pipeline.max_shift
                        for i in range(len(pipeline.stages))
                    ]

                    # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
                    # The first stage has offset 0
                    stage_idx_offsets = [
                        -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
                    ]

                    pipeline_history = get_full_pipeline_history(0)
                    all_insts = []
                    for step in range(r):
                        all_insts.append(Comment(f"total remains {r}, current {step}"))
                        op_calls = []
                        for stage_idx, stage in enumerate(pipeline.stages):
                            for op in stage.ops:
                                op_calls.append(
                                    OpCall(
                                        op,
                                        buf_idx=(step + stage_buf_offsets[stage_idx])
                                        % pipeline.nbuf,
                                        shift_constant=f"i + ({step + stage_idx_offsets[stage_idx]})",
                                    )
                                )
                        insts = default_schedule(op_calls, pipeline_history)
                        all_insts.extend(insts)

                    return dump_insts(all_insts)

                return templates.REMAINS_R_SKELETON.format(
                    r=r,
                    remains_r_body=self._indent_lines(dump_remains_r_body()),
                )

            def dump_remainder_dispatch() -> str:
                def dump_remainder_dispatch_body() -> str:
                    def dump_remains_i_call(i: int) -> str:
                        return f"remains_{i}(i, l, r);"

                    def dump_remains_i_switch_i(i: int) -> str:
                        return ("else " if i > 1 else "") + (
                            f"if (r - l - i == {i}) {{\n"
                            + self._indent_lines(dump_remains_i_call(i))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(
                        dump_remains_i_switch_i(i) for i in range(1, pipeline.nbuf)
                    )

                return templates.REMAINDER_SKELETON.format(
                    remainder_dispatch_body=self._indent_lines(
                        dump_remainder_dispatch_body()
                    ),
                )

            return (
                "\n\n".join(dump_remains_r(i) for i in range(1, pipeline.nbuf))
                + "\n\n"
                + dump_remainder_dispatch()
            )

        def dump_empties() -> str:
            def dump_empty_after_remain_r(r: int) -> str:
                def dump_empty_after_remain_r_body() -> str:
                    # Stage 0 is completed
                    # Stage 1, 2, 3, ... starts from [idx - shift0, idx - shift0 - shift1, idx - shift0 - shift1 - shift2, ...]
                    stage_counter = [
                        r + sum(pipeline.shifts[i:]) + pipeline.max_shift
                        for i in range(len(pipeline.stages))
                    ]

                    stage_idx_offsets = [
                        r - sum(pipeline.shifts[:i])
                        for i in range(len(pipeline.stages))
                    ]

                    pipeline_history = get_full_pipeline_history(r)
                    all_insts = []
                    total_step = 0
                    for i in range(1, len(pipeline.stages)):
                        all_insts.append(
                            Comment(f"stage {i}..{len(pipeline.stages) - 1}")
                        )
                        nsteps = pipeline.shifts[i - 1]
                        for _ in range(nsteps):
                            # Stage 1, 2, ... nstages - 1; Stage 2, 3, ... nstages - 1; ...; Stage nstages - 1
                            op_calls = []
                            for stage_idx, stage in enumerate(
                                pipeline.stages[i:], start=i
                            ):
                                for op in stage.ops:
                                    op_calls.append(
                                        OpCall(
                                            op,
                                            buf_idx=stage_counter[stage_idx]
                                            % pipeline.nbuf,
                                            shift_constant=f"i + ({total_step + stage_idx_offsets[stage_idx]})",
                                        )
                                    )
                                stage_counter[stage_idx] += 1
                            total_step += 1
                            insts = default_schedule(op_calls, pipeline_history)
                            all_insts.extend(insts)
                    return dump_insts(all_insts)

                return templates.EMPTY_AFTER_REMAIN_R_SKELETON.format(
                    r=r,
                    empty_after_remain_r_body=self._indent_lines(
                        dump_empty_after_remain_r_body()
                    ),
                )

            def dump_empty_after_remain_r_dispatch() -> str:
                def dump_empty_dispatch_body() -> str:
                    def dump_empty_after_remain_r_call(r: int) -> str:
                        return f"empty_after_remain_{r}(i, l, r);"

                    def dump_empty_after_remain_r_switch_r(r: int) -> str:
                        return ("else " if r > 0 else "") + (
                            f"if (r - l - i == {r}) {{\n"
                            + self._indent_lines(dump_empty_after_remain_r_call(r))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(
                        dump_empty_after_remain_r_switch_r(i)
                        for i in range(pipeline.nbuf)
                    )

                return templates.EMPTY_SKELETON.format(
                    empty_dispatch_body=self._indent_lines(dump_empty_dispatch_body()),
                )

            return (
                "\n\n".join(dump_empty_after_remain_r(i) for i in range(pipeline.nbuf))
                + "\n\n"
                + dump_empty_after_remain_r_dispatch()
            )

        def dump_dispatch() -> str:
            fill_len = sum(pipeline.shifts) + pipeline.max_shift

            def dump_short_pipe_dispatch() -> str:
                return f"short_pipe(k, l, r);"

            def dump_fill_dispatch() -> str:
                return f"fill(l, r);"

            def dump_loop_step_dispatch() -> str:
                return f"loop_step(i, l, r);"

            def dump_remainder_dispatch() -> str:
                return f"remainder(i, l, r);"

            def dump_empty_dispatch() -> str:
                return f"empty(i, l, r);"

            return templates.PIPELINE_DISPATCH_SKELETON.format(
                fill_len=fill_len,
                nbuf=pipeline.nbuf,
                short_pipe_dispatch=dump_short_pipe_dispatch(),
                fill_dispatch=dump_fill_dispatch(),
                loop_step_dispatch=dump_loop_step_dispatch(),
                remainder_dispatch=dump_remainder_dispatch(),
                empty_dispatch=dump_empty_dispatch(),
            )

        return (
            "// short pipe\n"
            + dump_short_pipe_device_function()
            + "\n\n// fill\n"
            + dump_fill()
            + "\n\n// loop_step\n"
            + dump_loop_step()
            + "\n\n// remainder\n"
            + dump_remainder()
            + "\n\n// empties\n"
            + dump_empties()
            + "\n\n// dispatch\n"
            + dump_dispatch()
        )

    def dump_nvop_class_defs(self, program: NvOpProgram) -> str:
        class_defs = []
        visible_ops = {}

        def _dump_sequential_for_loop_nvop(
            for_loop_op: ForLoopNvOp, current_level: int
        ) -> None:
            for op in for_loop_op.body.ops:
                _dump_nvop(op, current_level + 1)
            class_defs.append(
                self.dump_sequential_for_loop_nvop_class_def(
                    for_loop_op,
                    [
                        op
                        for l in range(current_level + 2)
                        for op in visible_ops.setdefault(l, [])
                    ],
                )
            )
            del visible_ops[current_level + 1]

        def _dump_pipelined_for_loop_nvop(
            for_loop_op: ForLoopNvOp, current_level: int
        ) -> None:
            assert isinstance(for_loop_op.body, NvOpPipeline)
            for stage in for_loop_op.body.stages:
                for op in stage.ops:
                    _dump_nvop(op, current_level + 1)
            class_defs.append(
                self.dump_pipelined_for_loop_nvop_class_def(
                    for_loop_op,
                    [
                        op
                        for l in range(current_level + 2)
                        for op in visible_ops.setdefault(l, [])
                    ],
                )
            )
            del visible_ops[current_level + 1]

        def _dump_nvop(op: NvOp, current_level: int) -> None:
            if isinstance(op, GmemTensorNvOp):
                class_defs.append(self.dump_gmem_tensor_nvop_class_def(op))
            elif isinstance(op, ForLoopNvOp):
                if isinstance(op.body, NvOpPipeline):
                    _dump_pipelined_for_loop_nvop(op, current_level)
                elif isinstance(op.body, NvOpSequence):
                    _dump_sequential_for_loop_nvop(op, current_level)
                else:
                    raise ValueError(f"Unknown body type: {type(op.body)}")
            elif isinstance(op, ConstantNvOp):
                class_defs.append(self.dump_constant_nvop_class_def(op))
            else:
                class_defs.append(self.dump_nvop_class_def(op))
            visible_ops.setdefault(current_level, []).append(op)

        for op in program.ops:
            _dump_nvop(op, 0)

        return "\n\n".join(class_defs)

    def dump_nvop_inits(self, program: NvOpProgram) -> str:
        nvop_inits = []
        visible_ops = {}

        def _dump_sequential_for_loop_nvop(
            for_loop_op: ForLoopNvOp, current_level: int
        ) -> None:
            for op in for_loop_op.body.ops:
                _dump_nvop(op, current_level + 1, nbuf=1)
            nvop_inits.append(
                self.dump_sequential_for_loop_nvop_create(
                    for_loop_op,
                    [
                        op
                        for l in range(current_level + 2)
                        for op in visible_ops.setdefault(l, [])
                    ],
                )
            )
            del visible_ops[current_level + 1]

        def _dump_pipelined_for_loop_nvop(
            for_loop_op: ForLoopNvOp, current_level: int
        ) -> None:
            assert isinstance(for_loop_op.body, NvOpPipeline)
            for stage in for_loop_op.body.stages:
                for op in stage.ops:
                    _dump_nvop(op, current_level + 1, nbuf=for_loop_op.body.nbuf)
            nvop_inits.append(
                self.dump_pipelined_for_loop_nvop_create(
                    for_loop_op,
                    [
                        op
                        for l in range(current_level + 2)
                        for op in visible_ops.setdefault(l, [])
                    ],
                )
            )
            del visible_ops[current_level + 1]

        def _dump_nvop(op: NvOp, current_level: int, nbuf: int) -> None:
            if isinstance(op, GmemTensorNvOp):
                nvop_inits.append(self.dump_gmem_tensor_nvop_create(op))
            elif isinstance(op, ForLoopNvOp):
                if isinstance(op.body, NvOpPipeline):
                    _dump_pipelined_for_loop_nvop(op, current_level)
                elif isinstance(op.body, NvOpSequence):
                    _dump_sequential_for_loop_nvop(op, current_level)
                else:
                    raise ValueError(f"Unknown body type: {type(op.body)}")
            elif isinstance(op, ConstantNvOp):
                nvop_inits.append(self.dump_constant_nvop_create(op))
            else:
                nvop_inits.append(self.dump_nvop_create(op, nbuf))
            visible_ops.setdefault(current_level, []).append(op)

        for op in program.ops:
            _dump_nvop(op, 0, nbuf=1)

        return "\n".join(nvop_inits)

    def dump_nvop_global_function(self, program: NvOpProgram) -> str:

        def dump_all_parameters() -> str:
            all_parameters = sorted(program.all_parameters())
            return "".join([f", class {param}" for param in all_parameters])

        def dump_gmem_ptrs() -> str:
            return ",\n".join(
                # [f"{io.dtype} *d{name}" for name, io in program.gmem_inouts.items()]
                [
                    f"{program.gmem_inouts[name].dtype} *d{name}"
                    for name in sorted(program.gmem_inouts.keys())
                ]
            )

        def dump_varlens() -> str:
            all_varlens = sorted(program.all_varlens())
            return ", ".join([f"int {name}" for name in all_varlens])

        def dump_nvop_calls() -> str:
            return "\n".join(
                [
                    self.dump_f_method_call(op, buf_idx_str="0")
                    for op in program.ops
                    if not isinstance(op, GmemTensorNvOp)
                ]
            )

        all_parameters = dump_all_parameters()
        kernel_name = program.name
        gmem_ptrs = self._indent_lines(dump_gmem_ptrs())
        varlens = self._indent_lines(dump_varlens())
        nvop_inits = self._indent_lines(self.dump_nvop_inits(program))
        nvop_calls = self._indent_lines(dump_nvop_calls())
        return templates.GLOBAL_FUNCTION_SKELETON.format(
            parameter_types=all_parameters,
            kernel_name=kernel_name,
            gmem_ptrs=gmem_ptrs,
            varlens=varlens,
            nvop_inits=nvop_inits,
            nvop_calls=nvop_calls,
        )

    def dump_nvop_program(self, program: NvOpProgram) -> str:
        device_helper_functions = templates.DTC_HELPER_FUNCTIONS
        nvop_defs = self.dump_nvop_class_defs(program)
        global_function = self.dump_nvop_global_function(program)
        return templates.CUDA_SOURCE_SKELETON.format(
            device_helper_functions=device_helper_functions,
            nvop_defs=nvop_defs,
            global_function=global_function,
        )
