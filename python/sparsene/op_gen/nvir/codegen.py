from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sparsene.op_gen.nvir.nvop import (
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
)
from python.sparsene.op_gen.nvir.sequence import NvOpSequence, NvOpStage
import sparsene.op_gen.nvir.templates as templates
from itertools import chain
from sparsene.logging import get_logger

import re

logger = get_logger(__name__)


def convert_name(name: str, to_snake: bool = True) -> str:
    if to_snake:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    else:
        return re.sub(
            r"_([a-z])", lambda match: match.group(1).upper(), name
        ).capitalize()


def to_snake(name: str) -> str:
    return convert_name(name, to_snake=True)


def to_pascal(name: str) -> str:
    return convert_name(name, to_snake=False)


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
                f"class TI{idx} = {inp.dtype}" for idx, inp in enumerate(op.inputs)
            ]
            out_types_strs = [
                f"class TO{idx} = {out.dtype}" for idx, out in enumerate(op.outputs)
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

            return f"template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK{parameter_types_str}{non_owning_tensor_types_str}{io_types_str}>"

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
                    output_constructor_params.append(f"Tensor_o{out.idx} &{out.name}")
                elif out.mem != "rmem":
                    output_constructor_params.append(f"{out.dtype} *{out.name}")
            varlen_modes_constructor_params = []
            for out in op.outputs:
                i: int = 0
                for s in out.shape:
                    if isinstance(s, VarlenShape):
                        varlen_modes_constructor_params.append(
                            f"int mode_o{out.idx}_{i}"
                        )
                        i += 1
            for inp in op.inputs:
                i: int = 0
                for s in inp.shape:
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
            elif out.mem != "rmem":
                output_constructor_init_list.append(
                    f"{out.name}(make_tensor(make_{out.mem}_ptr({out.name}), Layout_o{out.idx}{{}}))"
                )

        constructor_init_list = (
            hw_param_constructor_init_list
            + varlen_shapes_constructor_init_list
            + output_constructor_init_list
        )
        constructor_init_list_str = ", ".join(constructor_init_list)
        constructor_str = f"{constructor_head_str}\n: {constructor_init_list_str} {{}}"

        output_method_str = self.dump_get_output_method(op)

        f_method_str = self.dump_f_method(op)

        return (
            f"{template_head_str}\n{class_head_str}\n"
            f"{self._indent_lines(inputs_str)}\n\n"
            f"{self._indent_lines(outputs_str)}\n\n"
            f"{self._indent_lines(hw_param_str)}\n\n"
            f"{self._indent_lines(constructor_str)}\n\n"
            f"{self._indent_lines(output_method_str)}\n\n"
            f"{self._indent_lines(f_method_str)}\n"
            "};"
        )

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
                dump_static_assert(inp.shape, inp, [])
            # f"CUTE_STATIC_ASSERT_V((get<{i}>(shape_i{inp.idx}) == size<{i}>(shape({inp.name}))));"
            # f"CUTE_STATIC_ASSERT((is_static<decltype(size<{i}>(shape({inp.name})))>::value));"

        dump_static_assert()

        code_str = "\n".join([*static_assert_strs, op.impl.code_template])
        f_template_head_str = (
            "template<size_t buf_idx"
            # + (", size_t wait_num" if op.mem_type[0] == "s" else "")
            + "".join(
                f", class Tensor_i{inp.idx}" for inp in op.inputs if len(inp.shape) > 0
            )
            + ">"
        )
        f_input_list = []
        for inp in op.inputs:
            if len(inp.shape) > 0:
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
        indent_level: int = 0,
    ) -> str:
        # g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>()
        # l + i + shift1 + shift2
        # print(inp.source)
        if isinstance(inp.source, tuple):
            return f"{to_snake(inp.source[0].name)}.template output<{buf_idx_str}, {inp.source[1]}>()"
        else:
            if "{c}" in inp.source:
                if shift_constant is not None:
                    return inp.source.format(c=shift_constant)
                else:
                    raise ValueError("Missing value for 'c'")
            return inp.source

    def dump_f_method_call(
        self,
        op: NvOp,
        buf_idx_str: str,
        shift_constant: Optional[str] = None,
        indent_level: int = 0,
    ) -> str:
        # g2s_coo_atomic_format_load_val_op.template f<(i - shift1) % nbuf_rmem>(val_coo_val,
        #     g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 0>(),
        #     g2r_coo_atomic_format_load_off_op.template output<(i - shift1) % nbuf_rmem, 1>()
        # );
        arg_strs = [
            self.dump_f_method_call_arg(inp, buf_idx_str, shift_constant, indent_level)
            for inp in op.inputs
        ]
        # print(arg_strs)
        # print(op.name, buf_idx_str, shift_constant)
        return (
            f"{to_snake(op.name)}.template f<{buf_idx_str}>(\n"
            + self._indent_lines(",\n".join(arg_strs))  # 使用缩进
            + "\n);"  # 结束模板结构
        )

    def dump_short_parallel(self, stages: List[NvOpStage], shifts: List[int]) -> str:
        def get_wait_num(
            op: NvOp, i: int, pipeline_records: List[List[Tuple[NvOp, int]]]
        ) -> int:
            wait_num = 0
            find_flag = False
            # print("test_pipeline_records = ", pipeline_records)
            for list_op in reversed(pipeline_records):
                # print("test inputs: ", op.inputs)
                for inp in op.inputs:
                    # print("test source", inp.source, "source name = ", inp.source[0].name)
                    if not isinstance(inp.source, tuple):
                        continue

                    for item in list_op:
                        # print("test item = ", item)
                        if inp.source[0].name == item[0].name and i == item[1]:
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
            # > 2. op list 排序，g2s的op在前
            # return output_str
            op_list_g2s: List[Tuple[NvOp, int]] = []
            op_list_other: List[Tuple[NvOp, int]] = []
            for item in op_list:
                if item[0].mem_type == ("g", "s"):
                    op_list_g2s.append(item)
                else:
                    op_list_other.append(item)
            logger.debug("op_list_g2s:")
            for item in op_list_g2s:
                logger.debug(f"\t{{{item[0].name}, {item[1]}}}, ")
            logger.debug("op_list_other:")
            for item in op_list_other:
                logger.debug(f"\t{{{item[0].name}, {item[1]}}}, ")
            # return output_str

            # > 生成g2s操作
            for item in op_list_g2s:
                buf_idx_str = str(item[1] % nbuf_num)
                shift_constant = str(i)
                output_str += (
                    self.dump_f_method_call(item[0], buf_idx_str, shift_constant) + "\n"
                )

            # > 3. 所有g2s操作做完之后整体添加一个commit
            if len(op_list_g2s) != 0:
                output_str += "__pipeline_commit();\n"
                pipeline_records.append(op_list_g2s)
            # return output_str
            # > 4. 生成所有其他op
            for item in op_list_other:
                # > 如果是从SMEM到其他位置的op，需要插入一条wait prior指令
                if item[0].mem_type[0] == "s":
                    wait_num = get_wait_num(item[0], item[1], pipeline_records)
                    if wait_num != -1:
                        output_str += (
                            f"__pipeline_wait_prior({wait_num});\n__syncthreads();\n"
                        )
                buf_idx_str = str(item[1] % nbuf_num)
                shift_constant = str(i)
                output_str += self.dump_f_method_call(
                    item[0], buf_idx_str, shift_constant
                )

            return output_str

        def dump_short_parallel_K_total(K_total) -> str:
            pipeline_records = []
            output_strs = []
            for i in range(K_total + sum(shifts)):
                output_strs.append(dump_short_parallel_i(i, K_total, pipeline_records))
            return "\n\n".join(output_strs)

        total_stages = sum(shifts)

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
            coord_str = "_, " * len(op.outputs[idx].shape) + "buf_idx"
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
            "template<size_t buf_idx, size_t output_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + "\n".join(
                self._indent_lines(dump_output_with_buf_idx_switch(idx))
                if not op.outputs[idx].unique
                else self._indent_lines(dump_output_without_buf_idx_switch(idx))
                for idx in range(len(op.outputs))
            )
            + "\n"
            + "}"
        )

        output_without_buf_idx_str = (
            "template<size_t output_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + "\n".join(
                self._indent_lines(dump_output_without_buf_idx_switch(idx))
                for idx in range(len(op.outputs))
            )
            + "\n"
            + "}"
        )

        return output_with_buf_idx_str + "\n\n" + output_without_buf_idx_str

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
        if nbuf:
            shape_strs.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shape_strs)
        return f"make_shape({comma_seperated_shapes})"

    def dump_input_decl(self, inp: NvOpInput) -> str:
        def dump_decl_input_shape_type() -> str:
            make_shape_str = self.dump_decl_make_shape(inp.shape, nbuf=False)
            decl_shape_str = f"decltype({make_shape_str})"
            return f"using Shape_i{inp.idx} = {decl_shape_str};"

        def dump_decl_input_shape() -> str:
            return f"Shape_i{inp.idx} shape_i{inp.idx};"

        def dump_decl_input_tensor() -> str:
            return f"// Tensor_i{inp.idx} {inp.name};"

        if len(inp.shape) > 0:
            shape_type_str = dump_decl_input_shape_type()
            shape_str = dump_decl_input_shape()
            tensor_str = dump_decl_input_tensor()
            return f"{shape_type_str}\n{shape_str}\n{tensor_str}"
        else:
            return f"// TI{inp.idx} {inp.name};"

    def dump_output_decl(self, out: NvOpOutput) -> str:
        def dump_decl_output_shape_type() -> str:
            if out.unique:
                make_shape_str = self.dump_decl_make_shape(out.shape, nbuf=False)
            else:
                make_shape_str = self.dump_decl_make_shape(out.shape, nbuf=True)
            decl_shape_str = f"decltype({make_shape_str})"
            return f"using Shape_o{out.idx} = {decl_shape_str};"

        def dump_decl_output_layout_type() -> str:
            row_major = out.attrs.get("row_major", False)
            if (swizzle := out.attrs.get("swizzle", None)) is not None:
                return (
                    f"using Layout_o{out.idx} = decltype(composition(Swizzle<{', '.join(str(s) for s in swizzle)}>{{}}, "
                    f"make_layout(Shape_o{out.idx}{{}}{', LayoutRight{}' if row_major else ''})));"
                )
            else:
                return (
                    f"using Layout_o{out.idx} = decltype(make_layout(Shape_o{out.idx}{{}}"
                    f"{', LayoutRight{}' if row_major else ''}));"
                )

        def dump_decl_output_tensor_type() -> str:
            match out.mem:
                case "gmem":
                    decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr((TO{out.idx}*)nullptr), Layout_o{out.idx}{{}}))"
                case "smem":
                    decl_tensor_str = f"decltype(make_tensor(make_smem_ptr((TO{out.idx}*)nullptr), Layout_o{out.idx}{{}}))"
                case "rmem":
                    decl_tensor_str = (
                        f"decltype(make_tensor<TO{out.idx}>(Layout_o{out.idx}{{}}))"
                    )
            return f"using Tensor_o{out.idx} = {decl_tensor_str};"

        def dump_decl_output_shape() -> str:
            return f"Shape_o{out.idx} shape_o{out.idx};"

        def dump_decl_output_layout() -> str:
            return f"Layout_o{out.idx} layout_o{out.idx};"

        def dump_decl_output_tensor() -> str:
            if out.owning:
                return f"Tensor_o{out.idx} {out.name};"
            else:
                return f"Tensor_o{out.idx} &{out.name};"

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
                idx, inp.shape, is_output=False
            )
            if varlen_idx > 0:
                init_strs.append(f"shape_i{idx}({make_shape_str})")
        return init_strs

    def dump_init_output_varlen_shape(self, op: NvOp) -> List[str]:
        init_strs = []
        for idx, out in enumerate(op.outputs):
            make_shape_str, varlen_idx = self.dump_init_make_shape(
                idx, out.shape, is_output=True
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

    def dump_nvop_create(self, op: NvOp, nbuf: int) -> str:
        snake_name = to_snake(op.name)
        rmem_outputs = [out for out in op.outputs if out.mem == "rmem"]
        smem_outputs = [out for out in op.outputs if out.mem == "smem"]
        gmem_outputs = [out for out in op.outputs if out.mem == "gmem"]

        template_args = []
        template_args.append(f"Int<{nbuf}>")
        template_args.extend(["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"])
        template_args.extend([parameter for parameter in op.parameters])
        template_args.extend([f"decltype({out.name})" for out in op.outputs if not out.owning])
        template_args_str = ", ".join(template_args)
        using_type_str = f"using {snake_name}_t = {op.name}<{template_args_str}>;"

        smem_output_shapes = [
            f"using {snake_name}_layout_o{out.idx} = typename {snake_name}_t::Layout_o{out.idx};"
            for out in smem_outputs
        ]
        smem_output_array_defs = [
            f"__shared__ {out.dtype} {out.name}[cosize_v<{snake_name}_layout_o{out.idx}>];"
            for out in smem_outputs
        ]

        outputs = []
        for out in op.outputs:
            if not out.owning:
                if isinstance(out.source, Tuple):
                    outputs.append(
                        f"{to_snake(out.source[0].name)}.template output<{out.source[1]}>()"
                    )
                else:
                    outputs.append(out.source)
            elif out.mem == "smem":
                outputs.append(out.name)

        varlen_modes = []
        for io in chain(op.inputs, op.outputs):
            for s in io.shape:
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
            f"{snake_name}_t {snake_name}{{\n"
            + self._indent_lines(args_str)
            + "\n"
            + "};"
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
            + op_var_def_str
        )

    def dump_nvop_pipeline_device_functions(self, pipeline: NvOpSequence) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_types = ", ".join([f"class T_{op.name}" for op in ops])
        op_as_params = ", ".join(
            [f"T_{op.name} &{snake_name}" for op, snake_name in zip(ops, snake_names)]
        )
        gmem_tensor_types = "".join(
            [f", class T_{name}" for name in pipeline.gmem_inouts.keys()]
        )
        gmem_tensor_as_params = "".join(
            [f", T_{name} &{name}" for name in pipeline.gmem_inouts.keys()]
        )
        op_as_args = ", ".join(snake_names)
        gmem_tensor_as_args = "".join(
            [f", {name}" for name in pipeline.gmem_inouts.keys()]
        )

        nbuf = max(pipeline.shifts) + 1

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
            buf_idx_str: str
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

        @dataclass
        class PipelinePhase: ...

        @dataclass
        class Filling(PipelinePhase):
            stage: int
            step_in_stage: int

        @dataclass
        class LoopStep(PipelinePhase):
            step: int

        @dataclass
        class Remains(PipelinePhase):
            remains: int
            step: int

        @dataclass
        class Emptying(PipelinePhase):
            stage: int
            step_in_stage: int

        # TODO: more elegant way to find the earliest emptying stage without g2s ops
        no_g2s_emptying_stage_idx = pipeline.num_stages
        while not any(
            op.mem_type == ("g", "s")
            for op in pipeline.stages[no_g2s_emptying_stage_idx - 1].ops
        ):
            no_g2s_emptying_stage_idx -= 1

        def depends_on_g2s(op: NvOp) -> bool:
            for inp in op.inputs:
                if isinstance(inp.source, Tuple):
                    if inp.source[0].mem_type == ("g", "s"):
                        return True
            return False

        def g2s_first_schedule(
            op_calls: List[OpCall], phase: PipelinePhase
        ) -> List[Inst]:
            instructions = []
            g2s_ops = []

            # 1. extract g2s ops first
            for op_call in op_calls:
                if op_call.op.mem_type == ("g", "s"):
                    g2s_ops.append(op_call)

            instructions.extend(g2s_ops)
            if len(g2s_ops) != 0:
                instructions.append(PipelineCommit())

            # 2. schedule the rest
            for op_call in op_calls:
                if op_call.op.mem_type == ("g", "s"):
                    continue
                if depends_on_g2s(op_call.op):
                    if (
                        isinstance(phase, Emptying)
                        and phase.stage >= no_g2s_emptying_stage_idx
                    ):
                        wait_num = sum(
                            pipeline.shifts[
                                (no_g2s_emptying_stage_idx - 1) : (
                                    op_stage_idx_mapping[op_call.op]
                                )
                            ]
                        ) - (phase.step_in_stage + 1)
                    else:
                        wait_num = (
                            pipeline.shifts[op_stage_idx_mapping[op_call.op] - 1]
                            if op_stage_idx_mapping[op_call.op] > 0
                            else 0
                        )
                    instructions.append(PipelineWait(wait_num=wait_num))
                    instructions.append(SyncThreads())
                elif (
                    op_call.op.mem_type[0] == "s"
                ):  # sync smem write before reading, need sync
                    instructions.append(SyncThreads())
                instructions.append(op_call)

            return instructions

        def dump_insts(insts: List[Inst]) -> str:
            inst_strs = []
            for inst in insts:
                if isinstance(inst, OpCall):
                    inst_strs.append(
                        self.dump_f_method_call(
                            inst.op,
                            buf_idx_str=inst.buf_idx_str,
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
            short_pipe_body = self.dump_short_parallel(pipeline.stages, pipeline.shifts)
            return templates.SHORT_PIPE_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
                gmem_tensor_types=gmem_tensor_types,
                gmem_tensor_as_params=gmem_tensor_as_params,
                short_pipe_body=self._indent_lines(short_pipe_body),
            )

        def dump_fill() -> str:
            def dump_fill_body() -> str:
                # All stages start from 0
                stage_counter = [0 for _ in range(len(pipeline.shifts))]

                all_insts = []
                for i in range(len(pipeline.shifts)):
                    all_insts.append(Comment(f"stage 0..{i}"))
                    for step in range(pipeline.shifts[i]):
                        # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 2
                        op_calls = []  # [op0, op1, op2, ...]
                        for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
                            for op in stage.ops:
                                op_calls.append(
                                    OpCall(
                                        op,
                                        buf_idx_str=str(
                                            stage_counter[stage_idx] % nbuf
                                        ),
                                        shift_constant=f"{i}",
                                    )
                                )
                            stage_counter[stage_idx] += 1
                        insts = g2s_first_schedule(
                            op_calls, Filling(stage=i, step_in_stage=step)
                        )
                        all_insts.extend(insts)
                return dump_insts(all_insts)

            return templates.FILL_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
                gmem_tensor_types=gmem_tensor_types,
                gmem_tensor_as_params=gmem_tensor_as_params,
                fill_body=self._indent_lines(dump_fill_body()),
            )

        def dump_loop_step() -> str:
            def dump_loop_step_body() -> str:
                # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
                # The last stage has buf_offset 0
                stage_buf_offsets = [
                    sum(pipeline.shifts[i:]) for i in range(len(pipeline.stages))
                ]

                # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
                # The first stage has offset 0
                stage_idx_offsets = [
                    -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
                ]

                all_insts = []
                for step in range(nbuf):
                    all_insts.append(Comment(f"step {step}"))
                    op_calls = []
                    for stage_idx, stage in enumerate(pipeline.stages):
                        for op in stage.ops:
                            op_calls.append(
                                OpCall(
                                    op,
                                    buf_idx_str=str(
                                        (step + stage_buf_offsets[stage_idx]) % nbuf
                                    ),
                                    shift_constant=f"i + ({step + stage_idx_offsets[stage_idx]})",
                                )
                            )
                    insts = g2s_first_schedule(op_calls, LoopStep(step=step))
                    all_insts.extend(insts)

                return dump_insts(all_insts)

            return templates.LOOP_STEP_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
                gmem_tensor_types=gmem_tensor_types,
                gmem_tensor_as_params=gmem_tensor_as_params,
                loop_step_body=self._indent_lines(dump_loop_step_body()),
            )

        def dump_remainder() -> str:
            def dump_remains_r(r: int) -> str:
                def dump_remains_r_body() -> str:
                    # Stage k has buf_offset (shiftk + shiftk+1 + ... + shiftnstages-1)
                    # The last stage has buf_offset 0
                    stage_buf_offsets = [
                        sum(pipeline.shifts[i:]) for i in range(len(pipeline.stages))
                    ]

                    # Stage k has offset -(shift0 + shift1 + ... + shiftk-1)
                    # The first stage has offset 0
                    stage_idx_offsets = [
                        -sum(pipeline.shifts[:i]) for i in range(len(pipeline.stages))
                    ]

                    all_insts = []
                    for step in range(r):
                        all_insts.append(Comment(f"total remains {r}, current {step}"))
                        op_calls = []
                        for stage_idx, stage in enumerate(pipeline.stages):
                            for op in stage.ops:
                                op_calls.append(
                                    OpCall(
                                        op,
                                        buf_idx_str=str(
                                            (step + stage_buf_offsets[stage_idx]) % nbuf
                                        ),
                                        shift_constant=f"i + ({step + stage_idx_offsets[stage_idx]})",
                                    )
                                )
                        insts = g2s_first_schedule(
                            op_calls, Remains(remains=r, step=step)
                        )
                        all_insts.extend(insts)

                    return dump_insts(all_insts)

                return templates.REMAINS_R_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    gmem_tensor_types=gmem_tensor_types,
                    gmem_tensor_as_params=gmem_tensor_as_params,
                    r=r,
                    remains_r_body=self._indent_lines(dump_remains_r_body()),
                )

            def dump_remainder_dispatch() -> str:
                def dump_remainder_dispatch_body() -> str:
                    def dump_remains_i_call(i: int) -> str:
                        return (
                            f"remains_{i}(i, l, r, {op_as_args}{gmem_tensor_as_args});"
                        )

                    def dump_remains_i_switch_i(i: int) -> str:
                        return ("else " if i > 1 else "") + (
                            f"if (r - l - i == {i}) {{\n"
                            + self._indent_lines(dump_remains_i_call(i))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(dump_remains_i_switch_i(i) for i in range(1, nbuf))

                return templates.REMAINDER_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    gmem_tensor_types=gmem_tensor_types,
                    gmem_tensor_as_params=gmem_tensor_as_params,
                    remainder_dispatch_body=self._indent_lines(
                        dump_remainder_dispatch_body()
                    ),
                )

            return (
                "\n\n".join(dump_remains_r(i) for i in range(1, nbuf))
                + "\n\n"
                + dump_remainder_dispatch()
            )

        def dump_empties() -> str:
            def dump_empty_from_idx(from_idx: int) -> str:
                def dump_empty_from_idx_body(idx: int) -> str:
                    # Stage 0 is completed
                    # Stage 1, 2, 3, ... starts from [idx - shift0, idx - shift0 - shift1, idx - shift0 - shift1 - shift2, ...]
                    stage_counter = [
                        idx - sum(pipeline.shifts[: i + 1])
                        for i in range(len(pipeline.shifts))
                    ]

                    all_insts = []
                    total_step = 0
                    for i in range(len(pipeline.shifts)):
                        all_insts.append(
                            Comment(f"stage {i + 1}..{len(pipeline.stages) - 1}")
                        )
                        for step in range(pipeline.shifts[i]):
                            # Stage 1, 2, ... nstages - 1; Stage 2, 3, ... nstages - 1; ...; Stage nstages - 1
                            op_calls = []
                            for stage_idx, stage in enumerate(
                                pipeline.stages[i + 1 :], start=i
                            ):
                                for op in stage.ops:
                                    op_calls.append(
                                        OpCall(
                                            op,
                                            buf_idx_str=str(
                                                stage_counter[stage_idx] % nbuf
                                            ),
                                            shift_constant=f"i + ({total_step})",
                                        )
                                    )
                                stage_counter[stage_idx] += 1
                            total_step += 1
                            insts = g2s_first_schedule(
                                op_calls, Emptying(stage=i + 1, step_in_stage=step)
                            )
                            all_insts.extend(insts)

                    return dump_insts(all_insts)

                return templates.EMPTY_FROM_IDX_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    gmem_tensor_types=gmem_tensor_types,
                    gmem_tensor_as_params=gmem_tensor_as_params,
                    from_idx=from_idx,
                    empty_from_idx_body=self._indent_lines(
                        dump_empty_from_idx_body(from_idx)
                    ),
                )

            def dump_empty_from_idx_dispatch() -> str:
                def dump_empty_dispatch_body() -> str:
                    def dump_empty_from_idx_call(idx: int) -> str:
                        return (
                            f"empty_{idx}(i, l, r, {op_as_args}{gmem_tensor_as_args});"
                        )

                    def dump_empty_from_buf_idx_switch_i(idx: int) -> str:
                        return ("else " if idx > 0 else "") + (
                            f"if ((r - l - i + ({sum(pipeline.shifts)})) % {nbuf} == {idx}) {{\n"
                            + self._indent_lines(dump_empty_from_idx_call(idx))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(
                        dump_empty_from_buf_idx_switch_i(i) for i in range(nbuf)
                    )

                return templates.EMPTY_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    gmem_tensor_types=gmem_tensor_types,
                    gmem_tensor_as_params=gmem_tensor_as_params,
                    empty_dispatch_body=self._indent_lines(dump_empty_dispatch_body()),
                )

            return (
                "\n\n".join(dump_empty_from_idx(i) for i in range(nbuf))
                + "\n\n"
                + dump_empty_from_idx_dispatch()
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
        )

    def dump_dispatch(self, pipeline: NvOpSequence) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_as_args = ", ".join(snake_names)
        gmem_tensor_as_args = "".join(
            [f", {name}" for name in pipeline.gmem_inouts.keys()]
        )
        sum_shifts = sum(pipeline.shifts)

        nbuf = max(pipeline.shifts) + 1

        def dump_short_pipe_dispatch() -> str:
            return f"short_pipe(k, l, r, {op_as_args}{gmem_tensor_as_args});"

        def dump_fill_dispatch() -> str:
            return f"fill(l, r, {op_as_args}{gmem_tensor_as_args});"

        def dump_loop_step_dispatch() -> str:
            return f"loop_step(i, l, r, {op_as_args}{gmem_tensor_as_args});"

        def dump_remainder_dispatch() -> str:
            return f"remainder(i, l, r, {op_as_args}{gmem_tensor_as_args});"

        def dump_empty_dispatch() -> str:
            return f"empty(i, l, r, {op_as_args}{gmem_tensor_as_args});"

        return templates.GLOBAL_FUNCTION_DISPATCH_SKELETON.format(
            sum_shifts=sum_shifts,
            nbuf=nbuf,
            short_pipe_dispatch=dump_short_pipe_dispatch(),
            fill_dispatch=dump_fill_dispatch(),
            loop_step_dispatch=dump_loop_step_dispatch(),
            remainder_dispatch=dump_remainder_dispatch(),
            empty_dispatch=dump_empty_dispatch(),
        )

    def dump_prologue(self, pipeline: NvOpSequence) -> str:
        prologue_op_str = f"// prologue\n"
        prologue_ops = [
            self.dump_f_method_call(op, buf_idx_str="0") for op in pipeline.prologue
        ]
        prologue_op_str += "\n\n".join(prologue_ops)
        return prologue_op_str + "\n" + templates.DTC_PROLOGUE
        # return templates.DTC_PROLOGUE

    def dump_epilogue(self, pipeline: NvOpSequence) -> str:
        epilogue_op_str = f"// epilogue\n"
        epilogue_ops = [
            self.dump_f_method_call(op, buf_idx_str="0") for op in pipeline.epilogue
        ]
        epilogue_op_str += "\n\n".join(epilogue_ops)
        return epilogue_op_str
        # return templates.DTC_EPILOGUE

    def dump_nvop_pipeline_global_function(self, pipeline: NvOpSequence) -> str:

        def dump_all_parameters() -> str:
            all_parameters = pipeline.all_parameters()
            return "".join([f", class {param}" for param in all_parameters])

        def dump_shifts() -> str:
            return ", ".join(f"int shift{i + 1}" for i in range(len(pipeline.shifts)))

        def dump_gmem_ptrs() -> str:
            return ",\n".join(
                [f"{io.dtype} *d{name}" for name, io in pipeline.gmem_inouts.items()]
            )

        def dump_varlens() -> str:
            return ", ".join([f"int {name}" for name in pipeline.all_varlens()])

        def dump_gmem_tensor_init() -> str:
            def dump_make_shape(shape: Shape) -> str:
                shape_strs = []
                for s in shape:
                    if isinstance(s, IntShape):
                        shape_strs.append(f"Int<{s.value}>{{}}")
                    elif isinstance(s, ParamShape):
                        shape_strs.append(f"{s.param}{{}}")
                    elif isinstance(s, VarlenShape):
                        shape_strs.append(s.name)
                    elif isinstance(s, MnkShape):
                        shape_strs.append(
                            f"get<{0 if s.mnk == 'm' else 1 if s.mnk == 'n' else 2}>({s.param}{{}})"
                        )
                    elif isinstance(s, Shape):
                        shape_strs.append(dump_make_shape(s))
                    else:
                        raise ValueError(f"Unknown shape type: {type(s)}")
                comma_seperated_shapes = ", ".join(shape_strs)
                return f"make_shape({comma_seperated_shapes})"

            tensor_init_strs = ["// gmem tensors"]
            blk_slice_strs = {
                "B_val": "logical_divide({tensor}, make_shape(_1{{}}, get<1>(BLK_MNK{{}})))(_, make_coord(_, blockIdx.y))",
                "C_val": "logical_divide({tensor}, select<1, 0>(BLK_MNK{{}}))(make_coord(_, blockIdx.y), make_coord(_, blockIdx.x))",
            }
            for name, io in pipeline.gmem_inouts.items():
                make_tensor_str = (
                    f"make_tensor(make_gmem_ptr(d{name}), {dump_make_shape(io.shape)})"
                )
                if name in blk_slice_strs:
                    make_tensor_str = blk_slice_strs[name].format(
                        tensor=make_tensor_str
                    )
                tensor_init_strs.append(f"Tensor {name} = {make_tensor_str};")
            return "\n".join(tensor_init_strs)

        all_parameters = dump_all_parameters()
        shifts = dump_shifts()
        kernel_name = pipeline.name
        gmem_ptrs = self._indent_lines(dump_gmem_ptrs())
        varlens = self._indent_lines(dump_varlens())
        gmem_tensor_init = self._indent_lines(dump_gmem_tensor_init())
        nvop_inits = self._indent_lines(self.dump_nvop_inits(pipeline))
        prologue = self._indent_lines(self.dump_prologue(pipeline))
        dispatch = self._indent_lines(self.dump_dispatch(pipeline))
        epilogue = self._indent_lines(self.dump_epilogue(pipeline))
        return templates.GLOBAL_FUNCTION_SKELETON.format(
            all_parameters=all_parameters,
            shifts=shifts,
            kernel_name=kernel_name,
            gmem_ptrs=gmem_ptrs,
            varlens=varlens,
            gmem_tensor_init=gmem_tensor_init,
            nvop_inits=nvop_inits,
            prologue=prologue,
            dispatch=dispatch,
            epilogue=epilogue,
        )

    def dump_nvop_defs(self, pipeline: NvOpSequence) -> str:
        stages = []
        for stage_idx, stage in enumerate(pipeline.stages):
            stage_op_str = f"// stage {stage_idx}\n"
            ops = [self.dump_nvop_class_def(op) for op in stage.ops]
            stage_op_str += "\n\n".join(ops)
            stages.append(stage_op_str)
        prologue_op_str = f"// prologue\n"
        prologue_ops = [self.dump_nvop_class_def(op) for op in pipeline.prologue]
        prologue_op_str += "\n\n".join(prologue_ops)
        epilogue_op_str = f"// epilogue\n"
        epilogue_ops = [self.dump_nvop_class_def(op) for op in pipeline.epilogue]
        epilogue_op_str += "\n\n".join(epilogue_ops)
        return "\n\n".join([prologue_op_str] + stages + [epilogue_op_str])

    def dump_nvop_inits(self, pipeline: NvOpSequence) -> str:
        stages = []
        for stage_idx, stage in enumerate(pipeline.stages):
            stage_op_str = f"// stage {stage_idx}\n"
            ops = [self.dump_nvop_create(op, pipeline.nbuf) for op in stage.ops]
            stage_op_str += "\n\n".join(ops)
            stages.append(stage_op_str)
        prologue_op_str = f"// prologue\n"
        prologue_ops = [self.dump_nvop_create(op, 1) for op in pipeline.prologue]
        prologue_op_str += "\n\n".join(prologue_ops)
        epilogue_op_str = f"// epilogue\n"
        epilogue_ops = [self.dump_nvop_create(op, 1) for op in pipeline.epilogue]
        epilogue_op_str += "\n\n".join(epilogue_ops)
        return "\n\n".join([prologue_op_str] + stages + [epilogue_op_str])

    def dump_nvop_pipeline(self, pipeline: NvOpSequence) -> str:
        device_helper_functions = templates.DTC_HELPER_FUNCTIONS
        nvop_defs = self.dump_nvop_defs(pipeline)
        nvop_pipeline_device_functions = self.dump_nvop_pipeline_device_functions(
            pipeline
        )
        global_function = self.dump_nvop_pipeline_global_function(pipeline)
        return templates.CUDA_SOURCE_SKELETON.format(
            device_helper_functions=device_helper_functions,
            nvop_defs=nvop_defs,
            nvop_pipeline_device_functions=nvop_pipeline_device_functions,
            global_function=global_function,
        )
