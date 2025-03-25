from typing import List, Dict, Tuple, Optional
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
from sparsene.op_gen.nvir.pipeliner import NvOpPipeline, NvOpStage
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
        parameter_types_strs = [
            f"class {param} = Int<{op.parameters[param]}>" for param in op.parameters
        ]
        parameter_types_str = ", ".join(parameter_types_strs)
        parameter_types_str = ", " + parameter_types_str if parameter_types_str else ""

        in_types_strs = [
            f"class TI{idx} = {inp.dtype}" for idx, inp in enumerate(op.inputs)
        ]
        out_types_strs = [
            f"class TO{idx} = {out.dtype}" for idx, out in enumerate(op.outputs)
        ]
        io_types_str = ", ".join(chain(in_types_strs, out_types_strs))
        io_types_str = ", " + io_types_str if io_types_str else ""

        template_head_str = f"template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK{parameter_types_str}{io_types_str}>"

        class_head_str = f"class {op.name} {{\npublic:"

        inputs_str = f"// Inputs\n" + "\n".join(
            map(lambda inp: self.dump_input_decl(inp, op), op.inputs)
        )
        outputs_str = f"// Outputs\n" + "\n".join(
            map(lambda out: self.dump_output_decl(out, op), op.outputs)
        )

        hw_param_str = f"// Hardware parameters\nint tid, lid, wid;"

        hw_constructor_params = ["int tid", "int lid", "int wid"]
        output_ptr_constructor_params = [
            f"{out.dtype} *{out.name}" for out in op.outputs if out.mem != "rmem"
        ]
        varlen_modes_constructor_params = []
        for out in op.outputs:
            i: int = 0
            for s in out.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes_constructor_params.append(f"int mode_o{out.idx}_{i}")
                    i += 1
        for inp in op.inputs:
            i: int = 0
            for s in inp.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes_constructor_params.append(f"int mode_i{inp.idx}_{i}")
                    i += 1
        constructor_params = ", ".join(
            chain(
                hw_constructor_params,
                output_ptr_constructor_params,
                varlen_modes_constructor_params,
            )
        )

        constructor_head_str = f"CUTE_DEVICE {op.name}({constructor_params})"
        hw_param_constructor_init_list = ["tid(tid)", "lid(lid)", "wid(wid)"]
        varlen_shapes_constructor_init_list = self.dump_init_input_varlen_shape(
            op
        ) + self.dump_init_output_varlen_shape(op)
        output_tensor_constructor_init_list = [
            f"{out.name}(make_tensor(make_{out.mem}_ptr({out.name}), Shape_o{out.idx}{{}}))"
            for out in op.outputs
            if len(out.shape) > 0 and out.mem != "rmem"
        ]

        constructor_init_list = (
            hw_param_constructor_init_list
            + varlen_shapes_constructor_init_list
            + output_tensor_constructor_init_list
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
        code_str = op.impl.code_template
        f_template_head_str = (
            "template<size_t buf_idx"
            # + (", size_t wait_num" if op.mem_type[0] == "s" else "")
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

    def dump_get_output_method_switch(self, idx: int, op: NvOp) -> str:
        coord_str = "_, " * len(op.outputs[idx].shape) + "buf_idx"
        return (
            f"if constexpr (output_idx == {idx}) {{\n"
            + self._indent_lines(f"return {op.outputs[idx].name}({coord_str});")
            + "\n"
            + "}"
        )

    def dump_get_output_method(self, op: NvOp) -> str:
        return (
            "template<size_t buf_idx, size_t output_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + "\n".join(
                self._indent_lines(self.dump_get_output_method_switch(idx, op), 1)
                for idx in range(len(op.outputs))
            )
            + "\n"
            + "}"
        )

    def dump_input_decl(self, inp: NvOpInput, op: NvOp) -> str:
        if len(inp.shape) > 0:
            shape_type_str = self.dump_decl_input_shape_type(inp.idx, inp.shape)
            tensor_type_str = self.dump_decl_input_tensor_type(inp.idx, op)
            shape_str = self.dump_decl_input_shape(inp.idx)
            tensor_str = self.dump_decl_input_tensor(inp)
            return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}\n{tensor_str}"
        else:
            return f"// TI{inp.idx} {inp.name};"

    def dump_decl_input_shape_type(self, idx: int, shape: Shape) -> str:
        make_shape_str = self.dump_decl_make_shape(shape, is_output=False)
        decl_shape_str = f"decltype({make_shape_str})"
        return f"using Shape_i{idx} = {decl_shape_str};"

    def dump_decl_make_shape(self, shape: Shape, *, is_output: bool) -> str:
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
                shape_strs.append(self.dump_decl_make_shape(s, is_output=False))
            else:
                raise ValueError(f"Unknown shape type: {type(s)}")
        if is_output:
            shape_strs.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shape_strs)
        return f"make_shape({comma_seperated_shapes})"

    def dump_decl_input_tensor_type(self, idx: int, op: NvOp) -> str:
        match op.inputs[idx].mem:
            case "gmem":
                decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr((TI{idx}*)nullptr), Shape_i{idx}{{}}))"
            case "smem":
                decl_tensor_str = f"decltype(make_tensor(make_smem_ptr((TI{idx}*)nullptr), Shape_i{idx}{{}}))"
            case "rmem":
                decl_tensor_str = f"decltype(make_tensor(make_rmem_ptr((TI{idx}*)nullptr), Shape_i{idx}{{}}))"
        return f"using Tensor_i{idx} = {decl_tensor_str};"

    def dump_decl_input_shape(self, idx: int) -> str:
        return f"Shape_i{idx} shape_i{idx};"

    def dump_decl_input_tensor(self, inp: NvOpInput) -> str:
        return f"// Tensor_i{inp.idx} {inp.name};"

    def dump_output_decl(self, out: NvOpOutput, op: NvOp) -> str:
        shape_type_str = self.dump_decl_output_shape_type(out.idx, out.shape)
        tensor_type_str = self.dump_decl_output_tensor_type(out.idx, op)
        shape_str = self.dump_decl_output_shape(out.idx)
        tensor_str = self.dump_decl_output_tensor(out)
        return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}\n{tensor_str}"

    def dump_decl_output_shape_type(self, idx: int, shape: Shape) -> str:
        make_shape_str = self.dump_decl_make_shape(shape, is_output=True)
        decl_shape_str = f"decltype({make_shape_str})"
        return f"using Shape_o{idx} = {decl_shape_str};"

    def dump_decl_output_tensor_type(self, idx: int, op: NvOp) -> str:
        match op.outputs[idx].mem:
            case "gmem":
                decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr((TO{idx}*)nullptr), Shape_o{idx}{{}}))"
            case "smem":
                decl_tensor_str = f"decltype(make_tensor(make_smem_ptr((TO{idx}*)nullptr), Shape_o{idx}{{}}))"
            case "rmem":
                decl_tensor_str = f"decltype(make_tensor<TO{idx}>(Shape_o{idx}{{}}))"
        return f"using Tensor_o{idx} = {decl_tensor_str};"

    def dump_decl_output_shape(self, idx: int) -> str:
        return f"Shape_o{idx} shape_o{idx};"

    def dump_decl_output_tensor(self, out: NvOpOutput) -> str:
        return f"Tensor_o{out.idx} {out.name};"

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
                shape_strs.append(self.dump_decl_make_shape(s, is_output=False))
            else:
                raise ValueError(f"Unknown shape type: {type(s)}")
        if is_output:
            shape_strs.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shape_strs)
        return f"make_shape({comma_seperated_shapes})", varlen_idx

    def dump_nvop_create(self, op: NvOp) -> str:
        snake_name = to_snake(op.name)
        rmem_outputs = [out for out in op.outputs if out.mem == "rmem"]
        smem_outputs = [out for out in op.outputs if out.mem == "smem"]
        gmem_outputs = [out for out in op.outputs if out.mem == "gmem"]

        template_args = []
        if op.attrs.get("reduction", False) == True:
            template_args.append("Int<1> /* reduction */")
        else:
            # TODO: use per stage nbuf instead of per memtype
            if smem_outputs:
                template_args.append("Int<nbuf_smem>")
            elif rmem_outputs:
                template_args.append("Int<nbuf_rmem>")
        template_args.extend(["BLK_MNK", "MMA_MNK", "BLK_MMA_MNK"])
        template_args.extend([parameter for parameter in op.parameters])
        template_args_str = ", ".join(template_args)
        using_type_str = f"using {snake_name}_t = {op.name}<{template_args_str}>;"

        smem_output_shapes = [
            f"using {snake_name}_shape_o{out.idx} = typename {snake_name}_t::Shape_o{out.idx};"
            for out in smem_outputs
        ]
        smem_output_array_defs = [
            f"__shared__ {out.dtype} {out.name}[cosize_v<Layout<{snake_name}_shape_o{out.idx}>>];"
            for out in smem_outputs
        ]

        varlen_modes = []
        for io in chain(op.inputs, op.outputs):
            for s in io.shape:
                if isinstance(s, VarlenShape):
                    varlen_modes.append(s.name)

        args_0 = ["tid", "lid", "wid"]
        args_1 = [out.name for out in smem_outputs]
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

    def dump_short_pipe_device_function(self, pipeline: NvOpPipeline) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_types = ", ".join([f"class T_{op.name}" for op in ops])
        op_as_params = ", ".join(
            [f"T_{op.name} {snake_name}" for op, snake_name in zip(ops, snake_names)]
        )
        short_pipe_body = self.dump_short_parallel(pipeline.stages, pipeline.shifts)
        return "// short pipe\n" + templates.SHORT_PIPE_DEVICE_FUNCTION_SKELETON.format(
            op_types=op_types,
            op_as_params=op_as_params,
            short_pipe_body=self._indent_lines(short_pipe_body),
        )

    def dump_nvop_pipeline_device_functions(self, pipeline: NvOpPipeline) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_types = ", ".join([f"class T_{op.name}" for op in ops])
        op_as_params = ", ".join(
            [f"T_{op.name} {snake_name}" for op, snake_name in zip(ops, snake_names)]
        )
        op_as_args = ", ".join(snake_names)

        nbuf = max(pipeline.shifts) + 1

        def dump_fill() -> str:
            def dump_fill_body() -> str:
                # All stages start from 0
                stage_counter = [0 for _ in range(len(pipeline.shifts))]

                op_calls = []
                for i in range(len(pipeline.shifts)):
                    op_calls.append(f"// stage 0..{i}")
                    for _ in range(pipeline.shifts[i]):
                        # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 2
                        for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
                            for op in stage.ops:
                                op_calls.append(
                                    self.dump_f_method_call(
                                        op,
                                        buf_idx_str=str(
                                            stage_counter[stage_idx] % nbuf
                                        ),
                                        shift_constant=f"{i}",
                                    )
                                )
                            stage_counter[stage_idx] += 1

                return "\n".join(op_calls)

            return templates.FILL_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
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

                op_calls = []
                for it in range(nbuf):
                    for stage_idx, stage in enumerate(pipeline.stages):
                        for op in stage.ops:
                            op_calls.append(
                                self.dump_f_method_call(
                                    op,
                                    buf_idx_str=str(
                                        (it + stage_buf_offsets[stage_idx]) % nbuf
                                    ),
                                    shift_constant=f"i + ({it + stage_idx_offsets[stage_idx]})",
                                )
                            )

                return "\n".join(op_calls)

            return templates.LOOP_STEP_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
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

                    op_calls = []
                    for it in range(r):
                        op_calls.append(f"// remains {r - it} out of {r}")
                        for stage_idx, stage in enumerate(pipeline.stages):
                            for op in stage.ops:
                                op_calls.append(
                                    self.dump_f_method_call(
                                        op,
                                        buf_idx_str=str(
                                            (it + stage_buf_offsets[stage_idx]) % nbuf
                                        ),
                                        shift_constant=f"i + ({it + stage_idx_offsets[stage_idx]})",
                                    )
                                )

                    return "\n".join(op_calls)

                return templates.REMAINS_R_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    r=r,
                    remains_r_body=self._indent_lines(dump_remains_r_body()),
                )

            def dump_remainder_dispatch() -> str:
                def dump_remainder_dispatch_body() -> str:
                    def dump_remains_i_call(i: int) -> str:
                        return f"remains_{i}(i, l, r, {op_as_args})"

                    def dump_remains_i_switch_i(i: int) -> str:
                        return ("else " if i > 0 else "") + (
                            f"if (r - l - i == {i}) {{\n"
                            + self._indent_lines(dump_remains_i_call(i))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(
                        dump_remains_i_switch_i(i + 1)
                        for i in range(len(pipeline.shifts))
                    )

                return templates.REMAINDER_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    remainder_dispatch_body=self._indent_lines(
                        dump_remainder_dispatch_body()
                    ),
                )

            return (
                "\n\n".join(dump_remains_r(i + 1) for i in range(len(pipeline.shifts)))
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

                    op_calls = []
                    it = 0
                    for i in range(len(pipeline.shifts)):
                        op_calls.append(f"// stage {i + 1}..{len(pipeline.stages) - 1}")
                        for _ in range(pipeline.shifts[i]):
                            # Stage 1, 2, ... nstages - 1; Stage 2, 3, ... nstages - 1; ...; Stage nstages - 1
                            for stage_idx, stage in enumerate(
                                pipeline.stages[i + 1 :], start=i
                            ):
                                for op in stage.ops:
                                    op_calls.append(
                                        self.dump_f_method_call(
                                            op,
                                            buf_idx_str=str(
                                                stage_counter[stage_idx] % nbuf
                                            ),
                                            shift_constant=f"i + ({it})",
                                        )
                                    )
                                stage_counter[stage_idx] += 1
                            it += 1
                    return "\n".join(op_calls)

                return templates.EMPTY_FROM_IDX_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    from_idx=from_idx,
                    empty_from_idx_body=self._indent_lines(
                        dump_empty_from_idx_body(from_idx)
                    ),
                )

            def dump_empty_from_idx_dispatch() -> str:
                def dump_empty_dispatch_body() -> str:
                    def dump_empty_from_idx_call(idx: int) -> str:
                        return f"empty_{idx}(i, l, r, {op_as_args})"

                    def dump_empty_from_buf_idx_switch_i(idx: int) -> str:
                        return ("else " if idx > 0 else "") + (
                            f"if ((r - l - i + ({sum(pipeline.shifts)})) % {nbuf} == {idx}) {{\n"
                            + self._indent_lines(dump_empty_from_idx_call(idx))
                            + "\n"
                            + "}"
                        )

                    return "\n".join(
                        dump_empty_from_buf_idx_switch_i(i)
                        for i in range(len(pipeline.stages))
                    )

                return templates.EMPTY_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    empty_dispatch_body=self._indent_lines(dump_empty_dispatch_body()),
                )

            return (
                "\n\n".join(
                    dump_empty_from_idx(i + 1) for i in range(len(pipeline.shifts))
                )
                + "\n\n"
                + dump_empty_from_idx_dispatch()
            )

        return (
            "// fill\n"
            + dump_fill()
            + "\n\n// loop_step\n"
            + dump_loop_step()
            + "\n\n// remainder\n"
            + dump_remainder()
            + "\n\n// empties\n"
            + dump_empties()
        )

    def dump_dispatch(self, pipeline: NvOpPipeline) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_as_args = ", ".join(snake_names)
        sum_shifts = sum(pipeline.shifts)

        nbuf = max(pipeline.shifts) + 1

        def dump_short_pipe_dispatch() -> str:
            return f"short_pipe(k, l, r, {op_as_args});"

        def dump_fill_dispatch() -> str:
            return f"fill(i, l, r, {op_as_args});"

        def dump_loop_step_dispatch() -> str:
            return f"loop_step(i, l, r, {op_as_args});"

        def dump_remainder_dispatch() -> str:
            return f"remainder(i, l, r, {op_as_args});"

        def dump_empty_dispatch() -> str:
            return f"empty(i, l, r, {op_as_args});"

        return templates.GLOBAL_FUNCTION_DISPATCH_SKELETON.format(
            sum_shifts=sum_shifts,
            nbuf=nbuf,
            short_pipe_dispatch=dump_short_pipe_dispatch(),
            fill_dispatch=dump_fill_dispatch(),
            loop_step_dispatch=dump_loop_step_dispatch(),
            remainder_dispatch=dump_remainder_dispatch(),
            empty_dispatch=dump_empty_dispatch(),
        )

    def dump_prologue(self, pipeline: NvOpPipeline) -> str:
        return templates.DTC_PROLOGUE

    def dump_epilogue(self, pipeline: NvOpPipeline) -> str:
        return templates.DTC_EPILOGUE

    def dump_nvop_pipeline_global_function(self, pipeline: NvOpPipeline) -> str:

        def dump_all_parameters() -> str:
            return ", ".join([f"class {param}" for param in pipeline.all_parameters()])

        def dump_shifts() -> str:
            return ", ".join(f"int shift{i + 1}" for i in range(len(pipeline.shifts)))

        def dump_gmem_ptrs() -> str:
            return ", ".join([f"d{name}" for name in pipeline.all_gmem_inouts()])

        def dump_varlens() -> str:
            return ", ".join(pipeline.all_varlens())

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
            for io in pipeline.all_inputs_with_memtype(
                "gmem"
            ) + pipeline.all_outputs_with_memtype("gmem"):
                tensor_init_strs.append(
                    f"auto {io.name} = make_tensor(make_gmem_ptr(d{io.name}), {dump_make_shape(io.shape)});"
                )
            return "\n".join(tensor_init_strs)

        all_parameters = dump_all_parameters()
        shifts = dump_shifts()
        kernel_name = pipeline.name
        gmem_ptrs = dump_gmem_ptrs()
        varlens = dump_varlens()
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

    def dump_nvop_defs(self, pipeline: NvOpPipeline) -> str:
        stages = []
        for stage_idx, stage in enumerate(pipeline.stages):
            stage_op_str = f"// stage {stage_idx}\n"
            ops = [self.dump_nvop_class_def(op) for op in stage.ops]
            stage_op_str += "\n\n".join(ops)
            stages.append(stage_op_str)
        return "\n\n".join(stages)

    def dump_nvop_inits(self, pipeline: NvOpPipeline) -> str:
        stages = []
        for stage_idx, stage in enumerate(pipeline.stages):
            stage_op_str = f"// stage {stage_idx}\n"
            ops = [self.dump_nvop_create(op) for op in stage.ops]
            stage_op_str += "\n\n".join(ops)
            stages.append(stage_op_str)
        return "\n\n".join(stages)

    def dump_nvop_pipeline(self, pipeline: NvOpPipeline) -> str:
        device_helper_functions = templates.DTC_HELPER_FUNCTIONS
        nvop_defs = self.dump_nvop_defs(pipeline)
        short_pipe_device_function = self.dump_short_pipe_device_function(pipeline)
        long_pipe_device_functions = self.dump_nvop_pipeline_device_functions(pipeline)
        global_function = self.dump_nvop_pipeline_global_function(pipeline)
        return templates.CUDA_SOURCE_SKELETON.format(
            device_helper_functions=device_helper_functions,
            nvop_defs=nvop_defs,
            short_pipe_device_function=short_pipe_device_function,
            long_pipe_device_functions=long_pipe_device_functions,
            global_function=global_function,
        )
