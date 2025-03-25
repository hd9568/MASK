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

import re


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

    def dump_nvop_class_def(self, op: NvOp, indent_level: int = 0) -> str:
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

    def dump_f_method(self, op: NvOp, indent_level: int = 0) -> str:
        code_str = op.impl.code_template
        f_template_head_str = (
            "template<size_t buf_idx"
            + (", size_t wait_num" if op.mem_type[0] == "s" else "")
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

    def dump_get_output_method_switch(
        self, idx: int, op: NvOp, indent_level: int = 0
    ) -> str:
        coord_str = "_, " * len(op.outputs[idx].shape) + "buf_idx"
        return (
            f"if constexpr (output_idx == {idx}) {{\n"
            + self._indent_lines(f"return {op.outputs[idx].name}({coord_str});")
            + "\n"
            + "}"
        )

    def dump_get_output_method(self, op: NvOp, indent_level: int = 0) -> str:
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

    def dump_input_decl(self, inp: NvOpInput, op: NvOp, indent_level: int = 0) -> str:
        if len(inp.shape) > 0:
            shape_type_str = self.dump_decl_input_shape_type(inp.idx, inp.shape)
            tensor_type_str = self.dump_decl_input_tensor_type(inp.idx, op)
            shape_str = self.dump_decl_input_shape(inp.idx)
            tensor_str = self.dump_decl_input_tensor(inp)
            return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}\n{tensor_str}"
        else:
            return f"// TI{inp.idx} {inp.name};"

    def dump_decl_input_shape_type(
        self, idx: int, shape: Shape, indent_level: int = 0
    ) -> str:
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

    def dump_decl_input_tensor_type(
        self, idx: int, op: NvOp, indent_level: int = 0
    ) -> str:
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

    def dump_output_decl(self, out: NvOpOutput, op: NvOp, indent_level: int = 0) -> str:
        shape_type_str = self.dump_decl_output_shape_type(out.idx, out.shape)
        tensor_type_str = self.dump_decl_output_tensor_type(out.idx, op)
        shape_str = self.dump_decl_output_shape(out.idx)
        tensor_str = self.dump_decl_output_tensor(out)
        return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}\n{tensor_str}"

    def dump_decl_output_shape_type(
        self, idx: int, shape: Shape, indent_level: int = 0
    ) -> str:
        make_shape_str = self.dump_decl_make_shape(shape, is_output=True)
        decl_shape_str = f"decltype({make_shape_str})"
        return f"using Shape_o{idx} = {decl_shape_str};"

    def dump_decl_output_tensor_type(
        self, idx: int, op: NvOp, indent_level: int = 0
    ) -> str:
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

    def dump_init_input_varlen_shape(
        self, op: NvOp, indent_level: int = 0
    ) -> List[str]:
        init_strs = []
        for idx, inp in enumerate(op.inputs):
            make_shape_str, varlen_idx = self.dump_init_make_shape(
                idx, inp.shape, is_output=False
            )
            if varlen_idx > 0:
                init_strs.append(f"shape_i{idx}({make_shape_str})")
        return init_strs

    def dump_init_output_varlen_shape(
        self, op: NvOp, indent_level: int = 0
    ) -> List[str]:
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

    def dump_nvop_create(self, op: NvOp, indent_level: int = 0) -> str:
        snake_name = to_snake(op.name)
        rmem_outputs = [out for out in op.outputs if out.mem == "rmem"]
        smem_outputs = [out for out in op.outputs if out.mem == "smem"]
        gmem_outputs = [out for out in op.outputs if out.mem == "gmem"]

        template_args = []
        if op.attrs.get("reduction", False) == True:
            template_args.append("Int<1> /* reduction */")
        else:
            if smem_outputs:
                template_args.append("Int<nbuf_smem>")
            if rmem_outputs:
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

        return self._indent_lines(
            using_type_str
            + "\n"
            + (("\n".join(smem_output_shapes) + "\n") if smem_output_shapes else "")
            + (
                ("\n".join(smem_output_array_defs) + "\n")
                if smem_output_array_defs
                else ""
            )
            + op_var_def_str,
            indent_level,
        )

    def dump_f_method_call_arg(
        self, inp: NvOpInput, *, buf_idx: Optional[int] = None, **kwargs
    ) -> str:
        if isinstance(inp.source, tuple):
            assert buf_idx is not None
            return f"{inp.source[0].name}.template output<{buf_idx}, {inp.source[1]}>()"
        else:
            return inp.source.format(**kwargs)

    def dump_f_method_call(self, op: NvOp, buf_idx: int, **kwargs) -> str:
        # looks like:
        # g2r_coo_atomic_format_load_idx_op.f<(I + shift3 + shift2) % nbuf_rmem>(val_coo_idx,
        #     g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 0>(),
        #     g2r_coo_atomic_format_load_off_op.template output<(I + shift3 + shift2) % nbuf_rmem, 1>()
        # );
        arg_strs = [
            self.dump_f_method_call_arg(inp, buf_idx=buf_idx, **kwargs)
            for inp in op.inputs
        ]
        return (
            f"{op.name}.f<{buf_idx}>(\n"
            + self._indent_lines(",\n".join(arg_strs))
            + "\n"
            + ");"
        )

    def dump_nvop_pipeline_device_functions(
        self, pipeline: NvOpPipeline, indent_level: int = 0
    ) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_types = ", ".join([f"class T_{op.name}" for op in ops])
        op_as_params = ", ".join(
            [f"T_{op.name} {snake_name}" for op, snake_name in zip(ops, snake_names)]
        )
        op_as_args = ", ".join(snake_names)
        op_shifts = [
            sum(pipeline.shifts[:idx])
            for idx, stage in enumerate(pipeline.stages)
            for _ in stage.ops
        ]

        nbuf = max(pipeline.shifts) + 1

        def dump_fill() -> str:
            def dump_fill_body() -> str:
                # All stages start from 0
                stage_counter = [0 for _ in range(len(pipeline.shifts))]

                op_calls = []
                for i in range(len(pipeline.shifts)):
                    for _ in range(pipeline.shifts[i]):
                        # Stage 0; Stage 0, 1; Stage 0, 1, 2; ...; Stage 0, 1, 2, ..., nstages - 2
                        for stage_idx, stage in enumerate(pipeline.stages[: i + 1]):
                            for op in stage.ops:
                                op_calls.append(
                                    self.dump_f_method_call(
                                        op,
                                        buf_idx=stage_counter[stage_idx] % nbuf,
                                        global_i=f"{i}",
                                    )
                                )
                            stage_counter[stage_idx] += 1

                return "\n".join(op_calls)

            return templates.FILL_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
                fill_body=dump_fill_body(),
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
                                    buf_idx=(it + stage_buf_offsets[stage_idx]) % nbuf,
                                    global_i=f"i + ({it + stage_idx_offsets[stage_idx]})",
                                )
                            )

                return "\n".join(op_calls)

            return templates.LOOP_STEP_DEVICE_FUNCTION_SKELETON.format(
                op_types=op_types,
                op_as_params=op_as_params,
                loop_step_body=dump_loop_step_body(),
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
                        for stage_idx, stage in enumerate(pipeline.stages):
                            for op in stage.ops:
                                op_calls.append(
                                    self.dump_f_method_call(
                                        op,
                                        buf_idx=(it + stage_buf_offsets[stage_idx])
                                        % nbuf,
                                        global_i=f"i + ({it + stage_idx_offsets[stage_idx]})",
                                    )
                                )

                    return "\n".join(op_calls)

                return templates.REMAINS_R_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    r=r,
                    remains_r_body=dump_remains_r_body(),
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
                    remainder_dispatch_body=dump_remainder_dispatch_body(),
                )

            return (
                "\n".join(dump_remains_r(i + 1) for i in range(len(pipeline.shifts)))
                + "\n"
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
                        for _ in range(pipeline.shifts[i]):
                            # Stage 1, 2, ... nstages - 1; Stage 2, 3, ... nstages - 1; ...; Stage nstages - 1
                            for stage_idx, stage in enumerate(
                                pipeline.stages[i + 1 :], start=i
                            ):
                                for op in stage.ops:
                                    op_calls.append(
                                        self.dump_f_method_call(
                                            op,
                                            buf_idx=stage_counter[stage_idx] % nbuf,
                                            global_i=f"i + ({it})",
                                        )
                                    )
                                stage_counter[stage_idx] += 1
                            it += 1
                    return "\n".join(op_calls)

                return templates.EMPTY_FROM_IDX_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    from_idx=from_idx,
                    empty_from_idx_body=dump_empty_from_idx_body(from_idx),
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
                        dump_empty_from_buf_idx_switch_i(i + 1)
                        for i in range(len(pipeline.shifts))
                    )

                return templates.EMPTY_DEVICE_FUNCTION_SKELETON.format(
                    op_types=op_types,
                    op_as_params=op_as_params,
                    empty_dispatch_body=dump_empty_dispatch_body(),
                )

            return (
                "\n".join(
                    dump_empty_from_idx(i + 1) for i in range(len(pipeline.shifts))
                )
                + "\n"
                + dump_empty_from_idx_dispatch()
            )

        return (
            dump_fill()
            + "\n"
            + dump_loop_step()
            + "\n"
            + dump_remainder()
            + "\n"
            + dump_empties()
        )

    def dump_dispatch(self, pipeline: NvOpPipeline) -> str:
        ops = [op for stage in pipeline.stages for op in stage.ops]
        snake_names = [to_snake(op.name) for op in ops]
        op_types = ", ".join([f"class T_{op.name}" for op in ops])
        op_as_params = ", ".join(
            [f"T_{op.name} {snake_name}" for op, snake_name in zip(ops, snake_names)]
        )
        op_as_args = ", ".join(snake_names)
        sum_shifts = sum(pipeline.shifts)

        nbuf = max(pipeline.shifts) + 1

        def dump_short_pipe_dispatch() -> str:
            return f"short_pipe(i, l, r, {op_as_args});"

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

    def dump_epilogue(self) -> str:
        raise NotImplementedError

    def dump_nvop_pipeline(self, pipeline: NvOpPipeline, indent_level: int = 0) -> str:
        raise NotImplementedError
