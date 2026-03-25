from __future__ import annotations

import json
import keyword
import re
from typing import Dict, List, Sequence

from sparsene.op_gen.nvir.nvop import (
    ConstantNvOp,
    ForLoopCurrentIdx,
    ForLoopCurrentIter,
    ForLoopEnd,
    ForLoopNvOp,
    ForLoopStart,
    GmemInout,
    GmemTensorNvOp,
    IntShape,
    LayoutHint,
    MnkShape,
    NvOp,
    NvOpInput,
    NvOpOutput,
    NvOpPipeline,
    NvOpProgram,
    NvOpType,
    NvOpTensor,
    ParamShape,
    Shape,
    SwizzleLayout,
    VarlenShape,
)


class NvProgramPrinter:
    def __init__(
        self,
        indent_size: int = 2,
        show_io: bool = True,
        show_impl: bool = False,
        impl_max_lines: int = 8,
        debug_indent: bool = False,
    ):
        self.indent_size = indent_size
        self.show_io = show_io
        self.show_impl = show_impl
        self.impl_max_lines = impl_max_lines
        self.debug_indent = debug_indent
        self.debug_chars = "."
        self._lines: List[str] = []
        self._name_counter: Dict[str, int] = {}
        self._op_var_names: Dict[NvOp, str] = {}
        self._declared_ops: Dict[NvOp, bool] = {}
        self._program_var_name = "nvir_program"

    def _reset_state(self) -> None:
        self._lines = []
        self._name_counter = {}
        self._op_var_names = {}
        self._declared_ops = {}
        self._program_var_name = "nvir_program"

    def _get_indent(self, indent_level: int) -> str:
        return (
            (self.debug_chars if self.debug_indent else " ")
            * indent_level
            * self.indent_size
        )

    def _line(self, indent_level: int, text: str) -> str:
        return f"{self._get_indent(indent_level)}{text}"

    def _emit(self, indent_level: int, text: str) -> None:
        self._lines.append(self._line(indent_level, text))

    def _emit_blank(self) -> None:
        if self._lines and self._lines[-1] != "":
            self._lines.append("")

    def _quote(self, value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _sanitize_identifier(self, raw_name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        if not sanitized:
            sanitized = "op"
        if sanitized[0].isdigit():
            sanitized = f"v_{sanitized}"
        if keyword.iskeyword(sanitized):
            sanitized = f"{sanitized}_op"
        return sanitized

    def _allocate_name(self, base: str) -> str:
        idx = self._name_counter.get(base, 0)
        self._name_counter[base] = idx + 1
        return base if idx == 0 else f"{base}_{idx}"

    def _program_var(self, program_name: str) -> str:
        base = self._sanitize_identifier(program_name)
        if not base.endswith("_program"):
            base = f"{base}_program"
        return self._allocate_name(base)

    def _op_var(self, op: NvOp) -> str:
        if isinstance(op, GmemTensorNvOp):
            return f"{self._program_var_name}.gmem_tensor_ops[{self._quote(op.gmem_inout.name)}]"
        if op in self._op_var_names:
            return self._op_var_names[op]
        base = self._sanitize_identifier(op.name)
        var_name = self._allocate_name(base)
        self._op_var_names[op] = var_name
        return var_name

    def _format_scalar(self, value) -> str:
        if isinstance(value, str):
            return self._quote(value)
        return repr(value)

    def _format_layout_hint(self, hint: LayoutHint) -> str:
        return (
            "LayoutHint("
            f"Pr={hint.Pr}, Pc={hint.Pc}, Tr={hint.Tr}, Vr={hint.Vr}, "
            f"Tc={hint.Tc}, Vc={hint.Vc}, "
            f"Pmajor={self._quote(hint.Pmajor)}, "
            f"Tmajor={self._quote(hint.Tmajor)}, "
            f"Vmajor={self._quote(hint.Vmajor)}"
            ")"
        )

    def _format_swizzle(self, swizzle: SwizzleLayout) -> str:
        return f"SwizzleLayout(b={swizzle.b}, m={swizzle.m}, s={swizzle.s})"

    def _format_shape_ele(self, ele) -> str:
        if isinstance(ele, Shape):
            return self._format_shape(ele)
        if isinstance(ele, IntShape):
            return f"IntShape({ele.value})"
        if isinstance(ele, ParamShape):
            return f"ParamShape({self._quote(ele.param)})"
        if isinstance(ele, VarlenShape):
            return f"VarlenShape({self._quote(ele.name)})"
        if isinstance(ele, MnkShape):
            return f"MnkShape({self._quote(ele.param)}, {self._quote(ele.mnk)})"
        return repr(ele)

    def _format_shape(self, shape: Shape) -> str:
        if len(shape) == 0:
            return "Shape()"
        args = ", ".join(self._format_shape_ele(ele) for ele in shape)
        return f"Shape({args})"

    def _format_output_ref(self, out: NvOpOutput) -> str:
        producer = out.op
        if producer is None:
            return "None"
        if isinstance(producer, GmemTensorNvOp):
            key = producer.gmem_inout.name
            return f"{self._program_var_name}.gmem_tensor_ops[{self._quote(key)}].outputs[{out.idx}]"
        return f"{self._op_var(producer)}.outputs[{out.idx}]"

    def _format_source(self, source) -> str:
        if source is None:
            return "None"
        if isinstance(source, NvOpOutput):
            return self._format_output_ref(source)
        if isinstance(source, ForLoopStart):
            return f"ForLoopStart({self._op_var(source.for_loop)})"
        if isinstance(source, ForLoopEnd):
            return f"ForLoopEnd({self._op_var(source.for_loop)})"
        if isinstance(source, ForLoopCurrentIdx):
            return f"ForLoopCurrentIdx({self._op_var(source.for_loop)})"
        if isinstance(source, ForLoopCurrentIter):
            return f"ForLoopCurrentIter({self._op_var(source.for_loop)})"
        if isinstance(source, str):
            return self._quote(source)
        return str(source)

    def _format_tensor(self, tensor: NvOpTensor, *, include_source: bool) -> List[str]:
        args: List[str] = [
            f"shape={self._format_shape(tensor.shape)}",
            f"mem={self._quote(tensor.mem)}",
            f"dtype={self._quote(tensor.dtype)}",
        ]
        if include_source and tensor.source is not None:
            args.append(f"source={self._format_source(tensor.source)}")
        if tensor.row_major:
            args.append("row_major=True")
        if tensor.swizzle is not None:
            args.append(f"swizzle={self._format_swizzle(tensor.swizzle)}")
        return args

    def _format_mem_type(self, op: NvOp) -> str:
        src, dst = op.mem_type
        return f"({self._quote(src)}, {self._quote(dst)})"

    def _format_parameters(self, parameters: Dict[str, int]) -> str:
        if not parameters:
            return "{}"
        pairs = ", ".join(
            f"{self._quote(str(k))}: {self._format_scalar(v)}"
            for k, v in sorted(parameters.items())
        )
        return "{" + pairs + "}"

    def _format_impl_literal(self, code: str) -> List[str]:
        if not self.show_impl:
            return [self._quote("//! impl omitted by NvProgramPrinter")]

        code = code or ""
        raw_lines = code.splitlines()
        if self.impl_max_lines > 0 and len(raw_lines) > self.impl_max_lines:
            raw_lines = raw_lines[: self.impl_max_lines] + [
                f"//! ... {len(code.splitlines()) - self.impl_max_lines} more lines"
            ]

        truncated_code = "\n".join(raw_lines)
        if '"""' in truncated_code:
            return [repr(truncated_code)]

        if "\n" not in truncated_code:
            return [self._quote(truncated_code)]

        lines = truncated_code.splitlines()
        literal_lines = [f'r"""{lines[0]}']
        literal_lines.extend(lines[1:])
        literal_lines.append('"""')
        return literal_lines

    def _emit_tensor(self, tensor: NvOpTensor, indent_level: int, *, include_source: bool) -> None:
        self._emit(indent_level, "NvOpTensor(")
        for arg in self._format_tensor(tensor, include_source=include_source):
            self._emit(indent_level + 1, f"{arg},")
        self._emit(indent_level, ")")

    def _emit_input(self, inp: NvOpInput, indent_level: int) -> None:
        self._emit(indent_level, "NvOpInput(")
        self._emit(indent_level + 1, f"idx={inp.idx},")
        self._emit(indent_level + 1, f"name={self._quote(inp.name)},")
        self._emit(indent_level + 1, "tensor=")
        self._emit_tensor(inp.tensor, indent_level + 2, include_source=True)
        self._emit(indent_level + 1, ",")
        if inp.layout_hint is not None:
            self._emit(
                indent_level + 1,
                f"layout_hint={self._format_layout_hint(inp.layout_hint)},",
            )
        self._emit(indent_level, "),")

    def _emit_output(
        self,
        out: NvOpOutput,
        indent_level: int,
        *,
        include_origin: bool,
    ) -> None:
        self._emit(indent_level, "NvOpOutput(")
        self._emit(indent_level + 1, f"idx={out.idx},")
        self._emit(indent_level + 1, f"name={self._quote(out.name)},")
        self._emit(indent_level + 1, "tensor=")
        self._emit_tensor(out.tensor, indent_level + 2, include_source=False)
        self._emit(indent_level + 1, ",")

        if include_origin and out.origin is not None:
            self._emit(indent_level + 1, f"origin={self._format_output_ref(out.origin)},")
        if out.origin is None and out.unique:
            self._emit(indent_level + 1, "unique=True,")
        if out.layout_hint is not None:
            self._emit(
                indent_level + 1,
                f"layout_hint={self._format_layout_hint(out.layout_hint)},",
            )
        for key, value in sorted(out.attrs.items()):
            self._emit(
                indent_level + 1,
                f"{key}={self._format_scalar(value)},",
            )

        self._emit(indent_level, "),")

    def _emit_inputs(self, inputs: Sequence[NvOpInput], indent_level: int) -> None:
        self._emit(indent_level, "inputs=[")
        if self.show_io:
            for inp in inputs:
                self._emit_input(inp, indent_level + 1)
        self._emit(indent_level, "],")

    def _emit_outputs(
        self,
        outputs: Sequence[NvOpOutput],
        indent_level: int,
        *,
        include_origin: bool,
    ) -> None:
        self._emit(indent_level, "outputs=[")
        if self.show_io:
            for out in outputs:
                self._emit_output(out, indent_level + 1, include_origin=include_origin)
        self._emit(indent_level, "],")

    def _emit_constant_decl(self, op: ConstantNvOp, var_name: str, indent_level: int) -> None:
        self._emit(indent_level, f"{var_name} = ConstantNvOp(")
        self._emit(indent_level + 1, f"name={self._quote(op.name)},")
        self._emit(indent_level + 1, f"shape={self._format_shape(op.outputs[0].tensor.shape)},")
        self._emit(indent_level + 1, f"dtype={self._quote(op.outputs[0].tensor.dtype)},")
        self._emit(indent_level + 1, f"value={self._format_scalar(op.value)},")
        if op.parameters:
            self._emit(
                indent_level + 1,
                f"parameters={self._format_parameters(op.parameters)},",
            )
        self._emit(indent_level, ")")

    def _emit_nvop_decl(self, op: NvOp, var_name: str, indent_level: int) -> None:
        self._emit(indent_level, f"{var_name} = NvOp(")
        self._emit(indent_level + 1, f"name={self._quote(op.name)},")
        self._emit_inputs(op.inputs, indent_level + 1)
        self._emit_outputs(op.outputs, indent_level + 1, include_origin=True)

        self._emit(indent_level + 1, "impl=NvOpImpl(")
        for literal_line in self._format_impl_literal(op.impl.nv_src()):
            self._emit(indent_level + 2, literal_line)
        self._emit(indent_level + 1, "),")

        self._emit(indent_level + 1, f"mem_type={self._format_mem_type(op)},")
        if op.parameters:
            self._emit(
                indent_level + 1,
                f"parameters={self._format_parameters(op.parameters)},",
            )
        if op.op_type != NvOpType.UNKNOWN:
            self._emit(indent_level + 1, f"op_type=NvOpType.{op.op_type.name},")
        for key, value in sorted(op.attrs.items()):
            self._emit(indent_level + 1, f"{key}={self._format_scalar(value)},")
        self._emit(indent_level, ")")

    def _emit_for_loop_decl(self, op: ForLoopNvOp, var_name: str, indent_level: int) -> None:
        self._emit(indent_level, f"{var_name} = ForLoopNvOp(")
        self._emit(indent_level + 1, f"name={self._quote(op.name)},")
        if op.blk_idx_mapping is not None:
            self._emit(indent_level + 1, f"blk_idx_mapping={self._quote(op.blk_idx_mapping)},")

        self._emit(indent_level + 1, "loop_l=")
        self._emit_input(op.loop_l, indent_level + 2)
        self._emit(indent_level + 1, ",")

        self._emit(indent_level + 1, "loop_r=")
        self._emit_input(op.loop_r, indent_level + 2)
        self._emit(indent_level + 1, ",")

        self._emit(indent_level + 1, "loop_result=[")
        if self.show_io:
            for out in op.outputs:
                self._emit_output(out, indent_level + 2, include_origin=False)
        self._emit(indent_level + 1, "],")

        self._emit(indent_level + 1, "body=NvOpSequence(),")

        self._emit(indent_level + 1, "iter_args={")
        if self.show_io:
            for key, inp in op.iter_args.items():
                self._emit(indent_level + 2, f"{self._quote(key)}:")
                self._emit_input(inp, indent_level + 3)
                self._emit(indent_level + 2, ",")
        self._emit(indent_level + 1, "},")

        if op.parameters:
            self._emit(
                indent_level + 1,
                f"parameters={self._format_parameters(op.parameters)},",
            )
        for key, value in sorted(op.attrs.items()):
            self._emit(indent_level + 1, f"{key}={self._format_scalar(value)},")
        self._emit(indent_level, ")")

    def _emit_op_decl(self, op: NvOp, indent_level: int) -> None:
        if isinstance(op, GmemTensorNvOp):
            return

        var_name = self._op_var(op)
        if self._declared_ops.get(op, False):
            return
        self._declared_ops[op] = True

        if isinstance(op, ConstantNvOp):
            self._emit_constant_decl(op, var_name, indent_level)
            return
        if isinstance(op, ForLoopNvOp):
            self._emit_for_loop_decl(op, var_name, indent_level)
            return
        self._emit_nvop_decl(op, var_name, indent_level)

    def _emit_pipeline_plan(self, loop_op: ForLoopNvOp, indent_level: int) -> None:
        assert isinstance(loop_op.body, NvOpPipeline)
        loop_var = self._op_var(loop_op)
        stage_var_names: List[str] = []

        for stage_idx, stage in enumerate(loop_op.body.stages):
            stage_var = self._allocate_name(f"{loop_var}_stage_{stage_idx}")
            stage_var_names.append(stage_var)
            self._emit(indent_level, f"{stage_var} = NvOpSequence(")
            for stage_op in stage.ops:
                self._emit(indent_level + 1, f"{self._op_var(stage_op)},")
            self._emit(indent_level, ")")

        shifts_str = "[" + ", ".join(str(s) for s in loop_op.body.shifts) + "]"
        stages_str = ", ".join(stage_var_names)
        self._emit(
            indent_level,
            f"apply_pipeline({loop_var}, PipelinePlan([{stages_str}], {shifts_str}))",
        )

    def _emit_for_loop_set_results(self, op: ForLoopNvOp, indent_level: int) -> None:
        loop_var = self._op_var(op)
        for idx, out in enumerate(op.outputs):
            if out.origin is None:
                continue
            self._emit(
                indent_level,
                f"{loop_var}.set_loop_result({idx}, {self._format_output_ref(out.origin)})",
            )

    def _emit_op_recursive(
        self,
        op: NvOp,
        *,
        append_owner_expr: str,
        append_method: str,
        indent_level: int,
    ) -> None:
        if isinstance(op, GmemTensorNvOp):
            return

        self._emit_op_decl(op, indent_level)
        self._emit(
            indent_level,
            f"{append_owner_expr}.{append_method}({self._op_var(op)})",
        )

        if isinstance(op, ForLoopNvOp):
            if op.body.ops:
                self._emit_blank()
                self._emit(indent_level, f"# {self._op_var(op)} body")

                if isinstance(op.body, NvOpPipeline):
                    for body_op in op.body.ops:
                        self._emit_op_recursive(
                            body_op,
                            append_owner_expr=self._op_var(op),
                            append_method="append",
                            indent_level=indent_level,
                        )
                    self._emit_pipeline_plan(op, indent_level)
                else:
                    for body_op in op.body.ops:
                        self._emit_op_recursive(
                            body_op,
                            append_owner_expr=self._op_var(op),
                            append_method="append",
                            indent_level=indent_level,
                        )

            self._emit_for_loop_set_results(op, indent_level)

        self._emit_blank()

    def _emit_gmem_inout(self, name: str, io: GmemInout, indent_level: int) -> None:
        self._emit(indent_level, f"{self._quote(name)}: GmemInout(")
        self._emit(indent_level + 1, f"shape={self._format_shape(io.shape)},")
        self._emit(indent_level + 1, f"name={self._quote(io.name)},")
        self._emit(indent_level + 1, f"dtype={self._quote(io.dtype)},")
        self._emit(indent_level + 1, f"tensor_str={self._quote(io.tensor_str)},")
        self._emit(
            indent_level + 1,
            f"tensor_type_str={self._quote(io.tensor_type_str)},",
        )
        if io.parameters:
            self._emit(
                indent_level + 1,
                f"parameters={self._format_parameters(io.parameters)},",
            )
        self._emit(indent_level, "),")

    def dump_program(self, program: NvOpProgram) -> str:
        self._reset_state()
        self._program_var_name = self._program_var(program.name)

        self._emit(0, f"{self._program_var_name} = NvOpProgram(")
        self._emit(1, f"name={self._quote(program.name)},")
        self._emit(1, "gmem_inouts={")
        for name, io in program.gmem_inouts.items():
            self._emit_gmem_inout(name, io, indent_level=2)
        self._emit(1, "},")
        self._emit(0, ")")
        self._emit_blank()

        self._emit(0, "# program body")
        for op in program.ops:
            if isinstance(op, GmemTensorNvOp):
                continue
            self._emit_op_recursive(
                op,
                append_owner_expr=self._program_var_name,
                append_method="append_op",
                indent_level=0,
            )

        return "\n".join(self._lines).rstrip() + "\n"

    def print_program(self, program: NvOpProgram) -> None:
        print(self.dump_program(program))


class NvProgramDumper(NvProgramPrinter):
    pass
