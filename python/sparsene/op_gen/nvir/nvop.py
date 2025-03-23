from typing import Literal, List
from sparsene.op_gen.opir.op_ir import Op
from sparsene.op_gen.nvir.op_impl import NvOpImpl
from itertools import chain

IoMemType = Literal["gmem", "smem", "rmem"]
OpMemType = Literal[
    ("r", "s"),
    ("s", "r"),
    ("g", "s"),
    ("s", "g"),
    ("g", "r"),
    ("r", "g"),
]


class NvOpInput:
    idx: int
    shape: List[int | str]
    mem: IoMemType
    name: str
    dtype: str

    def __init__(
        self, idx: int, shape: List[int | str], mem: IoMemType, name: str, dtype: str
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype


class NvOpOutput:
    idx: int
    shape: List[int | str]
    mem: IoMemType
    name: str
    dtype: str

    def __init__(
        self, idx: int, shape: List[int | str], mem: IoMemType, name: str, dtype: str
    ):
        self.idx = idx
        self.shape = shape
        self.mem = mem
        self.name = name
        self.dtype = dtype


class NvOp:
    name: str

    inputs: List[NvOpInput]
    outputs: List[NvOpOutput]
    backend_op: Op
    impl: NvOpImpl

    mem_type: OpMemType

    parameters: List[str]

    def __init__(
        self,
        name: str,
        inputs: List[NvOpInput],
        outputs: List[NvOpOutput],
        impl: NvOpImpl,
        mem_type: OpMemType,
        # backend_op: Op,
        # parameters: List[str],
    ):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.impl = impl
        self.mem_type = mem_type
        # self.backend_op = backend_op
        # self.parameters = parameters


class NvOpPrinter:
    def __init__(self, indent_size: int = 3, debug_indent: bool = False):
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

    def dump_nvop(self, op: NvOp, indent_level: int = 0) -> str:
        # in_types_strs = [
        #     f"class TI{idx} = {inp.dtype}"
        #     for idx, inp in enumerate(op.inputs)
        # ]
        # out_types_strs = [
        #     f"class TO{idx} = {out.dtype}"
        #     for idx, out in enumerate(op.outputs)
        # ]
        # io_types_str = ", " + ", ".join(chain(in_types_strs, out_types_strs))

        # template_head_str = f"template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK{io_types_str}>"
        template_head_str = (
            f"template<class NBUF, class BLK_MNK, class MMA_MNK, class BLK_MMA_MNK>"
        )

        class_head_str = f"class {op.name} {{\npublic:"

        inputs_str = f"// Inputs\n" + "\n".join(
            map(lambda inp: self.dump_input_decl(inp), op.inputs)
        )
        outputs_str = f"// Outputs\n" + "\n".join(
            map(lambda out: self.dump_output_decl(out), op.outputs)
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
                if isinstance(s, str):
                    varlen_modes_constructor_params.append(f"int mode_o{out.idx}_{i}")
                    i += 1
        for inp in op.inputs:
            i: int = 0
            for s in inp.shape:
                if isinstance(s, str):
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
        varlen_shapes_constructor_init_list = [
            self.dump_init_input_varlen_shape(idx, inp.shape)
            for idx, inp in enumerate(op.inputs)
        ] + [
            self.dump_init_output_varlen_shape(idx, out.shape)
            for idx, out in enumerate(op.outputs)
        ]
        output_tensor_constructor_init_list = [
            f"{out.name}(make_tensor(make_{out.mem}_ptr({out.name}), Shape_o{out.idx}))"
            for out in op.outputs
        ]

        constructor_init_list = (
            hw_param_constructor_init_list
            + output_tensor_constructor_init_list
            + varlen_shapes_constructor_init_list
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
                f_input_list.append(f"{inp.dtype} {inp.name}")
        f_input_list_str = ", ".join(f_input_list)
        return (
            f"{f_template_head_str}\n"
            + f"CUTE_DEVICE void f({f_input_list_str}) {{\n"
            + self._indent_lines(code_str)
            + "\n"
            + "}"
        )

    def dump_get_output_method_switch(self, idx: int, indent_level: int = 0) -> str:
        return (
            f"if constexpr (output_idx == {idx}) {{\n"
            + self._indent_lines(f"return {op.name}(_, buf_idx);")
            + "\n"
            + "}"
        )

    def dump_get_output_method(self, op: NvOp, indent_level: int = 0) -> str:
        return (
            "template<size_t buf_idx, size_t output_idx>\n"
            + "CUTE_DEVICE auto output() {\n"
            + "\n".join(
                self._indent_lines(self.dump_get_output_method_switch(idx), 1)
                for idx in range(len(op.outputs))
            )
            + "\n"
            + "}"
        )

    def dump_input_decl(self, inp: NvOpInput, indent_level: int = 0) -> str:
        shape_type_str = self.dump_decl_input_shape_type(inp.idx, inp.shape)
        tensor_type_str = self.dump_decl_input_tensor_type(inp.idx, inp.mem, inp.dtype)
        shape_str = self.dump_decl_input_shape(inp.idx)
        return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}"

    def dump_decl_input_shape_type(
        self, idx: int, shape: List[int | str], indent_level: int = 0
    ) -> str:
        comma_seperated_shapes = ", ".join(
            map(lambda s: f"Int<{s}>{{}}" if isinstance(s, int) else "VARLEN", shape)
        )
        decl_shape_str = f"decltype(make_shape({comma_seperated_shapes}))"
        return f"using Shape_i{idx} = {decl_shape_str};"

    def dump_decl_input_tensor_type(
        self, idx: int, mem: IoMemType, dtype: str, indent_level: int = 0
    ) -> str:
        match mem:
            case "gmem":
                decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr(({dtype}*)nullptr), Shape_i{idx}))"
            case "smem":
                decl_tensor_str = f"decltype(make_tensor(make_smem_ptr(({dtype}*)nullptr), Shape_i{idx}))"
            case "rmem":
                decl_tensor_str = f"decltype(make_tensor(make_rmem_ptr(({dtype}*)nullptr), Shape_i{idx}))"
        return f"using Tensor_i{idx} = {decl_tensor_str};"

    def dump_decl_input_shape(self, idx: int) -> str:
        return f"Shape_i{idx} shape_i{idx};"

    def dump_output_decl(self, out: NvOpOutput, indent_level: int = 0) -> str:
        shape_type_str = self.dump_decl_output_shape_type(out.idx, out.shape)
        tensor_type_str = self.dump_decl_output_tensor_type(out.idx, out.mem, out.dtype)
        shape_str = self.dump_decl_output_shape(out.idx)
        tensor_str = self.dump_decl_output_tensor(out)
        return f"{shape_type_str}\n{tensor_type_str}\n{shape_str}\n{tensor_str}"

    def dump_decl_output_shape_type(
        self, idx: int, shape: List[int | str], indent_level: int = 0
    ) -> str:
        shapes = [f"Int<{s}>{{}}" if isinstance(s, int) else "VARLEN" for s in shape]
        shapes.append("NBUF{}")
        comma_seperated_shapes = ", ".join(shapes)
        decl_shape_str = f"decltype(make_shape({comma_seperated_shapes}))"
        return f"using Shape_o{idx} = {decl_shape_str};"

    def dump_decl_output_tensor_type(
        self, idx: int, mem: IoMemType, dtype: str, indent_level: int = 0
    ) -> str:
        match mem:
            case "gmem":
                decl_tensor_str = f"decltype(make_tensor(make_gmem_ptr(({dtype}*)nullptr), Shape_o{idx}))"
            case "smem":
                decl_tensor_str = f"decltype(make_tensor(make_smem_ptr(({dtype}*)nullptr), Shape_o{idx}))"
            case "rmem":
                decl_tensor_str = f"decltype(make_tensor<{dtype}>(Shape_o{idx}))"
        return f"using Tensor_o{idx} = {decl_tensor_str};"

    def dump_decl_output_shape(self, idx: int) -> str:
        return f"Shape_o{idx} shape_o{idx};"

    def dump_decl_output_tensor(self, out: NvOpOutput) -> str:
        return f"Tensor_o{out.idx} {out.name};"

    def dump_init_input_varlen_shape(
        self, idx: int, shape: List[int | str], indent_level: int = 0
    ) -> str:
        i: int = 0
        shape = list(shape)
        for s in shape:
            if isinstance(s, str):
                shape[i] = f"mode_i{idx}_{i}"
                i += 1
        comma_seperated_shapes = ", ".join(
            map(lambda s: f"Int<{s}>{{}}" if isinstance(s, int) else s, shape)
        )
        return f"shape_i{idx}(make_shape({comma_seperated_shapes}))"

    def dump_init_output_varlen_shape(
        self, idx: int, shape: List[int | str], indent_level: int = 0
    ) -> str:
        i: int = 0
        shape = list(shape)
        for s in shape:
            if isinstance(s, str):
                shape[i] = f"mode_o{idx}_{i}"
                i += 1
        shape.append("NBUF{}")
        comma_seperated_shapes = ", ".join(
            map(lambda s: f"Int<{s}>{{}}" if isinstance(s, int) else s, shape)
        )
        return f"shape_o{idx}(make_shape({comma_seperated_shapes}))"


if __name__ == "__main__":
    op = NvOp(
        name="G2sValSidxLoadOp",
        inputs=[
            NvOpInput(
                idx=0, shape=[16, "nnz_blk"], mem="gmem", name="dval_sidx", dtype="int"
            )
        ],
        outputs=[
            NvOpOutput(
                idx=0, shape=[16], mem="smem", name="tile_sidx_block", dtype="int"
            )
        ],
        impl=NvOpImpl(
            code_template
="""if (lid < 4) {
    auto src = local_tile(dval_sidx(_, i), thr_tiler, thr_coord);
    auto dst = local_tile(tile_sidx_block(_, buf_idx), thr_tiler, thr_coord);
    copy(Copy_Atom<UniversalCopy<uint128_t>, int>{}, src, dst);
}"""
        ),
        mem_type=("g", "s"),
    )
    print(NvOpPrinter().dump_nvop(op))
