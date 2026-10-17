from collections import Counter
from .op_ir import (
    Block,
    Op,
    ForLoopOp,
    LoopResultOp,
    ConstantOp,
    NoOp,
    ArrayRefOp,
    DeviceOp,
    MetaOp,
    ExternalSymbolOp,
    AddOp, 
    SubOp,
    MulOp,
    DivOp,
    PowOp,
    LoadOp,
    LoadOffsetOp,
    ArangeOp,
    Type,
    Value,
    OpOperand,
    IntType,
    FloatType,
    ArrayType,
    BlockArgument,
)
from .ops import (
    SparseOffsetLoadOp,
    CooAtomicFormatLoadOffOp,
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    ValSidxLoadOp,
    BValLoadOp,
    CooAtomicValRestoreOp,
    MmaOp,
    CValStoreOp,
)
from typing import Any, Sequence, Dict, Set, Optional

from sparsene.logging import get_logger

logger = get_logger(__name__)


class NameManager:
    class NameGenerator:
        def __init__(self):
            self.counter = Counter()

        def generate(self, name: Optional[str] = None) -> str:
            if name is None:
                new_name = f"v{self.counter[name]}"
                self.counter[name] += 1
            else:
                self.counter[name] += 1
                if self.counter[name] == 1:
                    new_name = name
                else:
                    new_name = f"{name}_{self.counter[name]}"
            return new_name

    def __init__(self):
        self.name_generator = NameManager.NameGenerator()
        self.name_map: Dict[Value, str] = {}
        self.used_names: Set[str] = set()

    def set_name(self, value: Value, name: str):
        self.name_map[value] = name

    def get_name(self, value: Value) -> str:
        if value not in self.name_map:
            name = value.name_hint
            while name is None or name in self.used_names:
                name = self.name_generator.generate(name)
                logger.debug(f"Generated name {name} for {value}")
            self.name_map[value] = name
            self.used_names.add(name)
        return self.name_map[value]


class Printer:
    def __init__(self, indent_size: int = 2, debug_indent: bool = False):
        self.name_manager = NameManager()
        self.debug_chars = "."

        self.indent_size = indent_size
        self.debug_indent = debug_indent

    def _get_indent(self, indent_level: int) -> str:
        return (
            (self.debug_chars if self.debug_indent else " ") * indent_level * self.indent_size
        )

    def _indent_lines(self, text: str, indent_level: int) -> str:
        """Helper method to indent all lines in a text string."""
        indent = self._get_indent(indent_level)
        return "\n".join(f"{indent}{line}" for line in text.split("\n"))

    def dump_for_loop_op(self, op: ForLoopOp, indent_level: int = 0) -> str:
        induction_var = self.dump_value(op.get_induction_var())
        start = self.dump_op_operand(op.operands[0])
        end = self.dump_op_operand(op.operands[1])
        iter_args = [
            f"{self.dump_value(arg)} = {self.dump_op_operand(operand, with_type=False)}"
            for arg, operand in zip(op.body.args[1:], op.operands[2:])
        ]
        body = self.dump_block(op.body)

        # Only split when there's a newline in the output
        return self._indent_lines(
            f"for_loop {induction_var} = {start} to {end} iter_args ({', '.join(iter_args)}) "
            f"{body}",
            indent_level,
        )

    def dump_loop_result_op(self, op: LoopResultOp, indent_level: int = 0) -> str:
        return self._indent_lines(
            f"loop_result {self.dump_op_operand(op.operands[0], with_type=False)}",
            indent_level,
        )

    def dump_constant_op(self, op: ConstantOp, indent_level: int = 0) -> str:
        value = op.attributes["value"]
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(f"constant {value}: {type_str}", indent_level)

    def dump_no_op(self, op: NoOp, indent_level: int = 0) -> str:
        return self._indent_lines("nop", indent_level)

    def dump_array_ref_op(self, op: ArrayRefOp, indent_level: int = 0) -> str:
        array = self.dump_op_operand(op.operands[0])
        indices = [self.dump_op_operand(index) for index in op.operands[1:]]
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(
            f"array_ref(array={array}, indices={', '.join(indices)}): {type_str}", indent_level
        )

    def dump_device_op(self, op: DeviceOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        other_attrs = {key: val for key, val in op.attributes.items() if key != "mem"}
        len_member_str = ""
        if hasattr(op, "len") and isinstance(getattr(op, "len"), Value):
            len_member_str = f" len={self.dump_value(getattr(op, 'len'), with_type=False)}"

        def _format_attr_value(value: Any) -> str:
            if isinstance(value, Value):
                return self.dump_value(value, with_type=False)
            return str(value)

        other_attrs_str = (
            " attr="
            + ", ".join(
                f"{key}={_format_attr_value(val)}" for key, val in other_attrs.items()
            )
            if other_attrs
            else ""
        )
        return self._indent_lines(
            f"{op.name}({', '.join(operands)}){len_member_str} mem={op.attributes['mem']}"
            f"{other_attrs_str}",
            indent_level,
        )

    def dump_meta_op(self, op: MetaOp, indent_level: int = 0) -> str:
        ops_str = "\n".join(
            [
                self._indent_lines(self.dump_op(op, indent_level + 1), indent_level + 1)
                for op in op.block.ops
            ]
        )
        args_str = ", ".join([self.dump_value(arg) for arg in op.block.args])

        # Only split when there's a newline in the output
        return self._indent_lines(f"{{\n" f"{ops_str}\n" f"}} args ({args_str})", indent_level)

    def dump_external_symbol_operation(
        self, op: ExternalSymbolOp, indent_level: int = 0
    ) -> str:
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(
            f"external @{op.attributes['symbol']}: {type_str}", indent_level
        )

    def dump_sparse_offset_load_op(self, op: SparseOffsetLoadOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"sparse_offset_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_coo_atomic_format_load_off_op(
        self, op: CooAtomicFormatLoadOffOp, indent_level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_off(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_coo_atomic_format_load_idx_op(
        self, op: CooAtomicFormatLoadIdxOp, indent_level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_idx(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_coo_atomic_format_load_val_op(
        self, op: CooAtomicFormatLoadValOp, indent_level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_val(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_val_sidx_load_op(self, op: ValSidxLoadOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"val_sidx_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_b_val_load_op(self, op: BValLoadOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"b_val_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_coo_atomic_val_restore_op(
        self, op: CooAtomicValRestoreOp, indent_level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_val_restore(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_mma_op(self, op: MmaOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"mma(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            indent_level,
        )

    def dump_c_val_store_op(self, op: CValStoreOp, indent_level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        return self._indent_lines(
            f"c_val_store(mem={op.attributes['mem']}, operands={', '.join(operands)})",
            indent_level,
        )

    def dump_add_op(self, op: AddOp, indent_level: int = 0) -> str:
        lhs = self.dump_op_operand(op.operands[0], with_type=False)
        rhs = self.dump_op_operand(op.operands[1], with_type=False)
        return self._indent_lines(f"add {lhs}, {rhs}", indent_level)

    def dump_mul_op(self, op: MulOp, indent_level: int = 0) -> str:
        lhs = self.dump_op_operand(op.operands[0], with_type=False)
        rhs = self.dump_op_operand(op.operands[1], with_type=False)
        return self._indent_lines(f"mul {lhs}, {rhs}", indent_level)

    def dump_sub_op(self, op: SubOp, indent_level: int = 0) -> str:
        lhs = self.dump_op_operand(op.operands[0], with_type=False)
        rhs = self.dump_op_operand(op.operands[1], with_type=False)
        return self._indent_lines(f"sub {lhs}, {rhs}", indent_level)

    def dump_div_op(self, op: DivOp, indent_level: int = 0) -> str:
        lhs = self.dump_op_operand(op.operands[0], with_type=False)
        rhs = self.dump_op_operand(op.operands[1], with_type=False)
        return self._indent_lines(f"div {lhs}, {rhs}", indent_level)

    def dump_pow_op(self, op: PowOp, indent_level: int = 0) -> str:
        lhs = self.dump_op_operand(op.operands[0], with_type=False)
        rhs = self.dump_op_operand(op.operands[1], with_type=False)
        return self._indent_lines(f"pow {lhs}, {rhs}", indent_level)

    def dump_load_op(self, op: LoadOp, indent_level: int = 0) -> str:
        array = self.dump_op_operand(op.operands[0])
        indices = [self.dump_op_operand(idx, with_type=False) for idx in op.operands[1:]]
        return self._indent_lines(
            f"load({array}, indices=([{', '.join(indices)}]))", indent_level
        )
    
    def dump_load_offset_op(self, op: LoadOffsetOp, indent_level: int = 0) -> str:
        # 1. 提取要访问的数组（通常打印时会带上 Type 信息，比如 array<int>[M/BLK_M + 1]）
        array = self.dump_op_operand(op.operands[0])
        
        # 2. 提取基准索引（例如 %i1）
        indices = [self.dump_op_operand(idx, with_type=False) for idx in op.operands[1:]]
        
        # 3. 格式化输出
        # 结果类似于: load_offset(%val_len_offset: array<int>[M/BLK_M + 1], indices=([%i1]))
        return self._indent_lines(
            f"load_offset({array}, indices=([{', '.join(indices)}]))", indent_level
        )

    def dump_arange_op(self, op: ArangeOp, indent_level: int = 0) -> str:
        # 1. 提取起始偏移的操作数（通常是第一个 operand）
        start = self.dump_op_operand(op.operands[0], with_type=False)
        
        # 2. 从结果的 ArrayType 中提取长度 (length)
        # 假设 ArangeOp 的结果类型是 ArrayType([length], IntType())
        assert isinstance(op.result.type, ArrayType)
        length = op.result.type.dims[0]
        
        # 3. 格式化输出：arange %start, length=BLK_K
        return self._indent_lines(f"arange ({start}, length={length})", indent_level)

    def dump_block(self, block: Block, indent_level: int = 0) -> str:
        ops_str = "\n".join(
            [self._indent_lines(self.dump_op(op), 1) for op in block.ops]
        )
        args_str = ", ".join([self.dump_value(arg) for arg in block.args])

        # Only split when there's a newline in the output
        return self._indent_lines(f"{{\n" f"{ops_str}\n}} args ({args_str})", indent_level)

    def dump_int_type(self, type: IntType) -> str:
        return "int"

    def dump_float_type(self, type: FloatType) -> str:
        return "float"

    def dump_array_type(self, type: ArrayType) -> str:
        dims = [str(dim) for dim in type.dims]
        return f"array<{self.dump_type(type.datatype)}>[{']['.join(dims)}]"

    def dump_value(self, value: Value, with_type: bool = True) -> str:
        #> 修改原来的dump value，他会讲
        if getattr(value, "defining_op", None) is None and value.name_hint == "_":
            return "_"
        type_str = self.dump_type(value.type)
        if with_type:
            return f"%{self.name_manager.get_name(value)}: {type_str}"
        else:
            return f"%{self.name_manager.get_name(value)}"

    def dump_op_operand(self, op_operand: OpOperand, with_type: bool = True) -> str:
        source = self.dump_value(op_operand.source, with_type)
        return source

    def dump_op(self, op: Op, indent_level: int = 0) -> str:
        if isinstance(op, ForLoopOp):
            op_str = self.dump_for_loop_op(op, 0)
        elif isinstance(op, ConstantOp):
            op_str = self.dump_constant_op(op, 0)
        elif isinstance(op, NoOp):
            op_str = self.dump_no_op(op, 0)
        elif isinstance(op, ArrayRefOp):
            op_str = self.dump_array_ref_op(op, 0)
        elif isinstance(op, DeviceOp):
            op_str = self.dump_device_op(op, 0)
        elif isinstance(op, MetaOp):
            op_str = self.dump_meta_op(op, 0)
        elif isinstance(op, ExternalSymbolOp):
            op_str = self.dump_external_symbol_operation(op, 0)
        elif isinstance(op, SparseOffsetLoadOp):
            op_str = self.dump_sparse_offset_load_op(op, 0)
        elif isinstance(op, CooAtomicFormatLoadOffOp):
            op_str = self.dump_coo_atomic_format_load_off_op(op, 0)
        elif isinstance(op, CooAtomicFormatLoadIdxOp):
            op_str = self.dump_coo_atomic_format_load_idx_op(op, 0)
        elif isinstance(op, CooAtomicFormatLoadValOp):
            op_str = self.dump_coo_atomic_format_load_val_op(op, 0)
        elif isinstance(op, ValSidxLoadOp):
            op_str = self.dump_val_sidx_load_op(op, 0)
        elif isinstance(op, BValLoadOp):
            op_str = self.dump_b_val_load_op(op, 0)
        elif isinstance(op, CooAtomicValRestoreOp):
            op_str = self.dump_coo_atomic_val_restore_op(op, 0)
        elif isinstance(op, MmaOp):
            op_str = self.dump_mma_op(op, 0)
        elif isinstance(op, CValStoreOp):
            op_str = self.dump_c_val_store_op(op, 0)
        elif isinstance(op, LoopResultOp):
            op_str = self.dump_loop_result_op(op, 0)
        elif isinstance(op, LoadOffsetOp):
            op_str = self.dump_load_offset_op(op, 0)
        elif isinstance(op, AddOp):
            op_str = self.dump_add_op(op, 0)
        elif isinstance(op, MulOp):
            op_str = self.dump_mul_op(op, 0)
        elif isinstance(op, SubOp):
            op_str = self.dump_sub_op(op, 0)
        elif isinstance(op, DivOp):
            op_str = self.dump_div_op(op, 0)
        elif isinstance(op, PowOp):
            op_str = self.dump_pow_op(op, 0)
        elif isinstance(op, LoadOp):
            op_str = self.dump_load_op(op, 0)
        elif isinstance(op, ArangeOp):
            op_str = self.dump_arange_op(op, 0)
        else:
            raise ValueError(f"Unknown op: {op}")

        prefix_str = (
            ", ".join(self.dump_value(result, with_type=False) for result in op.results)
            + " = "
        )
        result_types = [self.dump_type(result.type) for result in op.results]
        if op.num_results == 0:
            prefix_str = ""
        return_type_str = f" -> ({', '.join(result_types)})"
        if op.num_results == 0:
            return_type_str = ""
        elif isinstance(op, (ConstantOp, NoOp, ExternalSymbolOp, ArrayRefOp)):
            return_type_str = ""
        return self._indent_lines(
            f"{prefix_str}{op_str}{return_type_str}",
            indent_level,
        )

    def dump_type(self, type: Type) -> str:
        if isinstance(type, IntType):
            return self.dump_int_type(type)
        elif isinstance(type, FloatType):
            return self.dump_float_type(type)
        elif isinstance(type, ArrayType):
            return self.dump_array_type(type)
        else:
            raise ValueError(f"Unknown type: {type}")

    def dump_ops(self, ops: Sequence[Op], indent_level: int = 0) -> str:
        # Only split when there's a newline in the output
        #! change to this way
        body = "\n".join([
            self._indent_lines(self.dump_op(op, indent_level + 1), indent_level + 1)
            for op in ops
        ])
        return self._indent_lines(
            f"[\n{body}\n]",
            indent_level,
        )
        #! wrong at python 3.10.12
        # return self._indent_lines(
        #     f"[\n"
        #     f"{'\n'.join([self._indent_lines(self.dump_op(op, indent_level + 1), indent_level + 1) for op in ops])}\n"
        #     f"]",
        #     indent_level,
        # )

    def dump(self, obj: Any, indent_level: int = 0) -> str:
        if isinstance(obj, Op):
            return self.dump_op(obj, indent_level)
        elif isinstance(obj, Sequence) and all(isinstance(op, Op) for op in obj):
            return self.dump_ops(obj, indent_level)
        elif isinstance(obj, Type):
            return self.dump_type(obj)
        elif isinstance(obj, Block):
            return self.dump_block(obj, indent_level)
        elif isinstance(obj, Value):
            return self.dump_value(obj)
        elif isinstance(obj, OpOperand):
            return self.dump_op_operand(obj)
        else:
            raise ValueError(f"Unknown object: {obj}")


# Create a default printer instance for backward compatibility
default_printer = Printer(indent_size=3)
