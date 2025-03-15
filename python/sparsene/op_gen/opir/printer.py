from collections import Counter
from .op_ir import (
    Block,
    Op,
    ForLoopOp,
    ConstantOp,
    ArrayViewOp,
    NoOp,
    ArrayRefOp,
    DeviceOp,
    MetaOp,
    ExternalSymbolOperation,
    Type,
    Value,
    OpOperand,
    IntType,
    FloatType,
    RangeType,
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
from typing import Any, Sequence, Dict, Set, Optional, Tuple


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
            name = value.name_hint if isinstance(value, BlockArgument) else None
            while name is None or name in self.used_names:
                name = self.name_generator.generate(name)
                print(f"Generated name {name} for {value}")
            self.name_map[value] = name
            self.used_names.add(name)
        return self.name_map[value]


class Printer:
    def __init__(self, indent_level: int = 2, debug_indent: bool = False):
        self.name_manager = NameManager()
        self.debug_chars = "."

        self.indent_level = indent_level
        self.debug_indent = debug_indent

    def _get_indent(self, level: int) -> str:
        return (
            (self.debug_chars if self.debug_indent else " ") * level * self.indent_level
        )

    def _indent_lines(self, text: str, level: int) -> str:
        """Helper method to indent all lines in a text string."""
        indent = self._get_indent(level)
        return "\n".join(f"{indent}{line}" for line in text.split("\n"))

    def dump_for_loop_op(self, op: ForLoopOp, level: int = 0) -> str:
        induction_var = self.dump_value(op.get_induction_var())
        start = self.dump_op_operand(op.operands[0])
        end = self.dump_op_operand(op.operands[1])
        iter_args = [self.dump_value(arg) for arg in op.body.args[1:]]
        body = self.dump_block(op.body)

        # Only split when there's a newline in the output
        return self._indent_lines(
            f"for_loop {induction_var} = {start} to {end} iter_args ({', '.join(iter_args)})"
            f"{body}",
            level,
        )

    def dump_constant_op(self, op: ConstantOp, level: int = 0) -> str:
        value = op.attributes["value"]
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(f"constant {value}: {type_str}", level)

    def dump_array_view_op(self, op: ArrayViewOp, level: int = 0) -> str:
        array = self.dump_op_operand(op.operands[0])
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(f"array_view(array={array}): {type_str}", level)

    def dump_no_op(self, op: NoOp, level: int = 0) -> str:
        return self._indent_lines("nop", level)

    def dump_array_ref_op(self, op: ArrayRefOp, level: int = 0) -> str:
        array = self.dump_op_operand(op.operands[0])
        indices = [self.dump_op_operand(index) for index in op.operands[1:]]
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(
            f"array_ref(array={array}, indices={', '.join(indices)}): {type_str}", level
        )

    def dump_device_op(self, op: DeviceOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        return self._indent_lines(
            f"{op.name}({', '.join(operands)}) attr={op.attributes}",
            level,
        )

    def dump_meta_op(self, op: MetaOp, level: int = 0) -> str:
        ops_str = "\n".join(
            [
                self._indent_lines(self.dump_op(op, level + 1), level + 1)
                for op in op.block.ops
            ]
        )
        args_str = ", ".join([self.dump_value(arg) for arg in op.block.args])

        # Only split when there's a newline in the output
        return self._indent_lines(
            f"{{\n" f"{ops_str}\n"f"}}"  f"args ({args_str})" , level
        )

    def dump_external_symbol_operation(
        self, op: ExternalSymbolOperation, level: int = 0
    ) -> str:
        type_str = self.dump_type(op.result.type)
        return self._indent_lines(
            f"external @{op.attributes['symbol']}: {type_str}", level
        )

    def dump_sparse_offset_load_op(self, op: SparseOffsetLoadOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"sparse_offset_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_coo_atomic_format_load_off_op(
        self, op: CooAtomicFormatLoadOffOp, level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_off(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_coo_atomic_format_load_idx_op(
        self, op: CooAtomicFormatLoadIdxOp, level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_idx(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_coo_atomic_format_load_val_op(
        self, op: CooAtomicFormatLoadValOp, level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_format_load_val(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_val_sidx_load_op(self, op: ValSidxLoadOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"val_sidx_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_b_val_load_op(self, op: BValLoadOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"b_val_load(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_coo_atomic_val_restore_op(
        self, op: CooAtomicValRestoreOp, level: int = 0
    ) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"coo_atomic_val_restore(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_mma_op(self, op: MmaOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        results = [self.dump_value(result) for result in op.results]
        return self._indent_lines(
            f"mma(mem={op.attributes['mem']}, operands={', '.join(operands)}): {', '.join(results)}",
            level,
        )

    def dump_c_val_store_op(self, op: CValStoreOp, level: int = 0) -> str:
        operands = [self.dump_op_operand(operand) for operand in op.operands]
        return self._indent_lines(
            f"c_val_store(mem={op.attributes['mem']}, operands={', '.join(operands)})",
            level,
        )

    def dump_block(self, block: Block, level: int = 0) -> str:
        ops_str = "\n".join(
            [self._indent_lines(self.dump_op(op), 1) for op in block.ops]
        )
        args_str = ", ".join([self.dump_value(arg) for arg in block.args])

        # Only split when there's a newline in the output
        return self._indent_lines(f"{{\n" f"{ops_str}\n}} args ({args_str})", level)

    def dump_int_type(self, type: IntType) -> str:
        return "int"

    def dump_float_type(self, type: FloatType) -> str:
        return "float"

    def dump_range_type(self, type: RangeType) -> str:
        return "range"

    def dump_array_type(self, type: ArrayType) -> str:
        dims = [str(dim) for dim in type.dims]
        return f"array<{self.dump_type(type.datatype)}>[{']['.join(dims)}]"

    def dump_value(self, value: Value, with_type: bool = True) -> str:
        type_str = self.dump_type(value.type)
        if with_type:
            return f"%{self.name_manager.get_name(value)}: {type_str}"
        else:
            return f"%{self.name_manager.get_name(value)}"

    def dump_op_operand(self, op_operand: OpOperand) -> str:
        source = self.dump_value(op_operand.source)
        return source

    def dump_op(self, op: Op, level: int = 0) -> str:
        if isinstance(op, ForLoopOp):
            op_str = self.dump_for_loop_op(op, 0)
        elif isinstance(op, ConstantOp):
            op_str = self.dump_constant_op(op, 0)
        elif isinstance(op, ArrayViewOp):
            op_str = self.dump_array_view_op(op, 0)
        elif isinstance(op, NoOp):
            op_str = self.dump_no_op(op, 0)
        elif isinstance(op, ArrayRefOp):
            op_str = self.dump_array_ref_op(op, 0)
        elif isinstance(op, DeviceOp):
            op_str = self.dump_device_op(op, 0)
        elif isinstance(op, MetaOp):
            op_str = self.dump_meta_op(op, 0)
        elif isinstance(op, ExternalSymbolOperation):
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
        else:
            raise ValueError(f"Unknown op: {op}")
        
        prefix = ", ".join(self.dump_value(result, with_type=False) for result in op.results) + " = "
        result_types = [self.dump_type(result.type) for result in op.results]
        if op.num_results == 0:
            prefix = ""
        return self._indent_lines(f"{prefix}{op_str} -> ({', '.join(result_types)})", level)

    def dump_type(self, type: Type) -> str:
        if isinstance(type, IntType):
            return self.dump_int_type(type)
        elif isinstance(type, FloatType):
            return self.dump_float_type(type)
        elif isinstance(type, RangeType):
            return self.dump_range_type(type)
        elif isinstance(type, ArrayType):
            return self.dump_array_type(type)
        else:
            raise ValueError(f"Unknown type: {type}")

    def dump_ops(self, ops: Sequence[Op], level: int = 0) -> str:
        # Only split when there's a newline in the output
        return self._indent_lines(
            f"[\n"
            f"{'\n'.join([self._indent_lines(self.dump_op(op, level + 1), level + 1) for op in ops])}\n"
            f"]",
            level,
        )

    def dump(self, obj: Any, level: int = 0) -> str:
        if isinstance(obj, Op):
            return self.dump_op(obj, level)
        elif isinstance(obj, Sequence) and all(isinstance(op, Op) for op in obj):
            return self.dump_ops(obj, level)
        elif isinstance(obj, Type):
            return self.dump_type(obj)
        elif isinstance(obj, Block):
            return self.dump_block(obj, level)
        elif isinstance(obj, Value):
            return self.dump_value(obj)
        elif isinstance(obj, OpOperand):
            return self.dump_op_operand(obj)
        else:
            raise ValueError(f"Unknown object: {obj}")


# Create a default printer instance for backward compatibility
default_printer = Printer(indent_level=3)
