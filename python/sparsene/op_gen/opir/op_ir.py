from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, List, Dict, Tuple, overload, Sequence, Optional

from sparsene.format.format import Expr, Symbol, Number
from sparsene.logging import get_logger

logger = get_logger(__name__)


class OpBuilder:
    block_stack: List[Block]

    def __init__(self):
        self.block_stack = []

    def push_block(self, block: Block):
        self.block_stack.append(block)

    def pop_block(self) -> Block:
        return self.block_stack.pop()

    @contextmanager
    def block_scope(self, block: Block):
        self.push_block(block)
        try:
            yield block
        finally:
            self.pop_block()

    def op_scope(self, op: MetaOp | ForLoopOp):
        return self.block_scope(op.block)

    @property
    def current_block(self) -> Optional[Block]:
        return self.block_stack[-1] if self.block_stack else None

    @overload
    def build(
        self,
        op_type: type[ForLoopOp],
        induction_var: str,
        range: Tuple[Value, Value],
        iter_args: Dict[str, Value] = {},
    ) -> ForLoopOp: ...
    @overload
    def build(
        self, op_type: type[ConstantOp], value: Expr | int | str, type: Type
    ) -> ConstantOp: ...
    @overload
    def build(self, op_type: type[NoOp]) -> NoOp: ...
    @overload
    def build(self, op_type: type[ArrayRefOp], *args, **kwargs) -> ArrayRefOp: ...
    @overload
    def build(
        self, op_type: type[DeviceOp], mem: str, operands: Sequence[Value]
    ) -> DeviceOp: ...
    @overload
    def build(self, op_type: type[MetaOp]) -> MetaOp: ...
    @overload
    def build(
        self, op_type: type[ExternalSymbolOperation], symbol: str, type: Type
    ) -> ExternalSymbolOperation: ...
    @overload
    def build(self, op_type: type[LoopResultOp], op: Op) -> LoopResultOp: ...

    def build(
        self,
        op_type: type[Op],
        *args,
        **kwargs,
    ) -> Op:
        op = op_type(*args, **kwargs)
        op.residing_block = self.current_block
        if self.current_block is not None:
            self.current_block.add_op(op)
        return op

    @property
    def symbol_table(self) -> SymbolTable:
        if self.current_block is not None:
            return self.current_block.symbol_table
        else:
            raise ValueError("No symbol table available: not in a block. ")

    def lookup_symbol(self, name: str, type: Optional[Type] = None) -> Value:
        if (
            self.current_block is not None
            and (value := self.current_block.symbol_table[name]) is not None
        ):
            return value
        elif type is not None:
            return self.build(ExternalSymbolOperation, name, type).result
        else:
            raise ValueError(f"Symbol {name} not found")

    def add_symbol(self, name: str, value: Value):
        if self.current_block is not None:
            self.current_block.symbol_table.add_symbol(name, value)
        else:
            raise ValueError("No symbol table available: not in a block. ")


@dataclass
class Type:
    pass


@dataclass
class IntType(Type):
    pass


@dataclass
class FloatType(Type):
    pass


@dataclass
class RangeType(Type):
    pass


@dataclass
class ArrayType(Type):
    dims: List[Expr]
    datatype: Type

    # return f"{self.element_type}[{']['.join(str(dim) for dim in self.dims)}]"
    # return f"array[{']['.join(str(dim) for dim in self.dims)}]"


@dataclass(frozen=True)
class Value:
    type: Type
    name_hint: Optional[str]

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass(frozen=True)
class OpResult(Value):
    defining_op: Op
    result_idx_in_owner: int

    def __init__(self, type: Type, defining_op: Op, result_idx_in_owner: int, name_hint: Optional[str] = None):
        super().__init__(type)
        object.__setattr__(self, "defining_op", defining_op)
        object.__setattr__(self, "result_idx_in_owner", result_idx_in_owner)
        object.__setattr__(self, "name_hint", name_hint)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass(frozen=True)
class BlockArgument(Value):
    owner: Block
    name_hint: Optional[str]

    def __init__(self, type: Type, owner: Block, name_hint: Optional[str] = None):
        super().__init__(type)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "name_hint", name_hint)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass(frozen=True)
class OpOperand:
    source: Value


class Op(ABC):
    attributes: Dict[str, Any]
    operands: List[OpOperand]
    results: List[OpResult]
    residing_block: Optional[Block]

    def __init__(
        self,
        operands: Sequence[Value] = [],
        results: Sequence[OpResult] = [],
        attributes: Dict[str, Any] = {},
        residing_block: Optional[Block] = None,
    ):
        self.attributes = {**attributes}
        self.operands = [
            operand if isinstance(operand, OpOperand) else OpOperand(operand)
            for operand in operands
        ]
        self.results = list(results)
        self.residing_block = residing_block

    def add_attribute(self, name: str, value: Any):
        if name in self.attributes:
            raise ValueError(f"Attribute {name} already exists")
        self.attributes[name] = value

    def set_attribute(self, name: str, value: Any):
        self.attributes[name] = value

    @property
    def num_operands(self) -> int:
        return len(self.operands)

    @property
    def num_results(self) -> int:
        return len(self.results)

    def add_operand(self, operand: OpOperand):
        self.operands.append(operand)

    def add_result(self, result: OpResult):
        self.results.append(result)

    def has_single_result(self) -> bool:
        return len(self.results) == 1

    def get_result(self) -> OpResult:
        assert self.has_single_result()
        return self.results[0]

    def get_results(self) -> List[OpResult]:
        return list(self.results)

    @property
    def result(self) -> OpResult:
        return self.get_result()

    @property
    @abstractmethod
    def name(self) -> str: ...


class ExternalSymbolOperation(Op):

    def __init__(self, symbol: str, type: Type):
        super().__init__(
            results=[
                OpResult(type=type, defining_op=self, result_idx_in_owner=0),
            ],
            attributes={"symbol": symbol},
        )

    @property
    def name(self) -> str:
        return "external_symbol"


class ConstantOp(Op):

    def __init__(self, value: Expr | int | str, type: Type):
        if isinstance(value, str):
            value = Symbol(value)
        elif isinstance(value, int):
            value = Number(value)
        super().__init__(
            results=[
                OpResult(type=type, defining_op=self, result_idx_in_owner=0),
            ],
            attributes={"value": value},
        )

    @property
    def name(self) -> str:
        return "constant"


class NoOp(Op):
    def __init__(self):
        super().__init__(
            attributes={},
            operands=[],
            results=[],
        )

    @property
    def name(self) -> str:
        return "nop"


class ArrayRefOp(Op):
    def __init__(self, array: Value, indices: Sequence[Value]):
        assert isinstance(array.type, ArrayType)
        super().__init__(
            operands=[array] + list(indices),
            results=[
                OpResult(
                    type=ArrayType(
                        array.type.dims[len(indices) :], array.type.datatype
                    ),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "array_ref"


class DeviceOp(Op):
    """
    Operation that is likely to be executed as a device function.

    Attributes:
        mem: Literal["G2S", "G2R", "S2R", "R2S", "R2G", "S2G", "R2R"]
    """

    def __init__(
        self, mem: str, operands: Sequence[Value], results: Sequence[OpResult]
    ):
        super().__init__(operands, results)
        self.set_attribute("mem", mem)


class Block:
    ops: List[Op]
    args: List[BlockArgument]
    symbol_table: SymbolTable
    owning_op: Op

    def __init__(self, owner: Op):
        self.ops = []
        self.args = []
        self.owning_op = owner
        self.symbol_table = SymbolTable(self)

    def add_op(self, op: Op):
        self.ops.append(op)


class BlockOp(Op):
    block: Block

    def __init__(
        self,
        operands: Sequence[Value] = [],
        results: Sequence[OpResult] = [],
        attributes: Dict[str, Any] = {},
        residing_block: Optional[Block] = None,
    ):
        super().__init__(
            operands=operands,
            results=results,
            attributes=attributes,
            residing_block=residing_block,
        )
        self.block = Block(self)


class MetaOp(BlockOp):

    @property
    def name(self) -> str:
        return "meta"


class ForLoopOp(BlockOp):
    @property
    def body(self) -> Block:
        return self.block

    def __init__(
        self,
        induction_var: str,
        range: Tuple[Value, Value],
        iter_args: Dict[str, Value] = {},
    ):
        super().__init__(
            attributes={},
            operands=[range[0], range[1]] + list(iter_args.values()),
            results=[
                OpResult(
                    type=arg.type,
                    defining_op=self,
                    result_idx_in_owner=i,
                )
                for i, arg in enumerate(iter_args.values())
            ],
        )
        self.body.args.append(BlockArgument(IntType(), self.body, induction_var))
        for name, arg in iter_args.items():
            self.body.args.append(BlockArgument(arg.type, self.body, name))

    @property
    def name(self) -> str:
        return "for_loop"

    def get_induction_var(self) -> BlockArgument:
        return self.body.args[0]

    def get_iter_arg(self, idx: int) -> BlockArgument:
        return self.body.args[idx + 1]


class LoopResultOp(Op):
    def __init__(self, op: Op):
        super().__init__(
            operands=op.results,
            results=[],
        )

    @property
    def name(self) -> str:
        return "loop_result"


class SymbolTable:
    def __init__(self, block: Block):
        self.symbol_table: Dict[str, Value] = {}
        self.block = block

    def add_symbol(self, name: str, value: Value):
        self.symbol_table[name] = value
        return value

    def lookup_symbol(self, name: str) -> Optional[Value]:
        if name in self.symbol_table:
            return self.symbol_table[name]
        elif self.block.owning_op.residing_block is not None:
            return self.block.owning_op.residing_block.symbol_table[name]
        else:
            return None

    def __setitem__(self, name: str, value: Value):
        self.add_symbol(name, value)

    def __getitem__(self, name: str) -> Optional[Value]:
        return self.lookup_symbol(name)


op_builder = OpBuilder()
