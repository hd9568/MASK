from __future__ import annotations

from sparsene.format.format import Expr, Symbol, Number
from typing import Any, List, Dict, Tuple, overload, Sequence, Optional
from dataclasses import dataclass
from copy import deepcopy
from abc import ABC, abstractmethod


class OpIrContext:
    type_cache: Dict[type, Type] = {}

    @overload
    def get_type(self, t: type[IndexType]) -> IndexType: ...
    @overload
    def get_type(self, t: type[RangeType]) -> RangeType: ...
    @overload
    def get_type(self, t: type[ArrayType], dims: List[Expr]) -> ArrayType: ...

    def get_type(self, t: type, *args, **kwargs) -> Type:
        return t(*args, **kwargs)
        # if t not in self.type_cache:
        #     self.type_cache[t] = t(*args, **kwargs)
        # return self.type_cache[t]


class OpBuilder:
    def __init__(self):
        pass

    @overload
    def build(
        self,
        op_type: type[ForLoopOp],
        induction_var: str,
        range: Tuple[Value, Value],
        iter_args: Dict[str, Value],
    ) -> ForLoopOp: ...
    @overload
    def build(
        self, op_type: type[ConstantOp], value: Expr | int | str, type: Type
    ) -> ConstantOp: ...
    @overload
    def build(self, op_type: type[ArrayViewOp], *args, **kwargs) -> ArrayViewOp: ...
    @overload
    def build(self, op_type: type[NoOp]) -> NoOp: ...
    @overload
    def build(self, op_type: type[ArrayRefOp], *args, **kwargs) -> ArrayRefOp: ...
    @overload
    def build(self, op_type: type[DeviceOp], *args, **kwargs) -> DeviceOp: ...
    @overload
    def build(self, op_type: type[MetaOp]) -> MetaOp: ...
    @overload
    def build(
        self, op_type: type[ExternalSymbolOperation], symbol: str, type: Type
    ) -> ExternalSymbolOperation: ...

    def build(
        self,
        op_type: type[Op],
        *args,
        **kwargs,
    ) -> Op:
        op = op_type(*args, **kwargs)
        return op


@dataclass
class Type:
    @abstractmethod
    def __str__(self) -> str: ...


@dataclass
class IndexType(Type):
    def __str__(self) -> str:
        return "index"


@dataclass
class RangeType(Type):
    def __str__(self) -> str:
        return "range"


@dataclass
class ArrayType(Type):
    dims: List[Expr]
    # element_type: Type

    def __str__(self) -> str:
        # return f"{self.element_type}[{']['.join(str(dim) for dim in self.dims)}]"
        return f"array[{']['.join(str(dim) for dim in self.dims)}]"


@dataclass(frozen=True, init=False)
class Value:
    name: str
    type: Type
    defining_op: Op
    owner: Optional[Block]
    result_idx_in_owner: int
    is_defined_by_op: bool

    def __init__(
        self,
        name: str,
        type: Type,
        defining_op_or_owner: Op | Block,
        result_idx_in_owner: int,
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", type)
        if isinstance(defining_op_or_owner, Op):
            object.__setattr__(self, "owner", None)
            object.__setattr__(self, "defining_op", defining_op_or_owner)
            object.__setattr__(self, "defined_by_op", True)
        else:
            object.__setattr__(self, "owner", defining_op_or_owner)
            object.__setattr__(self, "defined_by_op", False)
        object.__setattr__(self, "result_idx_in_owner", result_idx_in_owner)

    def __repr__(self):
        return f"Value(name={self.name}, type={self.type}, owner={self.owner }, result_idx_in_owner={self.result_idx_in_owner})"

    def __str__(self) -> str:
        return f"{self.name}: {self.type}"


@dataclass(frozen=True)
class OpOperand:
    source: Value

    def __str__(self) -> str:
        return f"{self.source}"


class Op(ABC):
    attributes: Dict[str, Any]
    operands: List[OpOperand]
    results: List[Value]
    symbol_table: SymbolTable

    def __init__(
        self,
        operands: Sequence[OpOperand | Value] = [],
        results: Sequence[Value] = [],
        attributes: Dict[str, Any] = {},
    ):
        self.attributes = {**attributes}
        self.operands = [
            operand if isinstance(operand, OpOperand) else OpOperand(operand)
            for operand in operands
        ]
        self.results = list(results)
        self.symbol_table = SymbolTable(self)

    def add_attribute(self, name: str, value: Any):
        if name in self.attributes:
            raise ValueError(f"Attribute {name} already exists")
        self.attributes[name] = value

    def set_attribute(self, name: str, value: Any):
        self.attributes[name] = value

    def add_operand(self, operand: OpOperand):
        self.operands.append(operand)

    def add_result(self, result: Value):
        self.results.append(result)

    def has_single_result(self) -> bool:
        return len(self.results) == 1

    def get_result(self) -> Value:
        assert self.has_single_result()
        return self.results[0]

    def get_results(self) -> List[Value]:
        return list(self.results)

    @property
    def result(self) -> Value:
        return self.get_result()

    def __repr__(self):
        return f"{self.name} {self.operands} -> {self.results}"

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __str__(self) -> str: ...


class ExternalSymbolOperation(Op):

    def __init__(self, symbol: str, type: Type):
        super().__init__(
            results=[
                Value(
                    name="external_symbol",
                    type=type,
                    defining_op_or_owner=self,
                    result_idx_in_owner=0,
                ),
            ],
            attributes={"symbol": symbol},
        )

    @property
    def name(self) -> str:
        return "external_symbol"

    def __str__(self) -> str:
        return f"external @{self.attributes['symbol']}: {self.result.type})"


class ConstantOp(Op):

    def __init__(self, value: Expr | int | str, type: Type):
        if isinstance(value, str):
            value = Symbol(value)
        elif isinstance(value, int):
            value = Number(value)
        super().__init__(
            results=[
                Value(
                    name="constant",
                    type=type,
                    defining_op_or_owner=self,
                    result_idx_in_owner=0,
                ),
            ],
            attributes={"value": value},
        )

    @property
    def name(self) -> str:
        return "constant"

    def __str__(self) -> str:
        return f"constant(value={self.attributes['value']}): {self.result.type}"


class ArrayViewOp(Op):
    def name(self) -> str:
        return "array_view"

    def __init__(self, array: Value, indices: Sequence[Value]):
        assert isinstance(array.type, ArrayType)
        super().__init__(
            operands=[OpOperand(array)] + [OpOperand(index) for index in indices],
            results=[
                Value(
                    name="array_view",
                    type=context.get_type(ArrayType, array.type.dims[len(indices) :]),
                    defining_op_or_owner=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    def __str__(self) -> str:
        return f"array_view(array={self.operands[0].source}): {self.result.type}"


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

    def __str__(self) -> str:
        return f"nop"


class ArrayRefOp(Op):
    def __init__(self, array: Value, indices: Sequence[Value]):
        assert isinstance(array.type, ArrayType)
        super().__init__(
            operands=[OpOperand(array)] + [OpOperand(index) for index in indices],
            results=[
                Value(
                    name="array_ref",
                    type=context.get_type(ArrayType, array.type.dims[len(indices) :]),
                    defining_op_or_owner=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "array_ref"

    def __str__(self) -> str:
        return f"array_ref(array={self.operands[0].source}, indices={', '.join([str(index) for index in self.operands[1:]])}): {self.result.type}"


class DeviceOp(Op):
    """
    Operation that is likely to be executed as a device function.

    Attributes:
        mem: Literal["G2S", "G2R", "S2R", "R2S", "R2G", "S2G", "R2R"]
    """

    def __init__(self, mem: str, operands: Sequence[Value], results: Sequence[Value]):
        super().__init__(operands, results)
        self.set_attribute("mem", mem)

    @property
    def name(self) -> str:
        return "device"

    def __str__(self) -> str:
        return f"{self.name}({', '.join([str(operand) for operand in self.operands])}) attr={self.attributes}: {', '.join([str(result) for result in self.results])}"


class Block:
    operations: List[Op]
    arguments: List[Value]
    owner: Op

    def __init__(self, owner: Op):
        self.operations = []
        self.arguments = []
        self.owner = owner

    def add_operation(self, op: Op):
        self.operations.append(op)

    def __str__(self) -> str:
        return f"block(ops=[{';\n '.join([str(op) for op in self.operations])}], args={', '.join([str(arg) for arg in self.arguments])})"


class MetaOp(Op):
    block: Block

    def __init__(self):
        super().__init__(
            attributes={},
            operands=[],
            results=[],
        )
        self.block = Block(self)

    @property
    def name(self) -> str:
        return "meta"

    def __str__(self) -> str:
        return f"meta(ops={', '.join([str(op) for op in self.block.operations])}, args={', '.join([str(arg) for arg in self.block.arguments])})"


class ForLoopOp(Op):
    body: Block

    def __init__(
        self,
        induction_var: str,
        range: Tuple[Value, Value],
        iter_args: Dict[str, Value] = {},
    ):
        super().__init__(
            attributes={},
            operands=[OpOperand(range[0]), OpOperand(range[1])]
            + [OpOperand(value) for value in iter_args.values()],
            results=[],
        )
        self.body = Block(self)
        self.body.arguments.append(
            Value(induction_var, context.get_type(IndexType), self.body, 0)
        )
        for name, arg in iter_args.items():
            self.body.arguments.append(Value(name, arg.type, self.body, 0))

    @property
    def name(self) -> str:
        return "for_loop"

    def get_induction_var(self) -> Value:
        return self.body.arguments[0]

    def get_iter_arg(self, idx: int) -> Value:
        return self.body.arguments[idx + 1]

    def __str__(self) -> str:
        return f"for_loop {self.get_induction_var()} = {self.operands[0]} to {self.operands[1]} iter_args({', '.join([str(arg) for arg in self.body.arguments[1:]])}) body={self.body})"


class SymbolTable:
    def __init__(self, owner: Op):
        self.symbol_table: Dict[str, Op] = {}
        self.owner = owner

    def add_symbol(self, name: str, op: Op):
        self.symbol_table[name] = op
        return op

    def __getitem__(self, name: str) -> Op:
        if name in self.symbol_table:
            return self.symbol_table[name]
        elif self.owner is not None:
            return self.owner.symbol_table[name]
        else:
            raise ValueError(f"Symbol {name} not found")


context = OpIrContext()
op_builder = OpBuilder()
