from __future__ import annotations

from sparsene.format.format import Expr, Symbol, Number
from typing import Any, List, Dict, Tuple, overload, Sequence, Optional
from dataclasses import dataclass
from copy import deepcopy
from abc import ABC, abstractmethod


class OpIrContext:
    type_cache: Dict[type, Type] = {}

    @overload
    def get_type(self, t: type[IntType]) -> IntType: ...
    @overload
    def get_type(self, t: type[FloatType]) -> FloatType: ...
    @overload
    def get_type(self, t: type[RangeType]) -> RangeType: ...
    @overload
    def get_type(
        self, t: type[ArrayType], dims: List[Expr], datatype: Type
    ) -> ArrayType: ...

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
        iter_args: Dict[str, Value] = {},
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
    def build(
        self, op_type: type[DeviceOp], mem: str, operands: Sequence[Value]
    ) -> DeviceOp: ...
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

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass(frozen=True)
class OpResult(Value):
    defining_op: Op
    result_idx_in_owner: int

    def __init__(self, type: Type, defining_op: Op, result_idx_in_owner: int):
        super().__init__(type)
        object.__setattr__(self, "defining_op", defining_op)
        object.__setattr__(self, "result_idx_in_owner", result_idx_in_owner)

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
    symbol_table: SymbolTable

    def __init__(
        self,
        operands: Sequence[OpOperand | Value] = [],
        results: Sequence[OpResult] = [],
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


class ArrayViewOp(Op):
    def name(self) -> str:
        return "array_view"

    def __init__(self, array: Value, indices: Sequence[Value]):
        assert isinstance(array.type, ArrayType)
        super().__init__(
            operands=[OpOperand(array)] + [OpOperand(index) for index in indices],
            results=[
                OpResult(
                    type=context.get_type(
                        ArrayType, array.type.dims[len(indices) :], array.type.datatype
                    ),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )


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
            operands=[OpOperand(array)] + [OpOperand(index) for index in indices],
            results=[
                OpResult(
                    type=context.get_type(
                        ArrayType, array.type.dims[len(indices) :], array.type.datatype
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

    def __init__(self, mem: str, operands: Sequence[Value], results: Sequence[Value]):
        super().__init__(operands, results)
        self.set_attribute("mem", mem)


class Block:
    ops: List[Op]
    args: List[Value]
    owner: Op

    def __init__(self, owner: Op):
        self.ops = []
        self.args = []
        self.owner = owner

    def add_operation(self, op: Op):
        self.ops.append(op)


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
        self.body.args.append(
            BlockArgument(context.get_type(IntType), self.body, induction_var)
        )
        for name, arg in iter_args.items():
            self.body.args.append(BlockArgument(arg.type, self.body, name))

    @property
    def name(self) -> str:
        return "for_loop"

    def get_induction_var(self) -> Value:
        return self.body.args[0]

    def get_iter_arg(self, idx: int) -> Value:
        return self.body.args[idx + 1]


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
