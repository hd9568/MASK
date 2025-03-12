from __future__ import annotations

from sparsene.model.format import Expr, Symbol, Number
from typing import Any, List, Dict
from dataclasses import dataclass
from copy import deepcopy


class OpIrContext:
    type_cache: Dict[type, Type] = {}

    def get_type(self, t: type, *args, **kwargs) -> Type:
        if t not in self.type_cache:
            self.type_cache[t] = t(*args, **kwargs)
        return self.type_cache[t]


@dataclass
class Type:
    def __str__(self):
        return self.__class__.__name__


@dataclass
class Index(Type):
    pass


@dataclass
class Range(Type):
    pass


@dataclass
class Array(Type):
    dims: List[Expr]


@dataclass(frozen=True, init=False)
class Value:
    name: str
    type: Type
    owner: Op
    result_idx_in_owner: int

    def __init__(self, name: str, type: Type, owner: Op, result_idx_in_owner: int):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "result_idx_in_owner", result_idx_in_owner)

    def __repr__(self):
        return f"Value(name={self.name}, type={self.type}, owner={self.owner}, result_idx_in_owner={self.result_idx_in_owner})"


@dataclass(frozen=True, init=False)
class OpOperand:
    source: Value


class Block:
    operations: List[Op]
    owner: Op


class Op:
    name: str
    attributes: Dict[str, Any] = {}
    operands: List[OpOperand] = []
    results: List[Value] = []
    symbol_table: SymbolTable

    def __init__(
        self,
        name: str,
        attributes: Dict[str, Any],
        operands: List[OpOperand],
        results: List[Value],
    ):
        self.name = name
        self.attributes = attributes
        self.operands = operands
        self.results = results
        self.symbol_table = SymbolTable(self)

    def add_attribute(self, name: str, value: Any):
        self.attributes[name] = value

    def add_operand(self, operand: OpOperand):
        self.operands.append(operand)

    def add_result(self, result: Value):
        self.results.append(result)

    def get_result(self) -> Value:
        assert len(self.results) == 1
        return self.results[0]

    def get_results(self) -> List[Value]:
        return list(self.results)

    def __repr__(self):
        return f"{self.name} {self.operands} -> {self.results}"


class ConstantOp(Op):
    name: str = "constant"

    def __init__(self, name: str, value: Expr | int | str, type: Type):
        if isinstance(value, str):
            value = Symbol(value)
        elif isinstance(value, int):
            value = Number(value)
        self.attributes = {"value": value}
        self.results.append(
            Value(name=name, type=type, owner=self, result_idx_in_owner=0),
        )


class ArrayViewOp(Op):
    name = "array_view"


class NoOp(Op):
    def __init__(self):
        super().__init__(
            name="nop",
            attributes={},
            operands=[],
            results=[],
        )


class ArrayRefOp(Op):
    name = "array_ref"


class DeviceOp(Op):
    """
    Operation that is likely to be executed as a device function.

    Attributes:
        mem: Literal["G2S", "G2R", "S2R", "R2S", "R2G", "S2G", "R2R"]
    """


class MetaOp(Op):
    name = "meta"
    block: Block


class ForLoopOp(Op):
    """
    For loop operation.

    Attributes:
        range: Tuple[Value, Value]
        args: Dict[str, Value]
        body: Op
    """


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


class OpBuilder:
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table

    def build(
        self,
        name: str,
        op_type: type[Op],
        *args,
        **kwargs,
    ) -> Op:
        op = op_type(*args, **kwargs)
        self.symbol_table.add_symbol(name, op)
        return op
