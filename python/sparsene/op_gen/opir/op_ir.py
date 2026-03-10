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
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> ForLoopOp: ...
    @overload
    def build(
        self,
        op_type: type[ConstantOp],
        value: Expr | int | str,
        type: Type,
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> ConstantOp: ...
    @overload
    def build(
        self, op_type: type[NoOp], name_hint: Optional[str | Sequence[str]] = None
    ) -> NoOp: ...
    @overload
    def build(
        self,
        op_type: type[ArrayRefOp],
        *args,
        name_hint: Optional[str | Sequence[str]] = None,
        **kwargs,
    ) -> ArrayRefOp: ...
    @overload
    def build(
        self,
        op_type: type[DeviceOp],
        mem: str,
        operands: Sequence[Value],
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> DeviceOp: ...
    @overload
    def build(
        self, op_type: type[MetaOp], name_hint: Optional[str | Sequence[str]] = None
    ) -> MetaOp: ...
    @overload
    def build(
        self,
        op_type: type[ExternalSymbolOp],
        symbol: str,
        type: Type,
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> ExternalSymbolOp: ...
    @overload
    def build(
        self,
        op_type: type[LoopResultOp],
        op: Op,
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> LoopResultOp: ...
    @overload
    def build(
        self,
        op_type: type[AddOp | SubOp | MulOp | DivOp | PowOp],
        lhs: Value,
        rhs: Value,
        name_hint: Optional[str | Sequence[str]] = None,
    ) -> ArithmeticOp: ...
    @overload
    def build(
        self, 
        op_type: type[LoadOp],
        array: Value, 
        indices: Sequence[Value],
        name_hint: Optional[str | Sequence[str]] = None
    ) -> LoadOp: ...
    @overload
    def build(
        self,
        op_type: type[Op],
        *args,
        name_hint: Optional[str | Sequence[str]] = None,
        **kwargs,
    ) -> Op: ...

    def build(
        self,
        op_type: type[Op],
        *args,
        name_hint: Optional[str | Sequence[str]] = None,
        **kwargs,
    ) -> Op:
        op = op_type(*args, **kwargs)
        op.residing_block = self.current_block
        if self.current_block is not None:
            self.current_block.add_op(op)

        if name_hint is not None:
            if isinstance(name_hint, str):
                assert op.has_single_result()
                op.result.name_hint = name_hint
            elif isinstance(name_hint, Sequence):
                for result, name in zip(op.results, name_hint):
                    result.name_hint = name
        return op

    @property
    def symbol_table(self) -> SymbolTable:
        if self.current_block is not None:
            return self.current_block.symbol_table
        else:
            raise ValueError("No symbol table available: not in a block. ")

    def lookup_symbol(self, name: str, type: Optional[Type] = None) -> Value:
        #! DEBUG begin
        # if name == "_":
        #     assert 0
        #! DEBUG end
        if (
            self.current_block is not None
            and (value := self.current_block.symbol_table[name]) is not None
        ):
            return value
        elif type is not None:
            assert self.current_block is not None
            value = self.build(ExternalSymbolOp, name, type).result
            self.current_block.symbol_table.add_symbol(name, value)
            return value
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
class ArrayType(Type):
    dims: List[Expr]
    datatype: Type

    def __init__(self, dims: Sequence[Expr], datatype: Type):
        self.dims = list(dims)
        self.datatype = datatype

    # return f"{self.element_type}[{']['.join(str(dim) for dim in self.dims)}]"
    # return f"array[{']['.join(str(dim) for dim in self.dims)}]"


@dataclass
class Value:
    type: Type
    name_hint: Optional[str]

    def __init__(self, type: Type, name_hint: Optional[str] = None):
        object.__setattr__(self, "type", type)
        self.name_hint = name_hint

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass
class OpResult(Value):
    defining_op: Op
    result_idx_in_owner: int

    def __init__(
        self,
        type: Type,
        defining_op: Op,
        result_idx_in_owner: int,
        name_hint: Optional[str] = None,
    ):
        super().__init__(type, name_hint)
        self.defining_op = defining_op
        self.result_idx_in_owner = result_idx_in_owner

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: Any) -> bool:
        return id(self) == id(other)


@dataclass
class BlockArgument(Value):
    owner: Block
    name_hint: Optional[str]
    arg_idx: int

    def __init__(
        self,
        type: Type,
        owner: Block,
        result_idx_in_owner: int,
        name_hint: Optional[str] = None,
    ):
        super().__init__(type, name_hint)
        self.owner = owner
        self.arg_idx = result_idx_in_owner

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

    def operand_name(self, idx: int) -> str:
        return f"operand_{idx}"

    def result_name(self, idx: int) -> str:
        return f"result_{idx}"


class ExternalSymbolOp(Op):

    def __init__(self, symbol: str, type: Type):
        super().__init__(
            results=[
                OpResult(
                    type=type, defining_op=self, result_idx_in_owner=0, name_hint=symbol
                ),
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


# class ArrayRefOp(Op):
#     def __init__(self, array: Value, indices: Sequence[Value]):
#         assert isinstance(array.type, ArrayType)
#         array_dims = array.type.dims[len(indices) :]
#         for index in reversed(indices):
#             if isinstance(index.type, ArrayType):
#                 array_dims.insert(0, index.type.dims[0])
#         super().__init__(
#             operands=[array] + list(indices),
#             results=[
#                 OpResult(
#                     type=ArrayType(array_dims, array.type.datatype),
#                     defining_op=self,
#                     result_idx_in_owner=0,
#                 ),
#             ],
#         )

#     @property
#     def name(self) -> str:
#         return "array_ref"

class ArrayRefOp(Op):
    def __init__(self, array: Value, indices: Sequence[Value]):
        assert isinstance(array.type, ArrayType)

        old_dims = array.type.dims
        new_base_dims = []
        extra_dims = []

        # 1. 遍历 indices，对照原始维度进行判断
        for i, index in enumerate(indices):
            # 如果索引是 "_" (下划线占位符)，则保留该维度
            # 假设你的 "_" 被包装成了某种特殊的 Value 或其 name_hint 是 "_"
            if getattr(index, "name_hint", None) == "_":
                new_base_dims.append(old_dims[i])
            
            # 如果索引本身是带维度的数组（高级索引逻辑）
            if isinstance(index.type, ArrayType):
                extra_dims.extend(index.type.dims)

        # 2. 加上原数组中完全没有被索引到的剩余维度
        new_base_dims.extend(old_dims[len(indices):])

        # 最终形状 = 嵌套带来的维度 + 保留/剩余的维度
        array_dims = extra_dims + new_base_dims
        
        super().__init__(
            operands=[array] + list(indices),
            results=[
                OpResult(
                    type=ArrayType(array_dims, array.type.datatype),
                    defining_op=self,
                    result_idx_in_owner=0,
                ),
            ],
        )

    @property
    def name(self) -> str:
        return "array_ref"


class LoadOp(Op):
    """
    从数组或 ArrayRef 的结果中加载标量值。
    """
    def __init__(self, array_or_ref: Value, indices: Sequence[Value] = [], name_hint: str | None = None):
        # 这里的 array_or_ref 可能是 ExternalSymbol(ArrayType) 或者是 ArrayRefOp 的结果
        assert isinstance(array_or_ref.type, ArrayType)

        original_dims = array_or_ref.type.dims
        preserved_dims = []


        # 遍历索引，决定哪些维度被消除，哪些被保留
        for i, idx in enumerate(indices):
            # 如果是占位符 "_"，则保留该维度
            if getattr(idx, "name_hint", None) == "_":
                preserved_dims.append(original_dims[i])
            # 如果是普通索引（如 %i13），该维度被消除（不加入 preserved_dims）
            else:
                pass
        
        # 如果索引没覆盖完数组的所有维度，剩下的维度也要保留
        if len(indices) < len(original_dims):
            preserved_dims.extend(original_dims[len(indices):])

        # 确定最终返回类型
        if len(preserved_dims) == 0:
            # 所有维度都被索引了，返回标量
            result_type = array_or_ref.type.datatype
        else:
            # 还有维度保留，返回子数组（Tile）
            result_type = ArrayType(preserved_dims, array_or_ref.type.datatype)

        super().__init__(
            operands=[array_or_ref] + list(indices),
            results=[
                OpResult(
                    type=result_type,
                    defining_op=self,
                    result_idx_in_owner=0,
                    name_hint=name_hint
                )
            ]
        )
        
        # 如果提供了 indices，我们可以先内部构建一个隐式的 ArrayRef 逻辑，
        # 或者直接作为 LoadOp 的操作数。这里采用直接作为操作数。
        # super().__init__(
        #     operands=[array_or_ref] + list(indices),
        #     results=[
        #         OpResult(
        #             type=array_or_ref.type.datatype, # 返回数组的底层数据类型 (Int/Float)
        #             defining_op=self,
        #             result_idx_in_owner=0
        #         )
        #     ]
        # )

    @property
    def name(self) -> str:
        return "load"

class LoadOffsetOp(Op):
    def __init__(self, array: Value, indices: Sequence[Value], name_hint: str | None = None):
        # 返回两个结果：left_offset, right_offset
        results = [
            OpResult(type=IntType(), name_hint=f"{name_hint}_l" if name_hint else "l", defining_op=self, result_idx_in_owner=0),
            OpResult(type=IntType(), name_hint=f"{name_hint}_r" if name_hint else "r", defining_op=self, result_idx_in_owner=1)
        ]
        super().__init__(operands=[array] + list(indices), results=results)

    @property
    def name(self) -> str:
        return "load_offset"

class ArangeOp(Op):
    def __init__(self, start: Value, length: Expr, name_hint: str | None = None):
        res_type = ArrayType([length], IntType())
        super().__init__(
            operands=[start],
            results=[OpResult(type=res_type, defining_op=self, result_idx_in_owner=0, name_hint=name_hint)]
        )
    
    @property
    def name(self) -> str:  
        return "arange"

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
        self.body.args.append(
            BlockArgument(
                IntType(), self.body, result_idx_in_owner=0, name_hint=induction_var
            )
        )
        for i, (name, arg) in enumerate(iter_args.items()):
            self.body.args.append(
                BlockArgument(
                    arg.type, self.body, result_idx_in_owner=i + 1, name_hint=name
                )
            )

    @property
    def name(self) -> str:
        return "for_loop"

    @property
    def range(self) -> Tuple[Value, Value]:
        return self.operands[0].source, self.operands[1].source

    @property
    def num_iter_args(self) -> int:
        return len(self.body.args) - 1

    @property
    def induction_var(self) -> BlockArgument:
        return self.body.args[0]

    def get_induction_var(self) -> BlockArgument:
        return self.body.args[0]

    def get_iter_arg(self, idx: int) -> BlockArgument:
        return self.body.args[idx + 1]

#> original LoopResultOp, accept value.defining_op
# class LoopResultOp(Op):
#     def __init__(self, op: Op):
#         super().__init__(
#             operands=op.results,
#             results=[],
#         )

#     @property
#     def name(self) -> str:
#         return "loop_result"
#> current LoopResultOp, accept the value directly
class LoopResultOp(Op):
    def __init__(self, value: Value):
        print("(debug) LoopResultOp value", value)
        super().__init__(
            operands=[value],
            results=[],
        )

    @property
    def name(self) -> str:
        return "loop_result"



class ArithmeticOp(Op):
    """算术运算基类"""
    def __init__(self, lhs: Value, rhs: Value):
        # 推导结果类型：如果有一个是 Float，结果通常是 Float
        res_type = FloatType() if isinstance(lhs.type, FloatType) or isinstance(rhs.type, FloatType) else IntType()
        super().__init__(
            operands=[lhs, rhs],
            results=[
                OpResult(type=res_type, defining_op=self, result_idx_in_owner=0),
            ]
        )

class AddOp(ArithmeticOp):
    @property
    def name(self) -> str:
        return "add"

class SubOp(ArithmeticOp):
    @property
    def name(self) -> str:
        return "sub"

class MulOp(ArithmeticOp):
    @property
    def name(self) -> str:
        return "mul"
    
class DivOp(ArithmeticOp): 
    @property
    def name(self) -> str:
        return "div"
    
class PowOp(ArithmeticOp):
    @property
    def name(self) -> str:
        return "pow"


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
