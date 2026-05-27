from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sympy import Expr as SymExpr
from sympy import Function, Integer, Rational, Symbol, simplify
from sympy import Number as SymNumber
from sympy import sympify
from sympy import preorder_traversal

from sparsene.op_gen.opir.op_ir import (
    Op,
    OpBuilder,
    OpOperand,
    ConstantOp,
    ForLoopOp,
    ArrayRefOp,
    LoadOp,
    LoadOffsetOp,
    Value,
    IntType,
    MetaOp,
    ExternalSymbolOp,
    AddOp,
    SubOp,
    MulOp,
    DivOp,
    PowOp,
    ArrayType,
)
from sparsene.op_gen.opir.ops import (
    CooAtomicFormatLoadIdxOp,
    CooAtomicFormatLoadValOp,
    McoAtomicFormatLoadValOp,
)
from sparsene.op_gen.computent.computent import computent_from_rts, Computent, ArrayDef
from sparsene.op_gen.computent.computent import ArrayDefCollection
from sparsene.op_gen.computent.computent import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
    ArrayRef,
    DataType,
)

# @dataclass
# class _FlattenSpec:
#     prefix_rank: int
#     varlen_pos: int
#     scale_expr: SymExpr

@dataclass
class _FlattenSpec:
    prefix_rank: int
    varlen_pos: int
    a_expr: SymExpr  # 线性系数 a
    b_expr: SymExpr  # 常数项 b


class VarlenLoweringPass:
    def __init__(self, op_builder: Optional[OpBuilder], varlen2LenArrayTable: Optional[Dict[str, ArrayDef]] = None):
        self.op_builder = op_builder
        self._root: Optional[MetaOp] = None
        self._scalar_symbol_cache: Dict[str, Value] = {}
        self._array_symbol_cache: Dict[str, ExternalSymbolOp] = {}
        self.varlen2LenArrayTable = varlen2LenArrayTable or {}

        # [新增] 追踪数组更名：原始名 -> 当前名
        self._array_name_aliases: Dict[str, str] = {}

    def _get_current_name(self, original_name: str) -> str:
        """递归查找数组在经过多次压缩后的当前名称"""
        name = original_name
        while name in self._array_name_aliases:
            name = self._array_name_aliases[name]
        return name

    def run(self, ops):
        print("varlen2LenArrayTable:", self.varlen2LenArrayTable)

        root = self._normalize_root(ops)
        if root is None:
            return ops

        self._root = root

        while True:
            # 每次循环前必须重建符号缓存，因为上一轮可能修改了数组名称和维度
            self._rebuild_symbol_cache(root)

            # 1. 寻找下一个要消除的 varlen 定义（例如 "varlen(K)"）
            target_varlen = self._pick_next_varlen_to_lower(root)
            if target_varlen is None:
                break
            
            # 2. 获取对应的长度数组初始定义
            len_array_def = self.varlen2LenArrayTable[target_varlen]
            
            # [核心修改] 获取其当前在 IR 中的名称（可能已被加了 _new 后缀）
            current_len_name = self._get_current_name(len_array_def.name)
            
            # 在当前符号表中找到对应的 ExternalSymbolOp
            len_ext_op = self._array_symbol_cache.get(current_len_name)
            
            print(f"Target varlen to lower: {target_varlen}, corresponding length array: {current_len_name}")
            print(f"Found ExternalSymbolOp for length array: {len_ext_op is not None}")

            if len_ext_op is None:
                raise ValueError(f"Length array '{current_len_name}' for varlen '{target_varlen}' not found in IR.")

            # 3. 执行单轮消除，传入目标 varlen 标识和对应的长度算子
            changed = self._lower_single_varlen_round_v2(root, target_varlen, len_ext_op)
            if not changed:
                break

        return root

    def _lower_single_varlen_round_v2(self, root: MetaOp, target_varlen: str, len_ext_op: ExternalSymbolOp) -> bool:
        #! 1. 准备长度数组修改
        old_len_type = len_ext_op.result.type
        old_len_name = str(len_ext_op.attributes.get("symbol", len_ext_op.result.name_hint or ""))

        #! 将 [M/BLK_M] 修改为 [M/BLK_M + 1]
        assert isinstance(old_len_type, ArrayType), f"Expected length array to have ArrayType, got {old_len_type}"
        prefix_dims = list(old_len_type.dims)

        if len(prefix_dims) == 0:
            # 1. 只是简单改名，不增加维度，不生成 offset 数组
            new_len_name = f"{old_len_name}_new" if not old_len_name.endswith("_new") else f"{old_len_name}_1"
            self._array_name_aliases[old_len_name] = new_len_name
            len_ext_op.attributes["symbol"] = new_len_name
            len_ext_op.result.name_hint = new_len_name
            
            # 2. 依然去压缩依赖它的数组 (varlen(M) -> nnz_dim_varlen_M)
            flatten_specs = self._find_and_compress_dependent_arrays(root, target_varlen, prefix_dims)
            self._rebuild_symbol_cache(root)
            
            # 3. 遍历 Block 更新 Type，但不触发 LoadOffset 的替换
            self._rewrite_block(
                block=root.block,
                len_offset_value=len_ext_op.result,
                flatten_specs=flatten_specs,
                incoming_replace_map={}
            )
            return True

        new_len_dims = list(prefix_dims)
        new_len_dims[-1] = simplify(sympify(new_len_dims[-1]) + Integer(1))

        new_len_name = f"{old_len_name}_offset"
        len_ext_op.attributes["symbol"] = new_len_name
        len_ext_op.result.name_hint = new_len_name
        len_ext_op.result.type = ArrayType(new_len_dims, old_len_type.datatype)

        #! 2. 找到所有依赖于这个 target_varlen 的数据数组并压缩（如 val_sidx）
        # 这里复用你之前的逻辑：[M/B][varlen(K)/B][BLK_K] -> [nnz_block][BLK_K]
        flatten_specs = self._find_and_compress_dependent_arrays(root, target_varlen, prefix_dims)

        # 3. 重新建立缓存，使后续重写能找到新的符号
        self._rebuild_symbol_cache(root)

        # 4. 指令重写：进入 Block 替换所有的 Load 和 ArrayRef
        self._rewrite_block(
            block=root.block,
            len_offset_value=len_ext_op.result, # 传入新的 offset 数组 Value
            flatten_specs=flatten_specs,
            incoming_replace_map={}
        )
        return True

    def _find_and_compress_dependent_arrays(
        self, root: MetaOp, target_varlen: str, prefix_dims: List[SymExpr]
    ) -> Dict[Value, _FlattenSpec]:
        
        flatten_specs: Dict[Value, _FlattenSpec] = {}
        prefix_rank = len(prefix_dims)
        
        for op in list(self._array_symbol_cache.values()):
            arr_type = op.result.type
            if not isinstance(arr_type, ArrayType):
                continue
            
            old_dims = list(arr_type.dims)
            if not any(target_varlen in str(d) for d in old_dims):
                continue

            if len(old_dims) <= prefix_rank or not self._dims_equal(old_dims[:prefix_rank], prefix_dims):
                raise RuntimeError(
                    f"数组 {op.result.name_hint} 包含 {target_varlen}，"
                    f"但其前缀 {old_dims[:prefix_rank]} 与长度数组前缀 {prefix_dims} 不匹配！无法压缩。"
                )

            varlen_dim_expr = sympify(old_dims[prefix_rank])
            var_node = sympify(target_varlen)
            a_expr = simplify(varlen_dim_expr.diff(var_node))
            b_expr = simplify(varlen_dim_expr.subs(var_node, 0))

            old_name = str(op.attributes.get("symbol", op.result.name_hint or ""))
            new_name = f"{old_name}_new" if not old_name.endswith("_new") else f"{old_name}_1"
            
            # [核心修改] 记录数组更名历史
            self._array_name_aliases[old_name] = new_name

            # ================= NEW: 生成绝对安全的纯净维度名称 =================
            import re
            
            # 1. 从 "varlen(K/BLK_K)" 中提取出 "K/BLK_K"
            match = re.search(r'varlen\(([^)]+)\)', target_varlen)
            core_name = match.group(1) if match else "unknown"
            
            # 2. 将所有非字母数字和非下划线的字符（比如 '/', '*', '+', '-'）替换为 '_'
            safe_core_name = re.sub(r'[^a-zA-Z0-9_]', '_', core_name)
            
            # 3. 可选：把连续的多个下划线合并为一个，去掉首尾下划线，让名字更好看
            safe_core_name = re.sub(r'_+', '_', safe_core_name).strip('_')
            
            # 创建纯净的符号，例如：nnz_dim_K_BLK_K
            compressed_dim = Symbol(f"nnz_dim_{safe_core_name}")
            # =================================================================

            print(f"Compressing array '{old_name}' with varlen dim '{old_dims[prefix_rank]}' into new dim '{compressed_dim}'")

            # 更新维度... (省略细节，参考上一次的回复)
            # compressed_dim = Symbol(f"nnz_dim_{target_varlen.replace('(', '_').replace(')', '')}")
            new_dims = [compressed_dim] + old_dims[prefix_rank + 1 :]

            op.attributes["symbol"] = new_name
            op.result.name_hint = new_name
            op.result.type = ArrayType(new_dims, arr_type.datatype)

            flatten_specs[op.result] = _FlattenSpec(
                prefix_rank=prefix_rank,
                varlen_pos=prefix_rank,
                a_expr=a_expr,
                b_expr=b_expr
            )

        return flatten_specs

    def _build_fractional_mul(self, val: Value, coeff_expr: SymExpr, generated_ops: List[Op]) -> Value:
        """安全地计算 val * coeff，将分数 1/BLK_K 自动转化为 div op"""
        # 1. 处理系数为 0 的情况
        if simplify(coeff_expr) == 0:
            zero = ConstantOp(0, IntType())
            generated_ops.append(zero)
            return zero.result

        # 2. 处理系数为 1 的情况
        if self._is_one_expr(coeff_expr):
            return val

        # 3. 核心：拆解分子和分母
        # 例如：对于 1/BLK_K，num 会是 1，den 会是 BLK_K
        num, den = coeff_expr.as_numer_denom()
        res = val

        # 4. 如果分子不是 1，则生成乘法 (例如 a = 2)
        if simplify(num - 1) != 0:
            num_val = self._expr_to_value(num, generated_ops)
            mul = MulOp(res, num_val)
            generated_ops.append(mul)
            res = mul.result

        # 5. 如果分母不是 1，则直接生成除法 DivOp
        if simplify(den - 1) != 0:
            den_val = self._expr_to_value(den, generated_ops)  # 这里只会提取到 BLK_K
            div = DivOp(res, den_val)
            generated_ops.append(div)
            res = div.result

        return res

    def _get_base_offset(
        self,
        prefix_indices: Sequence[Value],
        scale_expr: SymExpr,
        len_offset_value: Value,
        prefix_ops: List[Op],
        offset_cache: Dict[Tuple[Value, ...], Value],
        scaled_cache: Dict[Tuple[Tuple[Value, ...], str], Value],
    ) -> Value:
        # 1. 获取（或生成）左边界偏移 %v0_l
        key = tuple(prefix_indices)
        if key not in offset_cache:
            load_l = LoadOp(len_offset_value, list(prefix_indices), name_hint="left_bnd")
            prefix_ops.append(load_l)
            offset_cache[key] = load_l.result
        
        base_offset = offset_cache[key]

        # 2. 如果不需要缩放 (scale_expr == 1)，直接返回
        if self._is_one_expr(scale_expr) or self._is_blk_scaled_one_expr(scale_expr):
            return base_offset

        # 3. 检查缩放缓存，避免重复生成相同的 div 指令
        scale_str = str(scale_expr)
        scale_key = (key, scale_str)
        if scale_key in scaled_cache:
            return scaled_cache[scale_key]

        # 4. 安全地执行分数乘法：这里会把 1/BLK_K 自动转为 DivOp
        scaled_val = self._build_fractional_mul(base_offset, scale_expr, prefix_ops)
        
        # 写入缓存并返回
        scaled_cache[scale_key] = scaled_val
        return scaled_val

    def _rewrite_array_indices_if_needed(
        self,
        op: Op,
        flatten_specs: Dict[Value, _FlattenSpec],
        len_offset_value: Value,
        prefix_ops: List[Op],
        offset_cache: Dict[Tuple[Value, ...], Value],
        length_cache: Dict[Tuple[Value, ...], Value],
        scaled_cache: Dict[Tuple[Tuple[Value, ...], str], Value],
        flat_index_cache: Dict[Tuple[Any, ...], Value],
    ) -> bool:
        if op.num_operands == 0:
            return False

        array_src = op.operands[0].source
        if array_src not in flatten_specs:
            return False

        spec = flatten_specs[array_src]

        # Keep COO (idx/val) explicit [ll, rr] bounds untouched.
        # Those bounds are already absolute offsets and should not be collapsed
        # into a single flattened index by generic array-index rewriting.
        if isinstance(op, (CooAtomicFormatLoadIdxOp, CooAtomicFormatLoadValOp)) and len(op.operands) == 3:
            return False

        # ================= NEW: 拦截最外层全局维度 =================
        if spec.prefix_rank == 0:
            # 说明这个 varlen 是稠密的最外层 (如 varlen(M))
            # 它的索引就是对的，不需要算 a*left + b*i，直接放行，返回 True 刷新 Type 即可。
            return True
        # ============================================================

        indices = [operand.source for operand in op.operands[1:]]

        if len(indices) <= spec.varlen_pos:
            return False

        prefix_indices = indices[: spec.prefix_rank]
        j_val = indices[spec.varlen_pos]
        rest_indices = indices[spec.varlen_pos + 1 :]

        # 1. 计算 term_a = a * left (利用 get_base_offset 内部的缓存)
        term_a = self._get_base_offset(
            prefix_indices=prefix_indices,
            scale_expr=spec.a_expr,  
            len_offset_value=len_offset_value,
            prefix_ops=prefix_ops,
            offset_cache=offset_cache,
            scaled_cache=scaled_cache,
        )

        i_val = prefix_indices[0] 

        # ==================== 新增缓存逻辑 ====================
        # 我们用 (term_a, i_val, b_expr) 作为键，缓存 dim1_base_idx 
        # 用 (dim1_base_idx, j_val) 作为键，缓存最终的 flat_idx
        
        base_cache_key = (term_a, i_val, str(spec.b_expr))
        if base_cache_key not in flat_index_cache:
            if self._is_zero_expr(spec.b_expr):
                dim1_base_idx = term_a
            else:
                # 2. 计算 term_b = b * i
                term_b = self._build_fractional_mul(i_val, spec.b_expr, prefix_ops)
                # 3. 相加: dim1_base_idx = term_a + term_b
                dim1_base_idx = self._build_add(term_a, term_b, prefix_ops)
            flat_index_cache[base_cache_key] = dim1_base_idx
        else:
            dim1_base_idx = flat_index_cache[base_cache_key]

        # 4. 加上后续维度 j_val
        flat_cache_key = (dim1_base_idx, j_val)
        if flat_cache_key not in flat_index_cache:
            if self._is_placeholder_value(j_val):
                flat_idx = dim1_base_idx
            else:
                flat_idx = self._build_add(dim1_base_idx, j_val, prefix_ops)
            flat_index_cache[flat_cache_key] = flat_idx
        else:
            flat_idx = flat_index_cache[flat_cache_key]
        # =======================================================

        # 5. 重写当前 Op 的操作数
        preserve_placeholder = isinstance(
            op,
            (CooAtomicFormatLoadIdxOp, CooAtomicFormatLoadValOp, McoAtomicFormatLoadValOp),
        ) and self._is_placeholder_value(j_val)

        if preserve_placeholder:
            new_indices = [flat_idx, j_val] + rest_indices
        else:
            new_indices = [flat_idx] + rest_indices

        op.operands = [OpOperand(array_src)] + [OpOperand(v) for v in new_indices]
        return True

    def _build_add(self, lhs: Value, rhs: Value, generated_ops: List[Op]) -> Value:
        if self._is_const_int(lhs, 0):
            return rhs
        if self._is_const_int(rhs, 0):
            return lhs
        
        add = AddOp(lhs, rhs)
        generated_ops.append(add)
        return add.result

    def _materialize_len_from_offset(
        self,
        op: LoadOp,
    ) -> Tuple[List[Op], Value, Value, Optional[Tuple[Value, ...]]]:
        """
        将原始 load(%val_len, [i]) 转换为：
        %l, %r = load_offset(%val_len_offset, [i])
        %len = sub %r, %l
        """
        array_val = op.operands[0].source  # 已经是修改后的 _offset 数组
        indices = [operand.source for operand in op.operands[1:]]
        
        generated: List[Op] = []

        # 1. 生成专用的 load_offset 算子
        load_off = LoadOffsetOp(array_val, indices, name_hint=op.result.name_hint)
        generated.append(load_off)
        
        left_val = load_off.results[0]
        right_val = load_off.results[1]

        # 2. 生成 sub 算子求长度
        # 推荐直接建立 SubOp 或复用你之前的 _build_sub
        length_val = self._build_sub(lhs=right_val, rhs=left_val, generated_ops=generated)
        
        if op.result.name_hint:
            length_val.name_hint = op.result.name_hint

        # 返回生成的指令、长度结果、以及用于后续 div 的左边界偏移
        return generated, length_val, left_val, tuple(indices)


    def _get_all_active_varlens(self, root: MetaOp) -> set[str]:
        """扫描所有 ExternalSymbolOp，提取当前 IR 中存在的所有 varlen(dim) 字符串"""
        active_varlens = set()
        for op in self._walk_ops(root.block):
            if isinstance(op, ExternalSymbolOp) and isinstance(op.result.type, ArrayType):
                for dim in op.result.type.dims:
                    # 使用正则或字符串搜索提取 varlen(xxx)
                    # 这里假设 dim 是字符串或包含 varlen 的表达式
                    found = self._extract_varlen_strings(str(dim))
                    active_varlens.update(found)
        return active_varlens

    def _extract_varlen_strings(self, dim_str: str) -> List[str]:
        """从维度字符串中提取出 'varlen(...)' 部分"""
        # 简单实现：寻找 'varlen(' 到匹配的 ')'
        import re
        return re.findall(r'varlen\([^)]+\)', dim_str)

    def _pick_next_varlen_to_lower(self, root: MetaOp) -> Optional[str]:
        """基于当前 IR 状态，动态寻找下一个没有任何依赖的 varlen"""
        active_varlens = self._get_all_active_varlens(root)
        if not active_varlens:
            return None

        for v_str in active_varlens:
            if v_str not in self.varlen2LenArrayTable:
                continue
            
            len_array_def = self.varlen2LenArrayTable[v_str]
            current_len_name = self._get_current_name(len_array_def.name)
            len_ext_op = self._array_symbol_cache.get(current_len_name)
            
            if len_ext_op is None:
                continue

            len_type = len_ext_op.result.type
            if not isinstance(len_type, ArrayType):
                continue
                
            # [核心修改] 检查该长度数组 *当前* 的维度中是否还有依赖
            has_dependency = False
            for dim in len_type.dims:
                if self._contains_varlen(dim):
                    deps = self._extract_varlen_strings(str(dim))
                    # 如果依赖的 varlen 也在 active 集合中，说明还不能消除当前 v_str
                    if any(d in active_varlens for d in deps):
                        has_dependency = True
                        break
            
            if not has_dependency:
                return v_str
        
        return None

    def _normalize_root(self, ops) -> Optional[MetaOp]:
        if isinstance(ops, MetaOp):
            return ops
        if isinstance(ops, Sequence) and len(ops) == 1 and isinstance(ops[0], MetaOp):
            return ops[0]
        return None

    def _rebuild_symbol_cache(self, root: MetaOp) -> None:
        self._scalar_symbol_cache = {}
        self._array_symbol_cache = {}
        for op in self._walk_ops(root.block):
            if isinstance(op, ExternalSymbolOp):
                symbol = str(op.attributes.get("symbol", op.result.name_hint or ""))
                if isinstance(op.result.type, ArrayType):
                    self._array_symbol_cache[symbol] = op
                else:
                    self._scalar_symbol_cache[symbol] = op.result

    def _walk_ops(self, block) -> List[Op]:
        collected: List[Op] = []
        for op in block.ops:
            collected.append(op)
            if isinstance(op, ForLoopOp):
                collected.extend(self._walk_ops(op.body))
            elif isinstance(op, MetaOp):
                collected.extend(self._walk_ops(op.block))
        return collected


    

    def _rewrite_block(
        self,
        block,
        len_offset_value: Value,
        flatten_specs: Dict[Value, _FlattenSpec],
        incoming_replace_map: Dict[Value, Value],
        incoming_offset_cache: Optional[Dict[Tuple[Value, ...], Value]] = None,
        incoming_length_cache: Optional[Dict[Tuple[Value, ...], Value]] = None,
        incoming_scaled_cache: Optional[Dict[Tuple[Tuple[Value, ...], str], Value]] = None,
        incoming_flat_index_cache: Optional[Dict[Tuple[Any, ...], Value]] = None,
    ) -> None:
        replace_map: Dict[Value, Value] = dict(incoming_replace_map)
        offset_cache: Dict[Tuple[Value, ...], Value] = (
            dict(incoming_offset_cache) if incoming_offset_cache else {}
        )
        length_cache: Dict[Tuple[Value, ...], Value] = (
            dict(incoming_length_cache) if incoming_length_cache else {}
        )
        scaled_cache: Dict[Tuple[Tuple[Value, ...], str], Value] = (
            dict(incoming_scaled_cache) if incoming_scaled_cache else {}
        )
        flat_index_cache: Dict[Tuple[Any, ...], Value] = (
            dict(incoming_flat_index_cache) if incoming_flat_index_cache else {}
        )

        new_ops: List[Op] = []

        for op in block.ops:
            self._remap_op_operands(op, replace_map)

            if isinstance(op, LoadOp) and op.num_operands > 0 and op.operands[0].source is len_offset_value:
                materialized_ops, length_value, left_offset_value, key = self._materialize_len_from_offset(op)
                if materialized_ops:
                    new_ops.extend(materialized_ops)
                    replace_map[op.result] = length_value
                    if key is not None:
                        offset_cache[key] = left_offset_value
                        length_cache[key] = length_value
                    continue

            prefix_ops: List[Op] = []
            rewritten = self._rewrite_array_indices_if_needed(
                op=op,
                flatten_specs=flatten_specs,
                len_offset_value=len_offset_value,
                prefix_ops=prefix_ops,
                offset_cache=offset_cache,
                length_cache=length_cache,
                scaled_cache=scaled_cache,
                flat_index_cache=flat_index_cache,
            )

            if isinstance(op, ForLoopOp):
                self._rewrite_block(
                    block=op.body,
                    len_offset_value=len_offset_value,
                    flatten_specs=flatten_specs,
                    incoming_replace_map=replace_map,
                    incoming_offset_cache=offset_cache,
                    incoming_length_cache=length_cache,
                    incoming_scaled_cache=scaled_cache,
                    incoming_flat_index_cache=flat_index_cache,
                )
            elif isinstance(op, MetaOp):
                self._rewrite_block(
                    block=op.block,
                    len_offset_value=len_offset_value,
                    flatten_specs=flatten_specs,
                    incoming_replace_map=replace_map,
                    incoming_offset_cache=offset_cache,
                    incoming_length_cache=length_cache,
                    incoming_scaled_cache=scaled_cache,
                    incoming_flat_index_cache=flat_index_cache,
                )

            if rewritten and isinstance(op, (ArrayRefOp, LoadOp)):
                self._refresh_array_like_result_type(op)

            if prefix_ops:
                new_ops.extend(prefix_ops)
            new_ops.append(op)

        block.ops = new_ops

    def _remap_op_operands(self, op: Op, replace_map: Dict[Value, Value]) -> None:
        if not replace_map:
            return

        if op.num_operands > 0:
            remapped_operands: List[OpOperand] = []
            for operand in op.operands:
                source = operand.source
                while source in replace_map:
                    source = replace_map[source]
                remapped_operands.append(OpOperand(source))
            op.operands = remapped_operands

        if (
            isinstance(op, (CooAtomicFormatLoadIdxOp, CooAtomicFormatLoadValOp, McoAtomicFormatLoadValOp))
            and hasattr(op, "len")
        ):
            len_source = op.len
            while len_source in replace_map:
                len_source = replace_map[len_source]
            op.len = len_source

    

    

    def _is_const_int(self, val: Value, target: int) -> bool:
        """Check if val is a ConstantOp with the given integer value."""
        if not hasattr(val, "defining_op") or val.defining_op is None:
            return False
        op = val.defining_op
        if not isinstance(op, ConstantOp):
            return False
        try:
            return int(op.attributes.get("value", None)) == target
        except (TypeError, ValueError):
            return False

    def _build_sub(self, lhs: Value, rhs: Value, generated_ops: List[Op]) -> Value:
        if self._is_const_int(rhs, 0):
            return lhs
        sub = SubOp(lhs, rhs)
        generated_ops.append(sub)
        return sub.result

    def _expr_to_value(self, expr: Any, generated_ops: List[Op]) -> Value:
        expr = simplify(sympify(expr))

        if isinstance(expr, Integer):
            const = ConstantOp(int(expr), IntType())
            generated_ops.append(const)
            return const.result

        if isinstance(expr, Rational):
            num = ConstantOp(int(expr.p), IntType())
            den = ConstantOp(int(expr.q), IntType())
            generated_ops.extend([num, den])
            div = DivOp(num.result, den.result)
            generated_ops.append(div)
            return div.result

        if isinstance(expr, SymNumber):
            const = ConstantOp(int(expr), IntType())
            generated_ops.append(const)
            return const.result

        if isinstance(expr, Symbol):
            name = str(expr)
            if name in self._scalar_symbol_cache:
                return self._scalar_symbol_cache[name]
            external = ExternalSymbolOp(name, IntType())
            external.result.name_hint = name
            generated_ops.append(external)
            self._scalar_symbol_cache[name] = external.result
            return external.result

        if getattr(expr, "is_Add", False):
            args = list(expr.args)
            acc = self._expr_to_value(args[0], generated_ops)
            for arg in args[1:]:
                rhs = self._expr_to_value(arg, generated_ops)
                add = AddOp(acc, rhs)
                generated_ops.append(add)
                acc = add.result
            return acc

        if getattr(expr, "is_Mul", False):
            args = list(expr.args)
            acc = self._expr_to_value(args[0], generated_ops)
            for arg in args[1:]:
                rhs = self._expr_to_value(arg, generated_ops)
                mul = MulOp(acc, rhs)
                generated_ops.append(mul)
                acc = mul.result
            return acc

        if getattr(expr, "is_Pow", False):
            base = self._expr_to_value(expr.args[0], generated_ops)
            exp = self._expr_to_value(expr.args[1], generated_ops)
            pow_op = PowOp(base, exp)
            generated_ops.append(pow_op)
            return pow_op.result

        fallback = ExternalSymbolOp(str(expr), IntType())
        fallback.result.name_hint = str(expr)
        generated_ops.append(fallback)
        self._scalar_symbol_cache[str(expr)] = fallback.result
        return fallback.result

    def _refresh_array_like_result_type(self, op: Op) -> None:
        if op.num_operands == 0:
            return

        array_src = op.operands[0].source
        if not isinstance(array_src.type, ArrayType):
            return
        old_dims = list(array_src.type.dims)
        indices = [operand.source for operand in op.operands[1:]]

        if isinstance(op, ArrayRefOp):
            new_base_dims: List[SymExpr] = []
            extra_dims: List[SymExpr] = []
            for i, idx in enumerate(indices):
                if i < len(old_dims) and self._is_placeholder_value(idx):
                    new_base_dims.append(old_dims[i])
                if isinstance(idx.type, ArrayType):
                    extra_dims.extend(idx.type.dims)
            if len(indices) < len(old_dims):
                new_base_dims.extend(old_dims[len(indices):])
            op.result.type = ArrayType(extra_dims + new_base_dims, array_src.type.datatype)
            return

        if isinstance(op, LoadOp):
            preserved_dims: List[SymExpr] = []
            for i, idx in enumerate(indices):
                if i < len(old_dims) and self._is_placeholder_value(idx):
                    preserved_dims.append(old_dims[i])
            if len(indices) < len(old_dims):
                preserved_dims.extend(old_dims[len(indices):])

            if len(preserved_dims) == 0:
                op.result.type = array_src.type.datatype
            else:
                op.result.type = ArrayType(preserved_dims, array_src.type.datatype)

    def _contains_varlen(self, expr: Any) -> bool:
        try:
            return bool(sympify(expr).has(Function("varlen")))
        except Exception:
            return "varlen(" in str(expr)

    def _is_one_expr(self, expr: Any) -> bool:
        try:
            normalized = simplify(sympify(expr))
            if normalized == 1:
                return True
            if getattr(normalized, "is_number", False):
                return bool(normalized.equals(1))
            return False
        except Exception:
            return str(expr).strip() == "1"

    def _is_blk_scaled_one_expr(self, expr: Any) -> bool:
        """Check if expr * BLK_K simplifies to 1 (i.e. expr = 1/BLK_K).
        This detects stride-scaling that becomes identity at block level."""
        try:
            return simplify(sympify(expr) * sympify("BLK_K") - 1) == 0
        except Exception:
            return False

    def _is_zero_expr(self, expr: Any) -> bool:
        try:
            return simplify(sympify(expr)) == 0
        except Exception:
            return False


    def _dims_equal(self, dims_a: Sequence[Any], dims_b: Sequence[Any]) -> bool:
        if len(dims_a) != len(dims_b):
            return False
        return all(str(a) == str(b) for a, b in zip(dims_a, dims_b))

    def _is_placeholder_value(self, value: Value) -> bool:
        return getattr(value, "name_hint", None) == "_"

        
