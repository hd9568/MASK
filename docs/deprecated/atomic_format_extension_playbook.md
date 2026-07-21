# 原子格式扩展与高性能开发参考手册

## 1. 目标

这份手册总结了 `COO / CSR / ELL` 三条原子格式开发路径中的共性做法、关键分叉点和性能闭环方法，目标是让后续继续扩展新的原子格式时：

- 可以快速判断“这个格式应该在哪些层改”
- 可以避免只改名字、不改真实计算语义
- 可以形成稳定的“修改 -> 生成 -> 编译 -> correctness -> 性能 -> 回退/收敛”闭环
- 可以在一开始就按高性能实现的思路开发，而不是功能打通后再完全返工

适用对象：

- 新增 `AtomicFormatType` 成员
- 新增 `format -> RTS -> computent -> OPIR -> NVIR -> host bridge -> codegen` 全链路
- 对新格式继续做 pipeline / fusion / restore 优化

## 2. 先回答的三个问题

在开始写代码前，先回答下面三个问题。后面的实现都要围绕这三个问题收敛。

### 2.1 这个格式的真实数据语义是什么

必须明确：

- 每个 block 的 metadata 是什么
- payload 是按什么顺序组织的
- 是否有 per-row / per-col / per-segment 结构
- kernel 恢复 tile 时需要哪些输入

例子：

- `COO`: 只需要 `[idx, val, range]`
- `CSR`: 除了 `[idx, val, range]` 还需要 `row_ptr`
- `ELL`: 需要 `ell_len` 约束 row-major 组织，但不需要 `row_ptr`

如果这一步回答不清，后面很容易出现“名字是新格式，运行时其实还是 COO”的问题。

### 2.2 这个格式的身份主要体现在哪一层

经验上，新格式身份通常同时体现在三层：

- `host payload 组织`
- `IR 中的操作数签名`
- `kernel restore / load` 的约束关系

注意：

- 并不是 restore 循环长得和 COO 相似，就等于退化成 COO
- 关键是 host 端是否真的按新格式组织数据，以及 kernel 输入签名是否真的依赖新 metadata

### 2.3 这个格式的性能风险点在哪里

新增原子格式最常见的性能损失来源不是 DRAM 总字节数，而是：

- 多余 metadata 物化
- 不必要的 gmem -> smem copy
- pipeline stage 拓扑更差
- restore/loader 之间多出一层中间 op
- naive 的逐行/逐段循环破坏 warp 利用率

所以新增格式时要尽早考虑：

- 哪些 metadata 只是语义存在，运行时不必真的物化
- 哪些索引计算可以 fusion / inline
- 哪些 op 应该和参考算子一样对齐到同一 stage

## 3. 全链路改动清单

新增一种原子格式时，建议按下面顺序改。每一步都应该有明确的产物和验证点。

### 3.1 format 层

目标：

- 定义新的 `AtomicFormat` 子类
- 提供工厂函数
- 在格式树中挂上新的叶子格式

典型文件：

- `python/sparsene/format/format.py`
- `python/sparsene/formats/<NEW>_TCF.py`

检查项：

- `atomic_type` 名称唯一
- 叶子轴的含义和新格式语义一致
- 格式树与现有 `ME_TCF / CSR_TCF / ELL_TCF` 同构或有意分叉

### 3.2 RTS / transformation 层

目标：

- 为新格式增加 transformation 类型
- 在 `derive_rts()` 中把格式名映射到新 transformation

典型文件：

- `python/sparsene/transform/transformation.py`
- `python/sparsene/transform/rts.py`

检查项：

- 格式字符串能正确触发新 transformation
- RTS 顺序不会把新 transformation 放错位置

### 3.3 computent 层

目标：

- 为新格式生成新的数组定义
- 明确 schedule args 的数量与顺序
- 把格式语义第一次“真正落地”为数组和 metadata 契约

典型文件：

- `python/sparsene/op_gen/computent/schedule.py`
- `python/sparsene/op_gen/computent/computent.py`

必须明确：

- 需要哪些数组：`len / idx / val / row_ptr / mask / ...`
- 每个数组的 `ArrayType`
- 每个数组的维度和 varlen 映射
- `AtomicFormatOp.args` 的顺序

经验：

- `schedule args` 是后续 OPIR/NVIR 的主契约，顺序一旦定义，后面所有层都要与之对齐
- 如果新格式有专属 metadata，不要“先借用 COO 的 args 结构，后面再补”，这样很容易把语义做歪

### 3.4 OPIR 层

目标：

- 为新格式定义独立的 restore op
- 在 OPIR 生成阶段，把 schedule args 转换为真正的 load/restore/mma 链路

典型文件：

- `python/sparsene/op_gen/opir/ops.py`
- `python/sparsene/op_gen/opir/generate.py`

检查项：

- 新 restore op 是否有独立名字
- 操作数个数是否真的体现新格式
- `generate_from_<new>_atomic_format_op()` 是否单独存在

经验：

- 即使 load 部分暂时复用 `CooAtomicFormatLoadIdxOp / LoadValOp`，也要保证 restore op 是独立的
- “独立 restore op + 独立 operands 签名” 是区分格式身份的重要证据

### 3.5 NVIR / codegen 层

目标：

- 为新格式注册 gmem 接口
- 补齐 logical kind dispatch
- 生成真正的 kernel 实现

典型文件：

- `python/sparsene/op_gen/nvir/generate.py`
- `python/sparsene/op_gen/nvir/compiler_driver.py`

检查项：

- `SPARSE_FORMAT_REGISTRY` 中是否注册了新格式
- `shape_overrides / dtypes / custom_layouts` 是否正确
- `_classify_logical_kind()` 是否能识别新 restore op
- `_choose_output_mem() / _choose_op_type()` 是否覆盖到新格式
- restore 代码是否真的使用了新格式需要的 operands

经验：

- 新格式第一次打通时，先保证 `correctness`
- 第二阶段再看哪些 metadata 只是“语义存在”，其实可以不物化或不参与热点路径
- 不要一开始就默认所有 metadata 都要进 smem

### 3.6 host bridge 层

目标：

- 把 reference/testbed 现有格式的数据，转换成新格式实际需要的 metadata 和 payload
- 这是“最终计算真的是新格式”的关键层

典型文件：

- `examples/src_fp32/dtc/testbed/host_program_generated_<new>.cu`

必须明确：

- 输入来源是什么
- 输出 metadata 是什么
- payload 如何重排
- sentinel / padding / row_ptr / mask 的规则是什么

经验：

- 新格式是否“真的成立”，最容易在 host bridge 看出来
- 如果 host 只是简单复制 `COO idx/val`，那大概率只是名字上分叉了

## 4. 高性能开发的闭环流程

建议每次都严格按下面顺序做，不要只改完代码就下结论。

1. 改代码
2. 重新生成 `.inc`
3. 编译 testbed
4. 用 `shipsec1` 跑 correctness
5. 跑性能
6. 对比参考 `kernel.inc` / `me_tcf`
7. 如退化，立刻回退并记录原因

这套流程的核心目的不是“多跑几遍”，而是确保每次性能变化都能追溯到具体改动。

## 5. 新格式必须准备的证据

当你说“新格式已经真正打通”时，至少要拿出下面几类证据。

### 5.1 结构证据

- `AtomicFormatType` 新成员
- 独立 transformation
- 独立 OPIR restore op
- 独立 NVIR logical kind

### 5.2 数据证据

- host bridge 真的生成了新 metadata
- payload 顺序和新格式语义一致

### 5.3 运行时证据

- correctness 通过
- kernel 参数列表中出现了新格式需要的 metadata
- `.inc` 中能看到新格式专属 op 或参数签名

### 5.4 性能证据

- 新格式和参考格式的时间对比
- `.inc` 中 `op_call / commit / wait` 对比
- 必要时用 `ncu` 看 cycles / dram / inst / active warps

## 6. 常见陷阱

### 6.1 只改名字，不改 payload

症状：

- IR 里已经有新名字
- 但 host 端还是原样复制 COO 数据

后果：

- correctness 可能过
- 但格式身份不真实，后续优化也无从谈起

### 6.2 语义正确，但热点路径过于保守

症状：

- 恢复逻辑写成逐行/逐段串行循环
- row_ptr / ell_len 等 metadata 每次都参与大量无效整数计算

后果：

- correctness 过
- 但 warp 利用率差，性能远低于参考实现

### 6.3 为了格式纯度强行 padding

症状：

- ELL 类格式严格做 padding
- kernel 需要处理大量无效 entry

后果：

- 语义看起来“更标准”
- 实际性能很差

经验：

- 如果框架允许，优先保留“格式语义”而不是“教科书式存储形式”
- 例如 row-major 紧凑排布，往往比强制 padding 更适合高性能实现

### 6.4 pipeline pass 直接套用到新格式

症状：

- 对新格式直接复用 COO/ELL 的 fusion 规则
- 忽略特殊 no-op alias 或额外 metadata consumer

后果：

- stale reference
- codegen 断言
- 性能没有提升，甚至编译失败

经验：

- 任何 pipeline/fusion 优化都要先检查依赖图
- 只有在消费者集合、输入签名和 stage 依赖都吻合时，才能套用

## 7. 推荐的性能分析顺序

建议按下面顺序排查，不要一上来就怀疑带宽。

1. 看 correctness 是否已经稳定
2. 看 host 转换是否被错误计入 kernel 计时
3. 对比 `.inc` 的 `op_call / commit / wait`
4. 对比 stage 划分是否贴近参考 `kernel.inc`
5. 再用 `ncu` 看：
   - `sm__cycles_elapsed.avg`
   - `dram__bytes.sum`
   - `sm__inst_executed.avg`
   - `sm__warps_active.avg.pct_of_peak_sustained_elapsed`

经验：

- 如果 `dram` 和 occupancy 接近，但 `cycles` 更高，通常是 pipeline bubble
- 如果多出的是 metadata 相关中间 op，优先尝试 fusion / inline / 去物化

## 8. 新格式开发模板

建议新增格式时，按下面模板建立任务单。

### 8.1 功能打通阶段

- 定义 format class 和 format tree
- 新增 transformation 和 RTS 映射
- 新增 `AtomicFormatType`
- 在 computent 中定义数组和 schedule args
- 在 OPIR 中新增 restore op 和生成函数
- 在 NVIR 中注册 gmem 接口和 logical kind
- 写 host bridge 完成 payload 转换
- 重新生成 `.inc` 并编译
- 用 `shipsec1` 做 correctness

### 8.2 性能收敛阶段

- 对比参考 `.inc`
- 定位多余 op 和 stage 差异
- 优化 restore / load / metadata 物化
- 做 fusion / inline / stage 对齐
- 每一步都重新生成、编译、跑 correctness 和性能

### 8.3 文档沉淀阶段

- 记录格式语义
- 记录必须改动的文件
- 记录关键 bug 和修复
- 记录最终性能数据
- 记录哪些“看起来合理但实际退化”的尝试

## 9. 当前项目中的经验结论

基于 `CSR / ELL` 的实际开发，当前项目中最重要的经验是：

- 新格式的“身份”优先体现在 `host payload 组织 + IR 操作数签名`
- 热点 restore 路径可以在不破坏格式语义的前提下，收敛到更高效的统一骨架
- 大部分剩余性能差距通常不是来自 DRAM，而是来自 pipeline 拓扑和多余依赖
- 只要 `host metadata` 和 `kernel contract` 是真的，新格式即使和 COO 共用部分 load/restore 骨架，也不等于退化成 COO

## 10. 推荐搭配材料

开发新格式时，建议同时参考：

- [ell_atomic_format_development.md](file:///home/huoda/sparsene/docs/ell_atomic_format_development.md)
- [atomic_format_analysis.md](file:///home/huoda/sparsene/docs/atomic_format_analysis.md)
- [sweep_tc_block_k_csr.sh](file:///home/huoda/sparsene/scripts/sweep_tc_block_k_csr.sh)
- [sweep_tc_block_k_ell.sh](file:///home/huoda/sparsene/scripts/sweep_tc_block_k_ell.sh)

如果后续新增 `HYB / DIA / SELL-C-sigma / BLOCKED-ELL` 等格式，建议直接以本手册为 checklist 开发，而不是从单点文件开始猜测。
