# Sparsene 已声明格式与设计语义

本目录中的 Python 对象是层次化 `Format` 声明。它们是格式设计智能体的
参考构件，但“能够声明/推导 RTS”不等于已经具备 host preprocessing、完整
metadata ABI、CUDA lowering、数值正确性和性能验证。

机器可读、会注入格式智能体提示词的同源知识位于
`design_catalog.py`。修改本目录格式时，应同步更新该 catalog 和格式智能体
测试。

## 性能评价边界

本项目当前主性能指标是：格式转换、padding/metadata 构造和行列重排均已完成
之后，使用 CUDA event 测得的 warmed GPU SpMM 计算 kernel 平均时间。

- 不计入主指标：格式转换、重排及正确性所需的逆置换、checkpoint、内存申请、
  H2D/D2H、代码生成/编译和结果校验；cuSPARSE 的 preprocess 也在计时外。
- 计入主指标：计时 kernel 内真实执行的 metadata 读取、间接 B gather、atomic
  restore/scatter、padding 零工作、同步、MMA/累加和 C 写回。
- 格式设计的主排名不得因为离线转换/重排较慢而降级候选；但仍必须完整定义
  置换/逆映射、metadata、额外存储、缓存身份和实现路径，作为独立工程信息。

因此本文档提到“预处理/重排成本”时，含义是可实现性和独立工程记录，不是
要把该时间加入当前 kernel-only 性能比较。

## 公共 DSL 语义

- `df`：稠密、固定长度；常用于规则切分和硬件 tile。
- `dv`：稠密、每个父节点实现长度可变。
- `sf`：稀疏选择、固定逻辑边界。
- `sv`：稀疏选择、每个父节点实现长度可变。
- `length` 是逻辑域上界或 split factor，不是必须静态等于“实际保存坐标数”。
  例如 `sv(Y_s, COL, K)` 的实际窗口内凝聚列数由变长元数据给出，`K` 是
  逻辑列域上界。因此不要把长度写成 `varies per window`。
- 相邻同方向轴表达切分。例如源码通过 `X_o(M)` 与 `X_i(BLK_M)` 表达
  行窗口切分，不需要写 `ceil(M/BLK_M)`。
- `Axis.reorder` 是显式稳定置换承诺；稀疏轴、局部列凝聚和排序都不自动
  等价于全局矩阵重排。

## 源码格式

| 源码声明 | 层次 | Atomic | 核心思想 | 当前事实/限制 |
|---|---|---|---|---|
| `DTC_SpMM.py::ME_TCF_FORMAT` | 两层 TCF | COO | BLK_M 行窗口内凝聚触及列，再按 BLK_K 切分；tile 内保存局部坐标和值 | 当前统一 DTC 路径只接收一致的 16x8/16x16 profile。局部凝聚不是全局重排 |
| `Acc_SpMM.py::BIT_TCF_FORMAT` | 两层 TCF | MCO | 保留 ME-TCF 分组，tile 内用 bitmask+紧凑 values 降低坐标元数据 | 常量名是 BIT-TCF，但源码 `Format.name` 仍是 `DTC-SpMM` |
| `CSR_TCF.py::CSR_TCF_FORMAT` | 两层 TCF | CSR | tile 内用 row_ptr+col_idx+values，适配局部行向遍历/行长度偏斜 | 需要专用 payload、restore、host 与数值验证 |
| `ELL_TCF.py::ELL_TCF_FORMAT` | 两层 TCF | ELL | tile 内每行填充到局部最大长度，行长度规则时访存整齐 | 长尾会造成 padding；需定义 ell_len 和 sentinel |
| `DIA_TCF.py::DIA_TCF_FORMAT` | 两层 TCF | DIA | tile 内保存少量对角 offset 和按对角填充的 values | 全局带状性不保证凝聚后 tile 仍少对角 |
| `SR_BCRS.py::SR_BCRS_FORMAT` | 两层 TCF | dense | 仅选择 active blocks，块内直接 dense restore/compute | 低结构密度会产生大量零填充与零计算 |
| `Spaden.py::BIT_BSR_FORMAT` | 单层 active block | MCO | 不做逐列凝聚，直接保存每个块行的活跃固定列块，块内使用 mask | `Y_o_s` 是稀疏活跃块选择，不是全网格 dense BSR |
| `HME_TCF.py::HME_TCF_FORMAT` | 五层多尺度 TCF | COO | 外层 TCF tile 内再次稀疏化并切为 BLK_M_I×BLK_K_I 子 tile | 源码注明 RTS 不稳定，属于实验声明，不能声称端到端可执行 |
| `ROW_REORDER_SR_BCRS.py::ROW_REORDER_SR_BCRS_FORMAT` | 稀疏行域+行块+active block | dense | 先表达稀疏变长行域，再做列凝聚和 dense block | 名称含 row reorder，但源码 `Axis.reorder` 全为 `None`；真正重排的置换和逆映射尚未由 DSL 声明 |
| `dense.py::DENSE_FORMAT/DENSE_T_FORMAT` | 单层稠密操作数布局 | 无 atomic child | 通过 ROW/COL 轴顺序表达普通与转置遍历 | 是操作数布局构件，不是格式智能体可原样返回的完整稀疏设计 |

## 如何在这些范式上发散

格式设计不是从上述名称中分类，而是组合四个相互独立的决策：

1. 分组：固定行窗口、稀疏行域、块行、多尺度或显式重排分组。
2. 列表示：逐列凝聚、固定活跃列块、对角集合或自定义分区。
3. 层次：单层、两层 TC block、HME 子 tile，或有证据的新层次。
4. 叶子：COO/MCO/CSR/ELL/DIA/dense/custom。

增加层次或混合编码必须说明 kernel 内 metadata、padding、restore、负载均衡，
并单独说明计时外的预处理和逆映射要求。合法的研究 DSL 可以尚未有后端，但不能把“尚未实现”和
“DSL 非法”混在一起。
