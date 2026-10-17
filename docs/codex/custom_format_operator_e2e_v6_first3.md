# v6 自定义 Format DSL 到算子端到端验证（前三个数据集）

> 2026-07-19 更新：本文是前三个数据集阶段记录；其中“尚未完成”部分已由
> `custom_format_operator_e2e_v6_all26_n512.md` 的 26/26、N=512 闭环取代。

日期：2026-07-18  
格式设计运行：`20260718_165146_deepseek_v4_pro_format26_v6`  
目标硬件：NVIDIA A800 80GB PCIe（SM80）

## 1. 本次范围与结论

本次没有再次调用付费模型。已将该运行目录下 26 份
`formats/<dataset>.format.json` 全部导出为
`python/sparsene/formats/custom/*.py`，并对运行清单中的前三个数据集完成：

`JSON Format DSL -> Format -> RTS -> Computent -> OPIR -> Varlen/CVal lowering -> NVIR -> semantic fusion/pipeline -> .inc -> NVCC -> A800 -> cuSPARSE correctness/performance`

三个设计均通过完整数值检查，`mismatches=0`。DD 的生成算子在本次
`N=64` 测量中略快于 cuSPARSE；ASIC_680k 与 Ga41As41H72 正确但仍慢于
cuSPARSE，因此不能宣称三个格式都已经达到高性能。

## 2. 26 个格式的离线、可审计表示

新增的公共 DSL 层 `python/sparsene/formats/dsl.py` 不依赖 Agent/API，负责
受限长度表达式、结构验证和 `Format` 树构造。导出脚本为：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
PYTHONPATH=python python python/test/export_custom_formats.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6
```

输出目录是 `python/sparsene/formats/custom`，其中：

- 每个数据集一个 Python 模块，包含完整 `FORMAT_SPEC`、`FORMAT`、`PARAMS` 和来源；
- `index.json` 保存运行 ID 和数据集到合法 Python 模块名的映射；
- `loader.py` 按原始数据集名加载，文件名清洗不会改变数据集身份；
- 26/26 模块均已实际导入、重新验证 DSL 并成功派生 RTS。

这些模块保存的是 Agent 原始设计语义，不会按当前后端能力偷偷改写成 ME-TCF。

## 3. 语义后端选择，而不是按名字硬编码

`python/sparsene/op_gen/backend_registry.py` 根据层次、axis 类型/方向、atomic
映射、layout、参数和 reorder 语义选择后端，不读取数据集名，也不依赖
`format_name` 的描述文本。目前完成的通用后端是
`tcf_mco_tensorcore_v1`：

- 两层 row-window column-condensation TCF；
- 最深层为 row-major MCO，`x_axis=X_i`、`y_axis=Y_si`；
- SM80 TF32 MMA 为 `m16n8k8`；
- 逻辑 `BLK_M` 支持 8/16，`BLK_K` 支持 8 的正整数倍；
- `row_rcm` 是可执行的预处理契约，不再因 RTS 忽略 `Axis.reorder` 而静默丢失；
- 逻辑 tile、物理 tile、MMA tile、Tile-N 和执行分段上限均写入后端计划。

同一份格式即使换一个全新的 `format_name`，只要物理语义不变，仍会选择同一
后端。未来迁移硬件时应新增 target/backend plan，而不是在数据集分支中更改
CUDA 常量。

前三个格式的映射为：

| 数据集 | Agent DSL | 逻辑 tile | A800 物理 tile | reorder |
|---|---|---:|---:|---|
| ASIC_680k | BIT-TCF-8x16 + MCO | 8×16 | 16×16 | 无 |
| DD | TCF-16x16-MCO-RowRCM | 16×16 | 16×16 | 真实 row_rcm |
| Ga41As41H72 | H-MCO-8x32 | 8×32 | 16×32 | 无 |

`BLK_M=8` 没有被改写为 16 行凝聚。预处理仍按每 8 行独立建立唯列序列和
MCO tile，只在物理 MMA A tile 的后 8 行补零。补零产生的所有 GPU 工作都在
计时内。

## 4. 发现并修复的跨层问题

### 4.1 BIT-TCF 的 BLK_K>8 实际只计算首个 K fragment

旧 ACC/BIT-TCF 测试床只实例化 `16×64×8`。NVIR 语义融合又把 A/B 寄存器
形状的 K 维压掉，`ldmatrix` 和 MMA 都没有 K 循环。因此 `BLK_K=16/32`
虽然能够生成源码，实际上只计算前 8 列。

修复位于 `compiler_driver.py`：A/B 寄存器形状保留
`BLK_MMA_MNK.k`，A/B `ldmatrix` 和 TF32 MMA 显式遍历 K fragment。DD 的
16 和 Ga 的 32 都已经通过真机数值验证。

### 4.2 `BLK_M=8` 与 SM80 m16 store/ldmatrix 不兼容

旧实现固定用 m16n8k8 MMA 和写 16 行 C；直接实例化 `BLK_M=8` 会出现非法
形状或越界。通用后端将“DSL 逻辑分组”与“硬件执行 tile”分开：8 行逻辑
窗口在 MCO mask 中物理补为 16 行，kernel 输出先写物理行，再在计时外恢复为
原始 M 行。物理补零是被测 kernel 工作，不能从性能结果中扣除。

### 4.3 DD 的 `row_rcm` 过去没有被执行

`derive_rts()` 当前不会为 `Axis.reorder` 生成转换。新 host bridge 在格式转换
阶段对方阵的对称化结构图执行确定性 Reverse Cuthill-McKee，映射定义为
`new_row -> original_row`；列和 B 不重排，因为 DSL 只在外层 X_o 请求行重排。
kernel 后在计时外按逆映射恢复原输出行，再与原 CSR 的 cuSPARSE 结果比较。

这与固定性能合同一致：重排/逆映射不计入 kernel-only 时间，但必须真实执行、
可复现并通过正确性。

### 4.4 大矩阵超过 `grid.y=65535`

ASIC_680k 有 85,358 个逻辑行窗口，旧 `grid(N/TileN, row_windows)` 会以
`invalid configuration argument` 失败。host bridge 现在将执行窗口分为多个
launch；同一次被测迭代包含全部 launch。元数据使用全局绝对 tile offset，C
指针按物理行段平移，格式语义没有变化。

### 4.5 ASIC 的长尾窗口使一个 CUDA block 串行 24,705 个 tile

ASIC 的 8×16 特征中，窗口 TC-block 数中位数为 1，但最大值是 24,705，CV
为 53.75。原生成 kernel 的一个 CUDA block 对应一个逻辑行窗口，因此该异常
窗口形成约 45 ms 的串行长尾。

通用执行后端按“每段最多 32 个 tile”将长窗口调度成多个 partial-C 段，再由
GPU reduction 合并为原逻辑窗口。分段不改变 TCF/MCO 存储语义；partial 计算
和 reduction 都在 CUDA event 内。没有长窗口时直接写最终 C，不支付 reduction
开销。该上限已经作为后端/编译参数暴露，后续应由策略智能体依据硬件与窗口
分布调优。

ASIC 由此从 44.94 ms 降到 1.484 ms，约 30.3×，且仍为零 mismatch。

### 4.6 旧 host 的验证与性能口径不足

旧 ACC host 固定 include、固定 16×8、最多打印 200 个错误且错误时仍返回 0，
cuSPARSE 也没有形成对称的 warmed event timing。新增
`custom_format_e2e.cu`：

- 编译期 include 指定的生成 `.inc` 和后端计划；
- 完整扫描所有输出，失败返回码为 2；
- 使用确定性的结构测试值 1..5，避免矩阵原值量纲掩盖 TF32 检查；
- 记录 mismatch、最大绝对误差、相对 L2 误差和误差位置；
- 生成 kernel 与 cuSPARSE 都使用 10 次 warmup、100 次 CUDA-event 重复；
- cuSPARSE descriptor、bufferSize 和 preprocess 全在计时外；
- 严格记录 `sparsene.kernel_only_post_conversion.v1`。

## 5. 生成物、编译和运行入口

算子生成命令：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
PYTHONPATH=python python python/test/generate_custom_format_operators.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6 \
  --first 3 --tile-n 64
```

生成文件严格采用：

`<dataset>_20260718_165146_deepseek_v4_pro_format26_v6.inc`

位置为每个数据集的 `datasets/<dataset>/operators/`。三个 `.inc` 的 SHA256
相同是预期结果：它是同一 TCF-MCO 语义模板，具体 8×16、16×16、8×32 由
后端计划和 CMake 编译期模板参数实例化，不代表三个格式被改成同一个物理参数。

统一端到端命令：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
PYTHONPATH=python python python/test/run_custom_format_e2e.py \
  logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6 \
  --first 3 --n 64 --warmup 10 --repeat 100 --skip-generate
```

该脚本逐数据集保存 CMake 配置、NVCC 输出、运行 stdout、生成清单和结果 JSON，
并在 run 根目录维护累计 CSV/JSON 汇总。

## 6. A800 最终结果

以下均为 `N=64`、warmup=10、repeat=100、格式转换后 kernel-only 时间：

| 数据集 | 生成算子 ms | cuSPARSE ms | cuSPARSE/生成 | mismatch | 相对 L2 误差 |
|---|---:|---:|---:|---:|---:|
| ASIC_680k | 1.48418 | 0.783176 | 0.527684× | 0 | 1.67909e-4 |
| DD | 0.303606 | 0.310159 | 1.02159× | 0 | 1.82677e-4 |
| Ga41As41H72 | 3.58360 | 2.07745 | 0.579710× | 0 | 1.58875e-4 |

结果来源：

- `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/operator_e2e_summary.json`
- `logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/operator_e2e_summary.csv`
- 各数据集 `operator_generation/e2e_result.json`

## 7. 验证记录

- 26/26 custom 模块：导入成功、DSL valid、RTS 非空；
- 前三个：语义后端计划断言通过，修改 `format_name` 不改变选择；
- `python -m compileall` 覆盖新增 Python 运行链；
- `python/test/test_format_design_agent.py`：13/13 通过；
- 三个 `.inc` 均由 NVCC 12.8 以 `sm_80` 编译；
- 三个真实 MatrixMarket 数据均在 A800 上完整输出比较并通过。

当前 venv 没有 `pytest` 模块，因此新增 pytest 风格的
`test_custom_format_runtime.py` 未由 pytest runner 执行；其等价断言已经用同一
venv 直接执行通过。本次没有安装新依赖。

## 8. 尚未完成与下一步

1. 当前通用 e2e 后端只覆盖两层 column-condensed TCF + row-major MCO；其余
   23 个模块已可离线导入，但不能据此声称都能执行。应逐一由语义 matcher
   分类，再补 BIT-BSR、其他层次、atomic 和 reorder 后端。
2. SM80 路径目前只支持逻辑 BLK_M=8/16 和 BLK_K 为 8 的倍数；更大 M tile
   需要多 warp 的 S2R/MMA/store，而不是继续补零。
3. `row_rcm` 当前要求方阵；矩形矩阵需要明确 row-similarity graph 定义。
4. ASIC 与 Ga 仍未超过 cuSPARSE。后续策略智能体应调优 Tile-N、长窗口分段、
   pipeline/寄存器压力，并用 NCU 区分补零、MCO restore、B gather、partial-C
   reduction 和内存吞吐瓶颈。
5. 未来硬件迁移应新增 target 描述（MMA shape、warp、shared-memory、grid
   限制、合法 swizzle）并选择对应 lowering，不应把 A800 假设扩散到 DSL 或
   数据集模块。
