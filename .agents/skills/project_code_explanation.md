# Sparsene 项目代码主线详解

本文以代码为主、论文为辅，解释这个项目到底在做什么，`inc` 文件分别扮演什么角色，以及你后续开发时最应该先抓住哪些主线。

## 1. 这个项目在做什么

一句话概括：

- 这个项目想把“一个层次化稀疏格式的描述”自动降到“可执行的 CUDA SpMM 内核”。

它不是简单地“给一个 CSR 然后生成一个 kernel”。
它做的是更强的一件事：

- 先用一套格式 DSL 描述稀疏格式的层次结构。
- 再从格式描述推导出遍历/调度逻辑。
- 再把调度逻辑降成中间表示。
- 再做变长轴消除、`C` 写回扁平化、张量布局绑定、流水线划分。
- 最后生成 `.inc` 形式的 CUDA/CUTLASS/CuTe 风格内核源码。

项目自己的 README 已经把这条链路写得很直白：

```text
format -> rts -> computent -> opir -> opgraph -> nvir -> cuda source
```

对应代码入口在 `python/sparsene/op_gen/README.md`。

## 2. 你真正要先理解的核心对象

### 2.1 `Format`

定义位置：

- `python/sparsene/format/format.py`

这里定义了最核心的格式抽象：

- `Axis`
- `Format`
- `AtomicFormat`
- `coo_atomic_format()`
- `mco_atomic_format()`
- `dense_atomic_format()`

`Axis` 有四个关键属性：

- 行/列方向 `direction`
- 是否稀疏 `is_sparse`
- 是否变长 `is_varlen`
- 轴长度 `length`

所以它不是在描述“数学张量循环”，而是在描述“物理存储层次和它的轴属性”。

这是整个项目和 TACO/MLIR sparse tensor 那类“迭代中心表示”最不一样的地方。

### 2.2 `ME_TCF_FORMAT`

定义位置：

- `python/sparsene/formats/DTC_SpMM.py`

核心定义：

```python
ME_TCF_FORMAT = Format(
    name="ME-TCF",
    axes=[
        df_axis("X_o", ROW, length="M"),
        sv_axis("Y_s", COL, length="K"),
        df_axis("X_i", ROW, length="BLK_M"),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", COL, length=Symbol("K") / Symbol("BLK_K")),
            df_axis("Y_si", COL, length="BLK_K"),
            df_axis("X_i", ROW, length="BLK_M"),
        ],
        child=coo_atomic_format("X_i", "Y_si"),
    ),
)
```

这段定义非常重要，因为它已经把 ME-TCF 的层次说清楚了：

- 外层按行块 `X_o` 分解。
- `K` 方向先看成稀疏变长轴 `Y_s`。
- 行块内部有一个固定长度 `X_i = BLK_M`。
- 到 child level 后，又把 `Y_s` 分成 `Y_so` 和 `Y_si=BLK_K`。
- 最底层原子格式是 COO。

直觉上就是：

- 先按行窗口看。
- 再把 `K` 方向压缩成若干 Tensor Core 友好的小块。
- 每个最小块内用 COO 表示非零并恢复成稠密 tile 参与 MMA。

## 3. 前端是怎么从格式推导出“变换序列”的

定义位置：

- `python/sparsene/transform/rts.py`

这里的 `derive_rts(format)` 会把 `Format` 推导成一个变换序列 `TransformationSequence`。

它干了几件事：

- `_densify()`：把格式里“逻辑上的稀疏轴”转换成后续要显式处理的 sparse transformation。
- `_squeeze_new()`：处理相邻同方向轴的合并/拆分逻辑。
- `_bubble_up()`：通过 `SwapTransformation` 把轴重排到更适合父层 major direction 的顺序。
- 递归进入 child format，插入 `SpawnTransformation()`。
- 到最底层 atomic format 时插入 `CooizeTransformation()` 或 `McoizeTransformation()`。

你可以把它理解成：

- `Format` 是“声明式层次结构”
- `RTS` 是“把这个层次结构真正展开成一条 lowering 路线”

## 4. `computent` 在干什么

定义位置：

- `python/sparsene/op_gen/computent/computent.py`

入口函数：

- `computent_from_rts(name, rts)`

它是把 `rts` 真正执行到“稀疏数据结构 + 调度 + 数组定义”上的那一步。

最关键的输出有三类：

- `schedule`
- `array_defs`
- `format`

也就是：

- 现在有哪些数组
- 这些数组各自的维度是什么
- 之后的循环/访存/atomic format 恢复逻辑怎么走

对 `ME_TCF` 来说，运行后会得到这类数组：

- `val_sidx`
- `val_len`
- `B_val`
- `C_val`
- `val_coo_idx`
- `val_coo_val`
- `val_coo_len`

这里已经能看出生成内核需要的元数据雏形了。

## 5. `opir` 是什么

定义位置：

- `python/sparsene/op_gen/opir/generate.py`

入口：

- `generate_from_computent(computent)`

这一步把 `computent` 转成更接近“算子图/循环体”的 IR。

在 `opir` 里你会看到这些操作：

- `ForLoopOp`
- `ArrayRefOp`
- `LoadOp`
- `LoadOffsetOp`
- `CooAtomicFormatLoadIdxOp`
- `CooAtomicFormatLoadValOp`
- `CooAtomicValRestoreOp`
- `MmaOp`
- `CValLoadOp`
- `CValStoreOp`

这一步本质上是在表达：

- 当前循环层次是什么
- 哪些元数据怎么读
- 哪些 tile 怎么 gather
- COO 数据怎么 restore 成稠密 tile
- 最后怎么做 MMA 和 C 写回

也就是说，从这里开始，项目已经不只是“格式系统”，而是开始显式表达 kernel 内部的数据通路了。

## 6. 两个非常关键的 lowering pass

### 6.1 `VarlenLoweringPass`

定义位置：

- `python/sparsene/op_gen/opir/varlenLoweringPass.py`

它做的事情特别重要：

- 把 `varlen(K)`、`varlen(nnz)` 这种变长维，变成 offset/prefix-sum 形式的显式数组。

对 `ME_TCF` 来说，这一步之后外部符号会变成：

- `val_sidx_new`
- `val_len_offset`
- `val_coo_idx_new_1`
- `val_coo_val_new_1`
- `val_coo_len_new_offset`

这就是为什么生成内核接口和 testbed 原始 `kernel.inc` 不一样。

原始 testbed 更像 DTC-SpMM 手写内核的接口：

- `val_sidx`
- `val_soff`
- `val_coo_idx`
- `val_coo_val`
- `val_coo_off`

而新生成内核已经被 lowering 成了“更统一的 offset 形式”。

### 6.2 `CValFlattenPass`

定义位置：

- `python/sparsene/op_gen/opir/cValFlattenPass.py`

它负责把层次化 `C_val` 访问统一扁平化到 `[M, N]`。

它解决的问题是：

- 前面循环切片时，`C` 可能还是分层 tile 视角。
- 但最终代码生成、写回和外部接口更适合统一成物理平坦的 `C[M, N]`。

因此这个 pass 会把：

- `ArrayRefOp`
- `CValLoadOp`
- `CValStoreOp`

改写成基于“物理 offset”的访问。

## 7. `nvir` 是什么

定义位置：

- `python/sparsene/op_gen/nvir/generate.py`

入口：

- `generate_nvir(opir, format_name="ME_TCF")`

这里开始进入真正“接近 CUDA 代码生成”的阶段。

这一步会做几件事：

- 建立 Gmem 接口。
- 插入外层 `blk_y_loop` / `blk_x_loop`。
- 识别 `mma`、`coo_atomic_val_restore`、`ldmatrix` 等逻辑 op。
- 绑定每个 op 的输入/输出张量所在 memory level：
  - `gmem`
  - `smem`
  - `rmem`
- 插入需要的 `ldmatrix`。

你可以把 `nvir` 看成：

- “已经基本是一个 CUDA kernel 的骨架”
- 只是还没最终打印成 C++/CUDA 源码

## 8. 代码生成与流水线

定义位置：

- `python/sparsene/op_gen/nvir/compiler_driver.py`

入口：

- `apply_software_pipeline_and_codegen(program, output_path)`

它主要做：

- 自动选择主循环
- 动态划分 pipeline stage
- 必要时包装软件流水线
- 最后调用 `NvIrCodeGenerator()` 输出 `.inc`

这次实际运行 `python/test/test_generate.py` 时，日志里已经显示它把主循环切成了 3 个 stage：

- Stage0: 元数据索引、B gather、COO 载入
- Stage1: COO restore、B 的寄存器加载
- Stage2: A 的寄存器加载、MMA

这和论文讲的“pTask/pipeline plan”在代码里是直接对上的。

## 9. `test_generate.py` 到底干了什么

入口：

- `python/test/test_generate.py`

它完整走了一遍：

1. `derive_rts(ME_TCF_FORMAT)`
2. `computent_from_rts("ME_TCF", rts)`
3. `generate_from_computent(computent)`
4. `VarlenLoweringPass`
5. `CValFlattenPass`
6. `generate_nvir(...)`
7. `apply_software_pipeline_and_codegen(...)`

最后输出：

- `python/me_tcf_kernel3.inc`

所以 `test_generate.py` 不是“单元测试”意义上的 test。

它实际上是：

- 当前生成器的一条完整 demo / smoke path

## 10. `inc` 文件都在干什么

### 10.1 `python/me_tcf_kernel3.inc`

这是你本次最关心的“自动生成内核”。

它的特点：

- 来源是新的 Python 生成链。
- 内核名是 `generated_kernel(...)`。
- 接口已经是 lowering 之后的新元数据：
  - `dval_coo_idx_new_1`
  - `dval_coo_len_new_offset`
  - `dval_coo_val_new_1`
  - `dval_len_offset`
  - `dval_sidx_new`

### 10.2 `examples/src_fp32/dtc/testbed/kernel.inc`

这是 testbed 里“已知理论正确”的 DTC/ME-TCF 内核。

它的特点：

- 内核名是 `dtc_spmm_kernel_fp32_val_idx_bind_stage2(...)`
- 与 `main.cu + host_program.cu` 的元数据接口严格匹配。
- 当前 `examples/src_fp32/dtc/testbed/main.cu` 的正确性对比，就是拿它去和 cuSPARSE 比。

这也是为什么它能直接跑，而 `python/me_tcf_kernel3.inc` 不能直接替换进去。

### 10.3 `kernel_multi_binding.inc` / `kernel_strict_lb.inc`

它们对应论文中的两种负载均衡策略：

- multi-binding
- strict load balance

主循环逻辑和格式基本一致，但 thread block 和 row window 的绑定策略不同。

### 10.4 `me_tcf_kernel_origin.inc` / `test_inc/me_tcf_kernel.inc`

这些更多是历史版本、对照版本、单 tile/编译 smoke 测试版本。

它们很有用，因为你后续调生成器时，可以把这些文件当参考：

- 看一个已知能编/已知能跑的版本
- 对比新生成的 `.inc` 少了什么、命名错了什么、布局错了什么

## 11. testbed 这一侧到底在干什么

### 11.1 `examples/src_fp32/dtc/testbed/main.cu`

它做三件事：

1. 读入 `.mtx` 或随机生成稀疏矩阵
2. 把 CSR 转成 ME-TCF 风格的 host 侧元数据
3. 调 `host_program.cu` 里的 kernel launcher

然后它再调用 `cusparse_spmm_all(...)`，把结果和 cuSPARSE 对比。

### 11.2 `examples/src_fp32/dtc/testbed/host_program.cu`

它负责：

- 把 host 元数据搬到 GPU
- 启动 `kernel.inc` 里的已知正确 kernel
- 把 `C` 拷回 host

也就是说：

- `main.cu` 负责“格式化输入和对比”
- `host_program.cu` 负责“真正启动 kernel”

## 12. 老接口和新接口的关系

通过这次实际跑 `VarlenLoweringPass`，可以把关系总结成：

- `val_soff` -> `val_len_offset`
- `val_sidx` -> `val_sidx_new`
- `val_coo_idx` -> `val_coo_idx_new_1`
- `val_coo_val` -> `val_coo_val_new_1`
- `val_coo_off` -> `val_coo_len_new_offset`

但要注意不是简单重命名。

关键变换有两个：

- `val_sidx_new` 需要从旧的 `[TCBlock_num, BLK_K]` 逻辑，转成生成内核实际按 `[BLK_K, TCBlock_num]` 访问的布局。
- `val_coo_len_new_offset` 不是旧 `val_coo_off` 的完整拷贝，而是只保留每个 TC block 的 padded prefix start，再额外保留末尾一个结束偏移。

## 13. 你作为后续开发者，应该怎么读这个项目

建议顺序：

1. 先读 `python/sparsene/format/format.py`
2. 再读 `python/sparsene/formats/DTC_SpMM.py`
3. 再读 `python/sparsene/transform/rts.py`
4. 再读 `python/sparsene/op_gen/computent/computent.py`
5. 再读 `python/sparsene/op_gen/opir/generate.py`
6. 再读 `VarlenLoweringPass` 和 `CValFlattenPass`
7. 再读 `python/sparsene/op_gen/nvir/generate.py`
8. 最后回头看 `python/me_tcf_kernel3.inc`

这样你会知道：

- 某个类为什么被生成出来
- 某个 offset 数组为什么长这样
- 某个 `array_ref` / `coo_atomic_val_restore` / `mma` 是从哪层语义降下来的

## 14. 对你最重要的一句总结

这个项目不是“写几个 CUDA kernel”的仓库，而是：

- 用层次化稀疏格式作为唯一输入语义源，
- 自动推导出数据结构、循环结构、访存方式、流水线结构，
- 再生成最终 CUDA kernel 的一个格式驱动 SpMM 编译/生成框架。

对你后续用大模型做格式设计、张量放置、流水线优化来说，这个抽象层次非常合适，因为它把问题天然拆成了：

- 格式设计问题
- lowering 问题
- pipeline/search 问题
- backend codegen 问题
