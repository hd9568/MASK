# `test_generate.py` 生成算子的验证记录

本文记录本次对 `python/test/test_generate.py` 生成内核的实际验证过程、结果、目前发现的问题，以及后续应如何继续推进。

## 1. 验证目标

目标不是验证 testbed 里的老 `kernel.inc`，因为它已经有：

- `examples/src_fp32/dtc/testbed/main.cu`
- `examples/src_fp32/dtc/testbed/host_program.cu`

这一套和 cuSPARSE 的对比链路。

真正要验证的是：

- `python/test/test_generate.py` 生成出的 `python/me_tcf_kernel3.inc`

也就是：

- 新生成链输出的 ME-TCF 内核是否正确
- 它能不能接上现有 ME-TCF 元数据
- 它能不能编译
- 它能不能在 GPU 上跑出和 cuSPARSE 一致的结果

## 2. 本次实际做了什么

### 2.0 当前验证模式

当前为了专注修复 `patched inc`，验证脚本已经默认切到：

- `PATCHED_ONLY=1`

也就是：

- 默认不再先执行 `python/test/test_generate.py`
- 直接使用 `python/me_tcf_kernel3_patched.inc`
- 直接编译并运行 validator

对应脚本：

- `examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh`

如果后续要重新打开自动生成步骤，可以显式传：

```bash
PATCHED_ONLY=0 examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh ...
```

### 2.1 跑通生成链

执行环境：

- 虚拟环境：`/mnt/lustre-client/huoda/venv/llm-sog`
- 工作目录：`/home/huoda/sparsene/python`

执行命令：

```bash
source /mnt/lustre-client/huoda/venv/llm-sog/bin/activate
cd /home/huoda/sparsene/python
PYTHONPATH=/home/huoda/sparsene/python python test/test_generate.py
```

结果：

- 生成成功
- 输出文件：`python/me_tcf_kernel3.inc`

日志中能看到：

- 主循环被识别成 `for_loop`
- 动态切成了 3 个 pipeline stage
- 最终成功 dump 到 `me_tcf_kernel3.inc`

### 2.2 分析新旧元数据接口

通过额外跑 Python 侧分析脚本，拿到了 lowering 之后的外部符号定义。

旧 testbed 的元数据：

- `val_sidx`
- `val_soff`
- `val_coo_idx`
- `val_coo_val`
- `val_coo_off`

生成内核 lowering 之后的元数据：

- `val_sidx_new`
- `val_len_offset`
- `val_coo_idx_new_1`
- `val_coo_val_new_1`
- `val_coo_len_new_offset`

映射关系：

- `val_soff` -> `val_len_offset`
- `val_sidx` -> `val_sidx_new`
- `val_coo_idx` -> `val_coo_idx_new_1`
- `val_coo_val` -> `val_coo_val_new_1`
- `val_coo_off` -> `val_coo_len_new_offset`

其中真正需要注意的地方：

- `val_sidx_new` 访问布局与旧 `val_sidx` 不同，需要 host 侧转成 `[BLK_K, TCBlock_num]` 的存放方式。
- `val_coo_len_new_offset` 需要从旧的二元 offset 数组中提取 padded prefix start。

### 2.3 搭建新验证 harness

为了让 `main.cu` 这套“CSR -> ME-TCF -> cuSPARSE 对比”的链路继续复用，我新增了两个文件：

- `examples/src_fp32/dtc/testbed/host_program_generated.cu`
- `examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh`

思路是：

- 保持 `main.cu` 不变
- 让 `host_program_generated.cu` 暴露与旧 `host_program.cu` 相同的函数签名
- 在 host 侧把旧元数据转换成 `generated_kernel(...)` 需要的新接口
- 再直接启动 `python/me_tcf_kernel3.inc` 里的 `generated_kernel`

这样做的好处是：

- testbed 的输入构造逻辑完全复用
- 验证重点集中在“生成内核是否正确”
- 不需要重写一套新的稀疏格式构造流程

## 3. 实际验证结果

## 3.1 生成成功

结论：

- `test_generate.py` 可以稳定生成 `python/me_tcf_kernel3.inc`

这说明：

- `Format -> RTS -> Computent -> OpIR -> NVIR -> codegen`
  这条链路在 Python 侧是可执行的。

## 3.2 编译失败

使用新脚本编译生成内核专用验证器时，出现了确定性的编译错误。

主要错误分三类：

### 问题 A：生成器没有做 C++ 标识符消毒

`python/me_tcf_kernel3.inc` 里直接生成了这些类名：

- `class div`
- `class const`
- `class add`
- `class mul`

这些名字会和 C++ 关键字或标准库符号冲突。

其中最严重的是：

- `const` 是 C++ 关键字，根本不能作为类名

所以这不是环境噪声，而是：

- 生成器当前输出的 C++ 源码本身就不合法

这是当前最直接、最明确的 blocker。

### 问题 B：代码里直接引用了未定义符号

生成内核内部还直接使用了：

- `BLK_K`
- `BLK_M`
- `BLK_N`

但文件自身没有保证这些宏或常量一定可见。

实际编译时报错：

- `identifier "BLK_K" is undefined`
- `identifier "BLK_M" is undefined`
- `identifier "BLK_N" is undefined`

这说明当前 codegen 把“符号表达式”和“最终可编译常量”混在了一起。

换句话说，生成器在把符号名写进代码时，没有统一替换成：

- 模板推导表达式
- 或本地 `constexpr`
- 或明确宏定义

### 问题 C：因此后续模板实例化也全部级联崩溃

因为前面几个基础类都没生成对，后面会继续出现：

- `div` ambiguous
- `const_t` / `add_t` / `mul_t` 推导失败
- `Layout_o0` 无法实例化
- `cosize_v<...>` 失败

这些错误本身不是根因，只是前两个问题导致的级联错误。

## 3.3 运行环境也不满足 GPU 执行

除了编译问题，本机当前环境还存在运行问题：

- `nvidia-smi` 无法访问驱动
- `/proc/driver/nvidia/version` 存在
- 但当前会话里没有 `/dev/nvidia*`

这意味着：

- 即便生成内核能编译，当前会话也无法真正启动 CUDA kernel 做数值对比

所以本次验证被两个问题同时卡住：

1. 生成代码当前不能编译
2. 当前环境没有可用 GPU 设备节点

## 4. 当前结论

结论非常明确：

- 现在的 `python/me_tcf_kernel3.inc` 还不能被认为是“正确算子”

原因不是“数值比对略有误差”，而是更早的阶段就失败了：

- 它目前连 C++/CUDA 编译正确性都不满足

因此从验证阶段划分来看：

- Python 生成阶段：通过
- 生成代码编译阶段：失败
- GPU 运行阶段：未达成
- 与 cuSPARSE 数值对比阶段：未达成

## 5. 这说明问题大概率出在哪

本次失败更像“后端 codegen bug”，而不是“ME-TCF 语义错了”。

优先级最高的可疑点：

### 5.1 `NvIrCodeGenerator` / printer 没有做符号合法化

需要检查的方向：

- 算术 op 的名字是否直接被拿来当类名/类型名
- 是否缺少统一的 `sanitize_identifier()`
- 是否只对 program/op 名字做了 sanitize，而没有对派生类名做 sanitize

从当前症状看，这一条几乎是确定问题。

### 5.2 符号常量没有在最终代码中 materialize

`BLK_M/BLK_N/BLK_K` 的来源在前端是合法的符号。

但到最终 C++ 时应该做成下面三种之一：

- `static constexpr int BLK_K = get<2>(BLK_MNK{});`
- `Int<...>` 风格的模板表达式直接内联
- 或者宏定义

现在生成文件里直接裸写 `BLK_K`，说明常量 materialization 还没闭合。

### 5.3 即便编译修好，后面还要继续检查 shared memory 对齐

`test_inc/README.md` 已经提醒过一个潜在问题：

- `array_ref` 的 `int` shared buffer 和 `array_ref_1` 的 `float` shared buffer 紧邻
- 这可能导致后续 `ldmatrix` 读取 `tileB` 时 shared 地址不满足对齐要求

本次新生成文件里依然能看到类似模式：

- `__shared__ int array_ref_tensor_o0[...]`
- `__shared__ float array_ref_1_tensor_o0[...]`

所以即使把编译问题修掉，下一阶段非常可能还会遇到：

- `ldmatrix` shared memory misaligned
- 或数值错误 / memcheck 报错

## 6. 我已经为后续验证留下了什么

### 6.1 新的 host 侧桥接文件

- `examples/src_fp32/dtc/testbed/host_program_generated.cu`

它已经把“老 testbed 元数据 -> 新 generated kernel 元数据”的桥接逻辑写出来了。

这个文件后续仍然有价值：

- codegen 修好后，可以直接继续用它做端到端验证

### 6.2 一键脚本

- `examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh`

它已经串起了：

1. 生成 `me_tcf_kernel3.inc`
2. 编译 generated-kernel validator
3. 若有 GPU 则直接运行

当前它默认显式使用：

- `gcc-toolset-13` 作为 host compiler
- `sm_80` 作为默认 CUDA 架构

这样至少可以保证：

- Tensor Core / `ldmatrix` 指令不会因为默认目标架构过低而在 ptxas 阶段失败

## 7. 手工修补版 `inc` 的结果

为了在不改 Python codegen 的前提下继续推进，我额外复制并修补了：

- `python/me_tcf_kernel3_patched.inc`

并让：

- `examples/src_fp32/dtc/testbed/host_program_generated.cu`

改为 include 这个 patched 版本。

### 7.1 手工修补了什么

这次手工修补主要分 6 类：

1. 把非法或冲突标识符重命名：
   - `div` -> `gen_div`
   - `const` -> `gen_const`
   - `add` -> `gen_add`
   - `mul` -> `gen_mul`
2. 修正 `const_t` 和 `const_1_t` 少传 `NBUF` 模板参数的问题
3. 把裸写的 `BLK_M / BLK_N / BLK_K` 改成 `get<0/1/2>(BLK_MNK{})`
4. 把错误输出成 `CUTE_STATIC_ASSERT_V` 的动态 shape 检查降成普通 `assert`
5. 修掉 `mma`、`c_val_store` 里明显不合法的 shape / copy 代码
6. 为 `coo_atomic_format_load_{idx,val}` 的 staging buffer 临时给出静态上界，绕过 `VARLEN` 布局无法做 `cosize_v` 的问题

### 7.2 手工修补后的编译状态

结论：

- patched 版已经可以成功编译出二进制

验证命令：

```bash
CUDA_ARCH=sm_80 \
examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

当前结果是：

- 第 1 步生成原始 `me_tcf_kernel3.inc` 成功
- 第 2 步基于 patched 版编译 `demo-dtc-generated-tf32` 成功
- 第 3 步因为当前环境没有 `/dev/nvidia*`，运行被脚本主动跳过

所以 patched 版把状态推进到了：

- 生成：通过
- 编译：通过
- GPU 数值正确性：等待有 GPU 的节点

### 7.3 还不能说“正确性通过”的原因

必须非常诚实地说：

- 现在还不能声称 GPU 数值正确性已经通过

因为当前环境里：

- 没有可用的 `/dev/nvidia*`
- 无法真正启动 kernel
- 也就无法和 cuSPARSE 做最终数值对比

因此目前最准确的说法是：

- patched 版已经完成“编译正确性闭环”
- 但“运行正确性闭环”还差最后一步 GPU 执行

### 7.4 新的 GPU 运行期问题与修复

在真实 GPU 节点上继续运行后，新的首个运行期错误变成了：

- `misaligned address (716)`

从日志可以看出：

- 这已经不是前面的 `array_ref` shape 断言
- 而是 kernel 继续往后执行后，在 Tensor Core `ldmatrix` 相关路径上触发的 shared-memory 对齐问题

这个判断同时被仓库已有说明佐证：

- `test_inc/README.md` 已明确指出 `array_ref` / `array_ref_1` 相邻的 `__shared__` 声明在部分配置下会导致 B tile 的 `ldmatrix` 访问 misaligned

因此 patched 版又补了一个最小修复：

- 给 `array_ref_tensor_o0`
- `array_ref_1_tensor_o0`
- `coo_atomic_format_load_val_tensor_o0`
- `coo_atomic_val_restore_tensor_o0`

都显式加上了 `__align__(16)` 的 shared-memory 对齐约束。

这一步的目的不是“最终优化”，而是：

- 先把当前 GPU 运行时最早的 misaligned fault 推过去
- 继续暴露后续更真实的数值或语义问题

### 7.5 更深一层的根因

进一步对照原 testbed 里已知正确的：

- `G2sMatrixBLoadOp`
- `S2sRestoreMatrixAOp`

可以发现，`patched inc` 原先不仅有对齐问题，还存在更根本的 shared layout 偏差：

- `array_ref_1` 的 B tile 不是原始 kernel 使用的 Tensor Core 友好 swizzled layout
- `coo_atomic_val_restore` 的 A tile 也不是原始 kernel 的 swizzled layout

这会带来两个后果：

1. `ldmatrix` 更容易直接触发 `misaligned address`
2. 即使不触发异常，shared 中的元素排布也可能和 Tensor Core 读取预期不一致

因此 patched 版又补了第二层修复：

- `array_ref_1` 改为与原 `G2sMatrixBLoadOp` 一致的
  `composition(Swizzle<3, 2, 3>{}, ...)`
- `coo_atomic_val_restore` 改为与原 `S2sRestoreMatrixAOp` 一致的
  `composition(Swizzle<1, 2, 3>{}, ...)`

这一轮修复比单纯加 `__align__(16)` 更接近根因，因为它修的是：

- Tensor Core 读取时真正依赖的 shared-memory 逻辑布局

### 7.6 再往后暴露出的 `mma` fragment 语义问题

在修正 shared layout 之后，新的首个运行期错误变成了：

- `mma::f()` 中对第三个输入 `operand_2` 的 shape 断言失败

日志里可以直接看出：

- `operand_0` 是 A 的寄存器 fragment
- `operand_1` 是 B 的寄存器 fragment
- `operand_2` 并不是普通的 `[BLK_M][N]` 平面 tile
- 而是和原 testbed `CalculateOp` 一样的寄存器累加器 fragment：
  `Int<4> x (tile_mma_m, tile_mma_n)`

这说明 patched 版里 `mma` 还有一层更本质的问题：

- 它被错误地写成了“普通二维 tile 的 fallback GEMM”
- 但上下游实际传递的是 Tensor Core fragment 语义

因此 patched 版又做了一次关键修复：

1. 把 `mma` 的 `Shape_i2` / `Shape_o0` 改回 fragment 形状
2. 去掉错误的 `N, N` 构造参数
3. 直接按原 testbed `CalculateOp` 的方式：
   - 从 `REGA` / `REGB` fragment 做 `cvt.rna.tf32.f32`
   - 调用 `mma_m16n8k8_fp32_tf32_tf32_fp32`
   - 在寄存器累加器 fragment 上原位更新

这一轮修改的意义是：

- 不再把 Tensor Core fragment 错当普通矩阵 tile
- 让 `mma` op 本身重新与上下游 op 的数据表示保持一致


## 8. 这次手工修补反推出的 codegen 问题

基于 patched 版中必须做的修改，可以把 Python 侧问题进一步缩成下面几类：

### 8.1 标识符合法化缺失

生成器必须统一处理：

- 关键字
- 与 `cute` / 标准库冲突的名字
- 派生辅助类名

### 8.2 模板参数传播不完整

`const_t` / `const_1_t` 暴露了一个问题：

- 某些 op 定义带 `NBUF`
- 但实例化时漏传了这个模板参数

这说明：

- op class printer
- op instantiation printer

两侧没有共享同一份模板签名来源。

### 8.3 符号 materialization 不闭合

`BLK_M/N/K` 的问题说明：

- 前端 symbol 到最终 C++ 常量的落地规则没有统一

### 8.4 动态 shape 被错误降成静态断言

`CUTE_STATIC_ASSERT_V` 出现在动态 `VARLEN` 相关路径上，说明 codegen 当前没有区分：

- 编译期 shape 约束
- 运行期 shape 约束

### 8.5 动态布局对象被错误用于静态 `cosize_v`

这类问题说明 printer 现在默认假设：

- 只要是 `Layout_o0` 就一定可以静态分配缓冲

但对变长 layout 这不成立。

### 8.6 某些 op 的 shape / copy 逻辑存在 printer 级拼接错误

例如：

- `mma` 输出 shape 维度数不一致
- `c_val_store` 试图把标量 copy 到 tile

这更像：

- backend printer 对 op 语义模板的映射写错了

## 9. 下一步最合理的顺序

1. 切到有 GPU 的节点，直接运行 patched 版 harness
2. 若数值通过，再把 patched 改动系统性回灌到 Python codegen
3. 若数值不通过，再用 patched 版为基准继续缩小问题到具体 op
4. 最后再做真正的 backend 修复，而不是继续长期维护手工 patched `inc`

## 10. 建议的下一步修复顺序

建议严格按这个顺序来，不要直接跳到数值调试：

1. 修 codegen 的 identifier sanitize
2. 修 `BLK_M/BLK_N/BLK_K` 等符号 materialization
3. 跑 compile smoke
4. 跑小尺寸随机矩阵
5. 和 cuSPARSE 做数值对比
6. 再用 `compute-sanitizer` 查 shared 对齐 / 越界问题

因为在当前状态下：

- 语义调试还没资格开始

## 11. 后续真正跑数值验证时建议怎么做

在修完编译问题且切到有 GPU 的节点后，建议先跑最小随机矩阵：

```bash
examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

再逐步扩大：

- `M=256, K=256, N=64`
- `M=512, K=512, N=64`

然后再切换到真实数据集：

- `/mnt/lustre-client/zhangkaige/selected_mtx`

## 12. 最终判断

截至目前，最准确的结论分两层：

- 原始 `me_tcf_kernel3.inc`：能生成文本，但不能编译
- 手工修补版 `me_tcf_kernel3_patched.inc`：已经能编译，但还没有 GPU 数值实证

所以本次验证的核心结论是：

- 当前自动生成 kernel 还没有达到“可直接验证数值正确性”的状态
- 但手工修补版已经把问题收缩到“只差 GPU 运行验证”

因此你后续如果要基于这个项目继续做大模型驱动优化，最好先把这条 backend codegen 链修到：

- 生成文件可编译
- 小规模数值对比通过

否则 LLM/Agent 上层再聪明，最终都会被 backend 的静态错误卡死。
