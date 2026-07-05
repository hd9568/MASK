# 四个格式算子的生成、编译与运行

本文档简要记录当前四个格式生成式算子的使用方法：

- `DTC-SpMM`
- `ACC-SpMM`
- `BITBSR`
- `SR_BCRS`

假设当前工作目录为仓库根目录：

- `REPO_ROOT=.`  

对应生成文件统一输出到：

- `python/test/results/dtc.inc`
- `python/test/results/acc.inc`
- `python/test/results/bitbsr.inc`
- `python/test/results/sr_bcrs.inc`

## 1. 生成方法

生成入口脚本：

- `python/test/generate_sparse_formats.py`

执行命令：

```bash
cd <REPO_ROOT>
PYTHONPATH=python python python/test/generate_sparse_formats.py
```

脚本内部会依次完成：

```text
Format -> RTS -> Computent -> OPIR -> VarlenLowering -> CValFlatten -> NVIR -> software pipeline + codegen
```

四个格式和输出文件映射如下：

- `ME_TCF -> dtc.inc`
- `BIT_TCF -> acc.inc`
- `BIT_BSR -> bitbsr.inc`
- `SR_BCRS -> sr_bcrs.inc`

## 2. 编译方法

### DTC

```bash
cd examples/src_fp32/dtc/testbed
cmake --build build-generated --target demo-dtc-generated-tf32 -j8
```

### ACC

```bash
cd examples/src_fp32/acc/testbed
cmake --build build-generated --target demo-acc-generated-tf32 -j8
```

### BITBSR

```bash
cd examples/src_fp32/bitbsr/testbed
cmake --build build-generated --target demo-bitbsr-generated-tf32 -j8
```

### SR_BCRS

```bash
cd examples/src_fp32/sr_bcrs/testbed
cmake --build build-generated --target demo-srbcrs-generated-tf32 -j8
```

## 3. shipsec1 运行方法

测试数据根目录：

- `/mnt/lustre-client/zhangkaige/selected_mtx`

例如：

- `/mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx`

统一运行参数：

- `-N 1024`
- `-warmup 5`
- `-repeat 20`

### DTC

```bash
cd examples/src_fp32/dtc/testbed
./build-generated/demo-dtc-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -N 1024 -warmup 5 -repeat 20
```

### ACC

```bash
cd examples/src_fp32/acc/testbed
./build-generated/demo-acc-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -N 1024 -warmup 5 -repeat 20
```

### BITBSR

```bash
cd examples/src_fp32/bitbsr/testbed
./build-generated/demo-bitbsr-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -N 1024 -warmup 5 -repeat 20
```

### SR_BCRS

```bash
cd examples/src_fp32/sr_bcrs/testbed
./build-generated/demo-srbcrs-generated-tf32 \
  -filename /mnt/lustre-client/zhangkaige/selected_mtx/shipsec1.mtx \
  -N 1024 -warmup 5 -repeat 20
```

## 4. 当前验证结果

在当前代码状态下，四个格式均已完成：

- 生成成功
- 编译成功
- `shipsec1` 正确性验证通过

当前一组参考性能结果如下：

- `DTC`: `mykernel_time: 3.355494 ms`
- `ACC`: `mykernel_time: 14.676839 ms`
- `BITBSR`: `mykernel_time: 23.759155 ms`
- `SR_BCRS`: `mykernel_time: 8.190105 ms`

正确性检查均满足：

```text
check nan: cusp 332.0 mykernel 332.0
```

## 5. 建议使用顺序

每次修改 codegen 之后，建议按以下顺序回归：

```bash
cd <REPO_ROOT>
PYTHONPATH=python python python/test/generate_sparse_formats.py
```

然后依次编译并运行四个 target：

1. `DTC`
2. `ACC`
3. `BITBSR`
4. `SR_BCRS`

这样可以最快判断：

- 生成过程是否正常
- 某个格式是否出现编译回退
- 某个共享 lowering 是否影响多个格式
