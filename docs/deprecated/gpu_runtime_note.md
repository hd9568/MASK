# GPU 运行环境说明

如果你在登录 shell 里能看到：

```bash
nvidia-smi
```

但在 IDE/Agent 触发的命令里看不到 GPU，这通常不是节点本身有问题，而是：

- IDE 命令执行在沙箱环境里
- 沙箱里没有映射 `/dev/nvidia*`
- 或沙箱不能和宿主机驱动通信

这会导致两类现象：

- `nvidia-smi` 在沙箱里失败
- CUDA 程序在沙箱里无法真正启动 kernel

## 当前项目怎么跑

当前生成内核的 patched 验证入口是：

- `examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh`

推荐直接在你自己的 GPU shell 中运行：

```bash
cd /home/huoda/sparsene
CUDA_ARCH=sm_80 \
examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

如果脚本仍然因为环境探测问题跳过运行，可以强制执行：

```bash
cd /home/huoda/sparsene
FORCE_RUN=1 CUDA_ARCH=sm_80 \
examples/src_fp32/dtc/testbed/scripts/validate_generated_kernel.sh \
  -mtx_flag 0 -M 128 -K 128 -N 64 -sparsity 0.9 -warmup 1 -repeat 1
```

## 当前结论

- 编译链已经打通
- 在真实 GPU shell 里可以继续做最终 correctness 对比
- IDE/Agent 沙箱里看到“GPU 不可用”不等于节点本身 GPU 不可用
