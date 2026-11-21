# 自适应搜索与精度门槛

## 正确性与数值模式

生成内核逐元素满足 `abs(got-ref) <= 1e-3 + 1e-3*abs(ref)`；出现非有限值或任意元素超限即判错。参考值为同一输入的 cuSPARSE FP32 输出，重复测量均需通过。相对 L2 误差用于诊断，不得替代逐元素门槛。旧精度记录不能作为满足新门槛的证据。

默认 TF32 生成内核使用三项乘加：将两输入分别拆成 TF32 高位和残差，计算高×高、低×高、高×低，低×低未计算。这会改变开销与数值，不能将其耗时与单次 TF32 的旧结果视为同一计算精度。可在 CUDA 编译选项显式定义 `SPARSENE_FAST_TF32` 以恢复原始单次 TF32；无论哪种模式，正确性阈值相同，快速模式不保证通过。比较性能时必须使用相同输入、精度门槛和计算模式。

## 停止策略

内层以通过完整正确性检查的策略时间为观测；外层以每个完整格式下最佳正确策略时间为观测。候选时间采用多次独立运行的中位数。顺序扫描时维护最近一次显著更新的最佳时间 `best`。当前相对收益为 `(best - time) / best`，更新阈值为 `max(0.015, 当前测量相对极差, best 对应测量相对极差)`；相对极差等于 `(max-min)/median`。有至少三个有效观测、且连续两个有效观测未超过该阈值时，停止对应层。首个有效观测建立 best。没有显著更新时，小幅改进会与保留的 best 累积比较，最终排名仍按实际最佳时间选取。

编译、运行、模型或正确性失败不会被当作无收益观测，并会中断连续无收益计数。阈值是抗测量波动的启发式，并非统计显著性检验，也不保证已找到全局最优。迭代上限依旧是资源安全边界，而非收敛判断；现有 `--format-iterations` 和 `--strategy-agent-iterations` 设置上限，`--disable-early-stop` 可用于完整预算对照。

每轮判断在对应层的 `early_stop/before_XX.json` 留存候选、收益、阈值和原因，`early_stop_decision.json` 是最新判断。正常结果目录另存格式设计、模型请求与回复、生成 IR 和 CUDA、编译日志、各次结果及 NCU；环境变量凭据不得写入。停机机制可用关闭早停的等预算实验对照评价，包括正确率、最佳时间、模型 token 和运行时间。

## 设计依据

KernelAgent 把硬件反馈用于迭代，并讨论局部最优下的搜索；Atrex 将预算耗尽、长期停滞或阻塞列为结束条件；自动算法配置研究中的 capping 依据观测提前终止低价值执行。capping 针对单次执行，本项目的规则针对多轮搜索，二者对象不同。本实现借鉴节省预算的思路设置两层停滞规则，具体阈值是当前工程选择，尚需开展消融与更大样本验证。

- [KernelAgent: Hardware-Guided GPU Kernel Optimization](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration/)
- [Atrex Kernel Agent design](https://github.com/alibaba/atrex-kernel-agent/blob/main/docs/design.md)
- [How to Speed-Up the Automated Configuration of Optimization Algorithms](https://cdn.aaai.org/ojs/18588/18588-77-22107-1-2-20210721.pdf)
