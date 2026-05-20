# Sparsene 项目技能索引

---

## Skill 文件清单

| Skill 文件 | 内容 | 适用角色 |
|-----------|------|---------|
| [sparsene-framework.skill.md](sparsene-framework.skill.md) | sparsene 框架核心：格式系统、RTS 变换、Computent、OPIR→NVIR→.inc 编译链 | 框架开发者 |
| [sparse-format-design.skill.md](sparse-format-design.skill.md) | 稀疏格式定义 DSL、已实现格式、格式设计决策树、新增格式步骤 | 格式设计者 / 第一个智能体 |
| [strategy-optimization.skill.md](strategy-optimization.skill.md) | ME_TCF 的 CUDA 优化策略空间：全部 knob、约束规则、有效组合数、测试流程 | 第二个智能体 / 调优开发者 |
| [strategy-agent-feedback.skill.md](strategy-agent-feedback.skill.md) | 策略智能体架构、prompt 设计、反馈循环、脚本入口、模型接入 | 第二个智能体开发者 / 实验执行者 |

---

## 阅读顺序建议

### 新开发者（理解全貌）
1. `sparsene-framework.skill.md` → 理解框架本身
2. `sparse-format-design.skill.md` → 理解格式系统
3. `strategy-optimization.skill.md` → 理解优化策略空间
4. `strategy-agent-feedback.skill.md` → 理解智能体和反馈回路

### 格式智能体开发者
1. `sparse-format-design.skill.md`
2. `sparsene-framework.skill.md`（第 2-4 节）

### 策略智能体开发者
1. `strategy-optimization.skill.md`
2. `strategy-agent-feedback.skill.md`
3. `sparsene-framework.skill.md`（第 6-7 节）

### 实验执行者
1. `strategy-agent-feedback.skill.md`（第 4-6 节）

---

## 关键文件速查

| 文件 | 所属 Skill |
|------|-----------|
| `python/sparsene/format/format.py` | sparsene-framework |
| `python/sparsene/formats/DTC_SpMM.py` | sparse-format-design |
| `python/sparsene/transform/rts.py` | sparsene-framework |
| `python/sparsene/op_gen/computent/computent.py` | sparsene-framework |
| `python/sparsene/op_gen/opir/generate.py` | sparsene-framework |
| `python/sparsene/op_gen/nvir/generate.py` | sparsene-framework + strategy-optimization |
| `python/sparsene/op_gen/nvir/codegen.py` | sparsene-framework |
| `python/sparsene/op_gen/nvir/compiler_driver.py` | sparsene-framework |
| `python/sparsene/op_gen/strategy_agent.py` | strategy-agent-feedback + strategy-optimization |
| `python/test/run_strategy_feedback_multidataset.py` | strategy-agent-feedback |
| `examples/.../scripts/run_agent_feedback_multidataset.sh` | strategy-agent-feedback |
| `examples/.../scripts/run_deepseek_feedback_shipsec1.sh` | strategy-agent-feedback |
| `examples/.../scripts/validate_generated_kernel.sh` | sparsene-framework |
| `examples/.../main.cu` | sparsene-framework |

---

## 更新日志

- v1.0 (2026-05): 初始版本，基于多轮 deepseek/mimo 实验的实际经验
