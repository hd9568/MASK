# Docs Index

- `project_code_explanation.md`
  - 以代码为主解释 Sparsene 的整体工作流、ME-TCF 格式、IR、lowering、以及各类 `inc` 文件的职责。
- `generated_kernel_validation.md`
  - 记录 `python/test/test_generate.py` 生成内核的验证过程、当前问题、以及下一步修复建议。
- `codegen_backport_gpu_validation.md`
  - 记录本轮将修复回灌到 Python codegen 后，对 generated 路线做的真实 GPU 验证结论，包含运行命令、日志位置和当前 remaining bug。
- `llm_agent_design.md`
  - 给出基于本项目做矩阵特征分析、格式设计、张量放置、流水线与负载均衡 Agent 的研发方案。
