# Codex Development Log

## 2026-07-22 — Agent hierarchical-format cost model and executable portfolio

- Diagnosed why the v6 Agent's mostly TCF+MCO formats underperform the repaired
  SR-BCRS route: current MCO lowering restores a full physical dense tile and
  executes full TF32 MMA/B gathers, so compact storage alone is not a kernel-time
  advantage; logical M=8 also incurs timed physical M=16 padding.
- Added an auditable MCO/dense and hardware-aligned dense tile portfolio without
  modifying saved Agent DSLs. Each sibling has an independent IR/include/build
  tree plus explicit `agent_proposal` or deterministic-sibling provenance.
- Repaired the custom runner's missing runtime execution ABI: automatic segment
  sizing and `auto`/`strict_lb` task mapping can now reach the generated dense
  backend and are encoded in result identity.
- ASIC N=512 FP32/TF32 GPU evidence (warmup=3/repeat=32, serial): the generated
  `dense_16x8_sibling` is correct and improves from 28.754 ms with fixed
  segmentation to 4.89523 ms with automatic load-balanced scheduling; no claim
  is made for the remaining 25 datasets.
- Full diagnosis, contracts, commands, validation, and limitations are in
  `docs/codex/agent_hierarchical_format_portfolio_20260722.md`.

## 2026-07-22 — MASK README rename and project positioning

- Renamed the root README presentation from `LLM-SOG` to the tentative project
  name `MASK: Matrix-Aware Agentic SpMM Kernel Generation`.
- Reframed the project summary around the three intended innovations:
  matrix-aware format design, agent-guided compiler/codegen strategy, and
  hardware-feedback-driven optimization.
- Updated README quick-start examples to current CLI flags and the requested
  `warmup=3`, `repeat=32` benchmark settings.
- Replaced deprecated document pointers with current `docs/codex` and
  `sota_results/summary` provenance references, while keeping the original
  paper as compiler-foundation context.
- This was documentation-only; no code, GPU benchmark, paid model call, or
  generated result artifact was changed. Details are in
  `docs/codex/mask_readme_rename_20260722.md`.

## 2026-07-19 — Historical A100 Sparsene best vs current A800 N=512

- Reduced the 234-row historical Sparsene table to one audited best per dataset in
  `sota_results/summary/sparsene_a100_n512.csv`; selection uses the minimum time among
  rows with an empty anomaly field and records considered/excluded counts.
- Excluded three pre-flagged implausible ACC observations for OVCAR-8H, Yeast, and
  YeastH instead of silently treating sub-microsecond log values as valid winners.
- Compared the 26 correct current A800 generated results with historical A100 bests under
  the user-requested approximate hardware-comparability assumption. Current wins 1/26
  (Yeast, nominal 1.044394x); including flagged anomalies mechanically would give 0/26.
- Geometric-mean historical/current speedup is 0.533922x, so current time is about 1.873x
  historical best. Historical winners are dominated by SR-BCRS: 25/26 variants.
- Added the reproducible comparison script, per-dataset CSV, audit JSON, and focused
  methodology/limitation record in
  `docs/codex/sparsene_a100_best_vs_current_a800_n512.md`. No GPU run was performed.

## 2026-07-19 — v6 custom formats 26/26, N=512 FP32 correctness/performance closure

- Reorganized the paper-original A100/N=512/FP32 data under `sota_results/summary`:
  594 operator observations across 23 source-qualified method labels, all 9 historical
  Sparsene variants, a separate hidden-512 GCN e2e table, paper-reported aggregates,
  coverage audit, and explicit anomaly/provenance handling.
- Extended semantic backend matching from the first three designs to all 26: 25
  column-condensed TCF+MCO formats and one active-fixed-block BIT-BSR+MCO format;
  added BLK_M 32, logical/physical K refinement, and executable row_rcm,
  first-column, and scalable Jaccard row reorder contracts.
- Fixed lost K fragments for BLK_K 16/32, lost M fragments for BLK_M 32, BIT-BSR
  grid-axis inversion, dc2 K=16 restore incompatibility, grid.y overflow, long-tail
  execution segmentation/reduction, physical-padding masks, and inverse output mapping.
- Added full-output TF32-aware verification with strict mismatch/nonfinite plus relative-L2
  and normalized-max-absolute metrics; result JSON retains every metric and refuses
  incorrect/nonfinite output.
- Added archived per-tuning results, per-dataset final tuning configuration, strict serial
  config-driven runner (`jobs=1` by default), explicit one-warp generated launch ABI, and
  audited final table generation that refuses missing/mixed/incorrect results.
- Final A800 N=512 run used warmup=10/repeat=100, strictly serial GPU execution: 26/26
  correctness PASS, 12/26 faster than cuSPARSE, geometric-mean speedup 1.0385x; maximum
  relative-L2 2.41863e-4 and normalized maximum absolute error 2.65992e-4.
- Full root causes, commands, tuning evidence, output paths, performance table, and honest
  remaining bottlenecks are in `docs/codex/custom_format_operator_e2e_v6_all26_n512.md`.

## 2026-07-18 — v6 custom Format DSL operator e2e（前三个数据集）

- 将 v6 运行的 26/26 格式导出到 `python/sparsene/formats/custom`，新增不依赖
  Agent/API 的公共 DSL 验证、构造和加载层；26 份均导入、验证并派生 RTS。
- 新增按层次/axis/atomic/reorder/params 匹配的 TCF-MCO 语义后端，选择不依赖
  数据集名或描述性 `format_name`；未来 format-agent capability 也使用该 matcher。
- 新增按要求命名的自定义 `.inc` 生成器及统一 CMake/CUDA e2e host，所有 IR、
  策略、编译、运行、正确性和性能产物保存在原 e2e run 的同一数据集目录。
- 修复 BIT-TCF 对 BLK_K>8 丢失 K-fragment 的真实 codegen bug；用物理 16 行
  补零执行逻辑 8 行窗口；为 DD 实现确定性 row_rcm 和计时外逆行映射。
- 修复大矩阵 grid.y 越界，并为极端长尾窗口加入通用 partial-C 分段与计时内
  GPU reduction；ASIC 性能由 44.94 ms 改善至 1.484 ms，仍保持零 mismatch。
- A800、N=64、10 warmup/100 repeat 最终结果：ASIC 1.48418/0.783176 ms、DD
  0.303606/0.310159 ms、Ga 3.58360/2.07745 ms（生成/cuSPARSE）；三者均
  mismatch=0。仅 DD 略超过 cuSPARSE，另外两个不得宣称高性能完成。
- 13 项 format-agent unittest、26 格式等价导入/RTS/后端断言、Python compileall、
  三份 NVCC sm80 编译和三份真实矩阵 A800 验证通过；venv 缺 pytest，未安装。
- 完整架构、错误根因、命令、产物与限制见
  `docs/codex/custom_format_operator_e2e_v6_first3.md`。

## 2026-07-18 — Codex skill knowledge consolidation

- Reviewed all 26 `docs` files, five `.agent` runtime skills, 18 `.trae`
  materials, and the existing `sparsene-development` Codex skill; used SHA256
  deduplication for 10 byte-identical `.trae`/`docs` copies.
- Verified durable claims against current format/RTS/IR source, format
  capability profiles, strategy normalization, Agent audit runtime, host cache
  and verification logic, the K sweep, and current CLI help.
- Added focused environment/operations and knowledge-provenance references;
  strengthened project map, format/codegen, Agent-loop, validation, security,
  dataset-path, cache, reorder, ratio-direction, and v5 design/execution rules.
- Explicitly excluded legacy credentials, historical performance numbers,
  fixed failure lists, old adaptive thresholds, and declaration-only support
  claims from current operational knowledge.
- Official skill validation passed, reference/UI metadata checks passed, and
  five primary CLI help commands imported successfully in the required venv.
- Full source inventory, corrections, and unresolved credential cleanup are in
  `docs/codex/sparsene_skill_knowledge_consolidation_20260718.md`.

## 2026-07-18 — Format design agent v5 (general research DSL)

- Removed the immutable ME-TCF/COO prompt and historical 16x8 prior from the
  research format-design phase.
- Added general one-to-six-level df/dv/sf/sv DSL design with COO/MCO/CSR/ELL/DIA/
  dense leaves plus fully specified custom atomic leaves, parameters, optional
  reorder, and explicit transformation plans.
- Added deterministic structural/semantic validation that preserves proposals;
  invalid or unsupported designs are never rewritten to ME-TCF.
- Added backend capability levels (`frontend_research`, `codegen_declared`,
  `standalone_validated`, `integrated_e2e`) and exact known-profile matching.
- Changed model/JSON failure to an explicit failed design rather than a hidden
  safe-format fallback.
- Updated batch/sample reporting to compare complete formats and summarize
  format family, atomic type, capability, and reference tile.
- Guarded the current DTC e2e route so valid unsupported research proposals stop
  clearly before strategy/codegen; paid connectivity probing is now opt-in.
- Updated runtime skills, README, Codex project memory, and focused tests. Full
  rationale and contracts are in `docs/codex/format_design_agent_v5.md`.

## 2026-07-17 — DeepSeek V4 Pro 26-dataset format-design runner

- Required and documented activation of
  `/mnt/lustre-client/huoda/venv/llm-sog`; system Python failed before model
  invocation because PyYAML was unavailable.
- Extended `test_run_llm.py` with model filtering and persistent request,
  response, usage, and summary logs. DeepSeek V4 Pro passed the audited test.
- Extended the common Agent runtime to persist exact system/user messages,
  response text, SHA256 content audit, API token usage, and failure/parse-failure
  audit records for every invocation.
- Strengthened the format prompt with feature semantics, five grounded evidence
  records, separate 16x8/16x16 analysis, Chinese format semantics, selection
  reason, and risks.
- Added `python/test/run_format_design_batch.py` with one unified run directory,
  per-dataset provenance, quality gates, resume support, final format artifacts,
  CSV/JSON summaries, events, and all-attempt token aggregation.
- Added `scripts/run_format_design_26_background.sh` for user-controlled nohup
  execution without overriding the model's configured output-token limit.
- Preserved the partially completed paid run: ASIC_680k is accepted and reusable;
  DD is a logged parse/fallback result that resume will replace.
- Added two runtime-audit tests; the combined format/runtime focused suite has
  eight passing tests. Full operating instructions are in
  `docs/codex/deepseek_format_design_26_run.md`.

## 2026-07-15 — Format design agent v4

- Removed contradictory prompt guidance that invited unsupported levels,
  atomic types, and reordering while claiming to require canonical ME-TCF.
- Added executable profile `me_tcf_coo_sm80_v1`, exact v3 feature digest,
  grounded feature evidence, and normalized `16x8`/`16x16` selection.
- Added deterministic validation of the complete DSL tree, atomic leaf,
  symbolic params, feature evidence, candidate availability, and confidence.
- Fixed `format_name` to the backend ABI value `ME_TCF`, preventing descriptive
  model names from bypassing ME-TCF-specific semantic fusion.
- Changed agent mode to apply the normalized candidate instead of silently
  overriding/ignoring the format-agent decision.
- Removed the source-level API credential fallback; default authentication now
  comes from `DEEPSEEK_API_KEY` or the resolved model configuration.
- Calibrated the safety prior with the existing 26-matrix sweep: BLK_K=8 won
  22/26 comparisons and median `time16/time8` was 1.3137836463785613.
- Synchronized runtime format skills and added six focused unit tests. Design
  and limitations are recorded in `docs/codex/format_design_agent_v4.md`.
- Structurally validated the canonical response through lowered OPIR and all
  seven expected COO global-memory interfaces.

## 2026-07-15 — Matrix feature extraction v3

- Replaced regular-grid block statistics with ME-TCF row-window condensation
  simulation for nine BLK_M/BLK_K candidates.
- Added schema/data-quality, long-tail distribution, locality, reorder-proxy,
  padding, load-balance, and storage-estimate features.
- Added a persistent 26-dataset batch workflow targeting
  `/mnt/lustre-client/huoda/selected_mtx`.
- Updated the format-agent prompt to prefer `hierarchical_candidates` while
  retaining compatibility with historical feature JSON.
- Fixed the shared MatrixMarket reader's missing `integer` branch, discovered
  because all 2,100,225 entries of `mc2depi` were initially rejected despite a
  successful return code.
- Compiled with NVCC for `sm_80` and ran CUDA synthetic tests on an NVIDIA A800
  80GB PCIe, covering rectangular shape, empty rows, duplicate coordinates,
  and symmetry expansion.
- Generated all 26 persistent feature files at
  `/mnt/lustre-client/huoda/selected_mtx`: 26/26 manifest rows are `ok`, all
  schema and SHA256 checks pass, and all nine hierarchy candidates are present.
- Cross-checked candidate `16x8` against existing ME-TCF `tbk8.log` host
  preprocessing output: 26/26 exact matches for total TC blocks, expanded NNZ,
  and padded NNZ.

## 2026-07-18 — Format design agent v6 legality and source-derived format catalog

- Diagnosed the incomplete v5 flexible-format run: 9/26 attempts completed;
  six were truncated at the 8192-token thinking+answer budget and three complete
  JSON responses hit prompt/validator typing or expression-contract conflicts.
- Added `python/sparsene/formats/design_catalog.py` and a formats README covering
  every current declaration, unique hierarchy, atomic payload meaning, design
  lever, capability caveat, HME instability, and the fact that the current
  row-reorder-named source does not set `Axis.reorder`.
- Bumped the research contract to `sparsene.format_design.v6`, injected the
  source catalog and exact feature path/value map, replaced the misleading
  quoted schema placeholder with a type-correct JSON example, and specified
  integer levels/params plus the exact length grammar and varlen semantics.
- Enabled official DeepSeek JSON Output for format design, raised the default
  format-design budget to 32768, rejected `finish_reason=length`, and fixed
  reasoning-token extraction from OpenAI completion detail fields.
- Preserved research freedom and capability separation; numeric strings are
  canonicalized only as non-structural JSON scalar cleanup, while invalid
  hierarchy/axis/atomic/expression proposals remain rejected and auditable.
- Updated the runtime format skill, batch manifest/audit fields, wrapper,
  current Codex skill memory, README, and schema-isolation guidance.
- No paid model or GPU call was made. Python compilation, wrapper syntax, nine
  catalog DSL validations, and 17 focused unit tests passed. Full evidence and
  rerun guidance are in `docs/codex/format_design_agent_v6.md`.

## 2026-07-18 — Kernel-only post-conversion performance contract

- Confirmed from CUDA source that ME-TCF format/metadata conversion, generated
  ABI conversion, allocation/copies, and cuSPARSE preprocess are outside the
  event timers; `mykernel_time` and `cusparse_time` time only warmed repeated
  compute launches.
- Added fixed contract `sparsene.kernel_only_post_conversion.v1` to the format
  Agent system/user prompts, normalized responses, batch manifests, format
  artifacts, and per-dataset Chinese design records.
- Centralized the contract in `python/sparsene/performance_contract.py`; current
  strategy-Agent requests/prompts, its runtime skill, and feedback-loop
  performance results now carry the same contract.
- Required design ranking to ignore format-conversion/reorder latency while
  retaining permutation, inverse-map, metadata, storage, cache, correctness,
  and implementation requirements as separate engineering information.
- Synchronized runtime Agent skill, source format catalog/README, Codex skill,
  agent-loop/codegen/validation references, and v6 design documentation.
- No paid Agent or GPU call was made; Python compilation、18项 focused tests 和
  本地 strategy prompt contract 检查均通过。
## 2026-07-22 — v7 empirical-feedback hierarchical format redesign

- Added `sparsene.format_design.empirical_feedback.v1` to the format-agent
  request contract.  For each dataset it carries compact, auditable prior
  N=512/FP32 kernel measurements for the previous Agent proposal and the
  `sr_bcrs_16x8` dense sibling, including timing scope, correctness, physical
  tile, and execution-policy metadata.
- The prompt now treats this data as calibration rather than a template: it
  explicitly forbids deriving runtime from storage estimates, requires a
  concrete changed mechanism when a prior design lost to dense, and requires a
  falsifiable later measurement.  The research DSL remains unconstrained by
  current backend profiles.
- Extended `run_format_design_batch.py` with `--feedback-run-dir`; it verifies
  complete source coverage, hashes both source summaries, persists per-dataset
  `empirical_feedback.json`, and records the source provenance in the run
  manifest and input provenance.
- Started the new serialized 26-dataset design run at
  `logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback`.
  It uses the v6 Agent/MCO and SR-BCRS measurements only as directional
  feedback; later performance validation remains warmup=3, repeat=32.
- Validated the new prompt/loader with `py_compile`, 13 focused unit tests, a
  two-dataset source-feedback contract check, CLI help, and `git diff --check`.
  The remote run is intentionally single-request serial; its long model calls
  are tracked through per-dataset audit artifacts.
