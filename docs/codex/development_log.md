# Codex Development Log

## 2026-08-15 — five-round strategy Agent campaign analysis

- The resumed remaining-dataset campaign completed 25 dataset slots (ASIC_680k
  excluded): 16 completed, 2 completed with isolated failures, 5 failed due the
  Agent/GPU startup path, and 2 unsupported by the current backend.
- Eighteen datasets had a correct iteration-0 baseline and at least one correct
  Agent candidate. Relative to the same-campaign baseline, 12 improved and 6
  selected the baseline. Mean per-dataset kernel-time reduction was 5.27%, median
  0.10%; six datasets exceeded the 1.5% noise threshold, averaging 15.66%.
- The primary comparison table is retained at
  `logs/e2e_runs/20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback/strategy_agent_vs_iteration0_comparison.csv`;
  interpretation, failure accounting, and strategy-field explanations are in
  `docs/codex/strategy_agent_vs_baseline_analysis_20260815.md`.

## 2026-08-15 — remaining datasets: five-round strategy feedback and artifact cleanup

- Changed the feedback driver default from three to five Agent iterations (plus
  iteration 0 heuristic baseline). The generic wrapper now also requests five;
  the ASIC wrapper remains an explicit 12-iteration historical campaign.
- Added `--exclude-datasets` and `--dataset-workers`. The remaining-dataset
  wrapper excludes `ASIC_680k`, overlaps two dataset pipelines during Agent/code
  generation and compilation, and uses a process-wide GPU lock for correctness,
  each timing sample, and NCU collection. Thus measurements remain uncontended
  while CPU/network waits do not serialize the whole campaign.
- Added `docs/codex/strategy_feedback_remaining5_scheduler_cleanup_20260815.md`
  documenting the scheduling invariants, resume/status semantics, compact patch
  protocol, and evidence-retention policy.
- A first two-worker launch stopped during concurrent lowering before any
  summary was produced. The generated source showed the compiler's process-global
  state is not thread-safe; that diagnostic log is retained as
  `strategy_tuning_remaining_v2_5iter_driver_initial_concurrency_abort.log`.
  Generation/compilation is now guarded by a process lock, while Agent waits and
  GPU sample-level scheduling remain overlapped. The resumed campaign is running
  in session `77081` with its output redirected to the normal driver log.
- Removed only reproducible CMake/NVCC build trees from the v7 run directory:
  45 trees / 685,944,787 bytes. Generated sources, strategy and format artifacts,
  commands, correctness/performance/NCU evidence, summaries, and figures remain;
  the first-pass paths were recorded before the later resume-attempt cleanup;
  `cleanup_manifest.json` now records the latest six paths plus cumulative
  totals (51 trees / 766,587,700 bytes).
- Removed the temporary `strategy_tuning_remaining_v2_5iter_lockcheck` dry-run
  directories and their three run-level summaries after the scheduler check;
  they contained no experiment evidence.
- The real campaign was stopped after the first executable datasets returned
  CUDA error 100 (`no CUDA-capable device is detected`). The attempt status and
  exact resume command are recorded in
  `strategy_tuning_remaining_v2_5iter_attempt_status.json`; no Agent iterations
  were spent after the systemic GPU failure was confirmed.

## 2026-08-15 — ASIC_680k strategy-feedback diagnosis and 12-iteration campaign

- Audited all four retained ASIC_680k iterations. The 96-register candidate was
  only 0.86% slower than baseline (inconclusive under one-shot timing), while
  `multi_bind` and `tile_b=32` were 22.79% and 26.25% slower because they greatly
  increased task/grid work and reduced the reuse/bandwidth advantages of the
  memory-dominant tile-64 baseline.
- Fixed an NCU parser collision where percent and byte/s variants of `Memory
  Throughput` were averaged into an impossible percentage. Historical retained
  CSV is now reparsed into separate utilization and bandwidth fields.
- Expanded Strategy Agent feedback with requested-versus-effective execution,
  actual segments/tasks/task groups, all-iteration scoreboard, backend-aware
  tested/untested field coverage, targeted experiment guidance, and an explicit
  1.5% noise policy. Correctness exit code 2 now preserves its JSON evidence.
- Each compiled strategy is independently evaluated three times at N=512,
  FP32, warmup=3/repeat=32 and ranked by median. Added isolated `--tuning-name`
  output and `scripts/run_asic_strategy_feedback_12.sh` for baseline plus 12
  serial Agent iterations without overwriting the historical four rounds.
- Added the canonical 21-field catalog and generated/checkable maintenance guide
  `docs/codex/strategy_agent_search_space.md`. Its raw group products are
  384 × 180 × 48 × 400 = 1,327,104,000 before normalization/applicability gates.
- GPU timing/NCU/paid Agent calls were not started because this node currently
  cannot communicate with the NVIDIA driver. Diagnosis, evidence and recovery
  command are in `docs/codex/asic_680k_strategy_feedback_analysis_20260815.md`.

### 2026-08-15 follow-up — ASIC_680k 12-iteration result

- The isolated campaign completed 13 total rounds (heuristic baseline plus 12
  Agent slots). 11 candidates passed compilation and all three full-output
  correctness measurements; iterations 2 and 11 were Agent response failures
  caused by `finish_reason=length`/truncated JSON, not kernel failures.
- Iteration 4 (`task_group_tiles=128`) is the stable winner: median generated
  time 4.72234 ms versus the same-campaign baseline median 4.90566 ms, a
  3.74% reduction with only 0.067% three-sample spread and 1.05149x median
  speedup over cuSPARSE. Iteration 7 (register cap 112 plus group 128) is tied
  within 0.024%; later group 256, tile 32, wait-depth, stage-shift and
  multi-bind candidates did not improve it.
- Runtime feedback identified a severe schedule hazard: runtime segment=64
  caused effective row binding and 29.49810 ms (5.0x slower), while multi-bind
  increased task/grid work and reached 5.98467 ms. The best NCU profile remains
  memory-dominant (DRAM 81.08%, compute 27.99%, registers 114), confirming that
  the gain is task/atomic scheduling amortization rather than occupancy growth.
- One normal timing candidate (iteration 9) had an NCU device profiling error
  with return code 9; its correctness and repeated timing remain valid. The
  campaign covered 25/61 searchable discrete values for this dense backend.
- Full evidence and interpretation are appended to
  `docs/codex/asic_680k_strategy_feedback_analysis_20260815.md`; raw artifacts
  are under `strategy_tuning_asic_v2_12iter`.
- Replaced the iterative Agent's full-strategy response contract with a compact
  `strategy_patch` relative to best-correct. The driver merges and normalizes
  patches locally, while preserving full final strategies and provenance; the
  default completion budget is now 16,384. Added regression coverage for the
  merge behavior and an explicit `completed_with_failures` campaign status.
- Added the reproducible two-panel trajectory plot
  `asic_strategy_iteration_performance.{pdf,png}` plus CSV data and plotting
  script. It shows median/min/max timing, per-round cuSPARSE speedup, and gaps
  for the two truncated Agent responses.

## 2026-08-12 — v7 Strategy Agent + NCU closed-loop campaign

- Extended the structured strategy contract from legacy ME-TCF placement/pipeline
  knobs to four groups: tensor movement, software pipeline, compiler resources,
  and runtime task scheduling. The prompt-visible raw Cartesian space contains
  1,327,104,000 combinations before deterministic safety/applicability normalization.
- Added backend-aware normalization, explicit one-warp launch-bounds lowering,
  optional NVCC register caps, exact run-owned format context, and removed the
  obsolete source-code API-key default in favor of environment/model config.
- Added `python/test/run_v7_strategy_feedback.py`: iteration 0 heuristic plus
  three Agent iterations, isolated generation/build/E2E/NCU artifacts, strict
  serial GPU execution, correct-only best selection, incremental summaries, and
  resumability under the existing v7 run directory.
- Added reusable NCU collection/parser support and persisted configure/build/run/
  profile commands. Updated runtime strategy/feedback skills to match the v2
  space and fixed-format semantic-backend contract.
- Corrected the legacy fake `stage_shifts=[0]/[2]` axis, which the three-stage
  compiler ignored because its length was invalid. The v2 space now supplies
  two legal inter-stage distances in `1..3`; `[1,1]` is canonicalized to `[]`.
  JSON Output and campaign-fingerprint resume validation were also enabled.
- Validation completed without paid calls: five strategy/NCU tests, focused
  semantic-fusion/SR-BCRS tests, all-26 dry-run (24 executable, two unsupported),
  Python compilation, and an ASIC NVCC smoke proving tile=32, stage shifts=[2,3],
  launch-bounds=4, segment=64 and maxrregcount=96 reach the binary. The current environment could
  not contact the NVIDIA driver, so GPU timing/NCU/Agent iterations were not
  started. Full design, commands and evidence are in
  `docs/codex/v7_strategy_agent_ncu_feedback_20260812.md`.

## 2026-08-11 — v7 format provenance/correctness audit and kernel-mechanism guides

- Added a read-only reproducible audit (`python/test/audit_v7_format_e2e.py`) for
  the v7 format run. It cross-checks each run-owned format SHA, generated include
  SHA, backend plan, CMake cache, executable, logs, N=512/warmup=3/repeat=32
  settings, and full-output TF32 correctness gate without re-running GPU work.
- Audit result for `20260722_232100_mask_deepseek_v4_pro_format26_v7_empirical_feedback`:
  24/24 executable formats have an unbroken compile-to-run provenance chain and
  pass numerical validation over 4,593,541,632 output elements; two valid DSL
  formats (YeastH and eu-2005) remain explicitly backend-unsupported and are not
  counted as executed or correct.
- Added operator-engineer-oriented explanations of the optimized fixed SR-BCRS
  path and the redesigned-format path in
  `docs/codex/sparsene2_kernel_optimization_20260811.md` and
  `docs/codex/format_agent_redesign_kernel_performance_20260811.md`. The second
  document explicitly records the TF32/deterministic-value validation boundary
  and the follow-up work required for unsupported/MCO formats.
- Expanded both guides with cross-layer maintenance contracts, concrete backend
  work packages, NCU hypotheses, same-machine ablation matrices, paper-ready
  claim/evidence boundaries, reproducibility fields, and suggested figures/RQs.
  Historical A100/A800 comparisons remain background evidence rather than a
  substitute for a controlled paper experiment.

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
## 2026-08-11 — v7 格式算子全量验证与加速比图

- 修复 custom-format 生成入口读取全局 `formats/custom` 旧导出模块的问题；现在从
  指定 run 的 `formats/<dataset>.format.json` 直接构建 DSL，并在 generation
  manifest 中保存 artifact SHA256，避免跨 run 静默替换格式。
- 添加严格单 GPU 串行的 `run_v7_format_e2e_batch.py`：24 个 v7 格式完成
  生成、sm80 编译、完整正确性和 N=512 FP32 warmup=3/repeat=32 计时；YeastH 与
  eu-2005 被诚实记录为当前无等价语义后端的 `unsupported`。
- 输出可复现论文风格 26 数据集加速比图和伴随来源 CSV，记录在
  `docs/codex/v7_speedup_figure_20260811.md`。

## 2026-08-15 — Sparsene-2Agent 四系统论文图表

- 补测原批量任务中因 CUDA/API 不可用而失败的 DD、Ga41As41H72、OVCAR-8H、
  Si41Ge41H72 和 Yeast iteration 0；五者均完成三次 N=512 FP32
  warmup=3/repeat=32 测量并通过正确性。
- 修复策略反馈 `--resume` 将 `failed` 当作终态的问题：只缓存 validated/incorrect，
  失败评估优先复用已保存的 structured strategy 后重试。
- 新增 `build_sparsene_2agent_comparison.py`，严格从 iteration 0--5 的正确候选选择
  最低 generated time，生成 26 数据集四组柱状图、来源 CSV、审计 JSON 和逐行最佳
  加粗的 LaTeX table。
- Sparsene-2Agent 几何平均为 1.8472x，23/26 快于 cuSPARSE、16/26 快于原
  Sparsene、11/26 快于 Sparsene 2.0，9/26 为四者最佳；总体仍低于 Sparsene 2.0
  的 1.9791x，详细证据与论文表述边界见
  `docs/codex/sparsene_2agent_26dataset_comparison_20260815.md`。
