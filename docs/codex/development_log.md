# Codex Development Log

## 2026-10-14 — Four-version SparseKG model comparison table

- Generated a 26-dataset N=512 FP32 LaTeX/CSV comparison for Sparsene 1.0,
  SparseKG (DeepSeek v4 Pro), SparseKG (GPT-6.1-Sol), and SparseKG (Opus 5.5).
  cuSPARSE is normalized to 1.0 and omitted; each row bolds the maximum using
  full precision. GPT and Opus select the fastest correctness-gated result
  across initial and retry summaries; DeepSeek uses the previously audited
  four-method comparison data; Sparsene uses the paper ratios in `summary.csv`.
- Geometric mean speedups: 1.915x, 1.858x, 1.616x, and 1.104x respectively;
  fastest-row counts are 11, 11, 4, and 0. Opus ASIC_680k uses canonical
  TCF-MCO fallback rather than its proposed heavy-row segmented format. The
  table caption records cross-campaign cuSPARSE normalization and this caveat.
- Builder: `python/test/build_four_model_speedup_table.py`; outputs:
  `logs/e2e_runs/incumbents/a100_a800_n512_fp32/four_model_speedup_comparison_table.tex`
  and the companion `four_model_speedup_comparison_data.csv`. Python
  compilation and `git diff --check` passed; no benchmark was run.

## 2026-10-14 — Opus backend-fix retry follow-up

- Audited `strategy_claude_opus_5_5_3iter_w2_r16_backendfix_retry`: all 20
  selected datasets produced a best correctness-gated operator; 14 summaries
  are `completed`, six are `completed_with_failures`, and zero are `failed` or
  `unsupported`. ASIC_680k now runs through canonical TCF-MCO (6.81485 ms,
  0.726x cuSPARSE), not its unimplemented heavy-row/segmented representation.
- The six warning states each contain one failed intermediate Agent call:
  two content-filter/empty responses, two malformed strategy payloads, and
  two TLS EOFs. Other iterations and the incumbent operator remain usable.
  Updated the retry wrapper to read the latest retry summary by default, so a
  second start selects only these six; `--resume` reuses validated iterations.
- No new model or GPU work was started during this audit. Details are in
  `docs/codex/asic680k_backend_fallback_20261014.md`.

## 2026-10-14 — ASIC_680k canonical backend fallback and failed-case retry

- Traced ASIC_680k's `X_s` rejection to a three-level segmented MCO proposal
  taking the sparse-row SR-BCRS exact branch, followed by canonical TCF
  fallback rejection of its unimplemented `stable_heavy_row_partition`.
- Enabled an explicit, auditable canonical TCF-MCO physicalization for this
  standalone reorder token. The source DSL remains immutable; plan notes state
  that heavy-row partitioning and SEG_K segmentation are not executed. Other
  unsupported or compound reorder contracts remain rejected.
- Added a failed-case wrapper that selects the 19 `completed_with_failures`
  datasets plus ASIC_680k from the historical summary, and runs a separate
  tuning name using the existing Opus 5.5, N=512, warmup=2, repeat=16,
  three-agent-round settings.
- Validation: all 26 format artifacts pass backend selection, physicalization,
  DSL validation, RTS derivation, and physical backend selection; ASIC_680k
  now selects the canonical TCF-MCO plan. Both launchers pass `bash -n` and
  `git diff --check` passes. Pytest is unavailable in the configured venv, and
  no API/GPU execution was performed. Details: `docs/codex/asic680k_backend_fallback_20261014.md`.

## 2026-10-14 — Opus 5.5 strategy result audit and v7 failure guardrails

- Audited the 26-dataset Opus 5.5 strategy result against the GPT-6.1 Sol
  campaign, the stored DeepSeek v4 Pro two-agent comparison, and Sparsene 1.0
  paper speedups. There are 25 executable correct-best records (13 faster than
  cuSPARSE; geometric mean 1.113x), but zero strict wins over all four
  comparators. Details and source provenance are in
  `docs/codex/opus55_strategy_campaign_analysis_20261014.md`.
- Classified strategy-call failures: 41 TLS EOF/endpoint failures, 9
  incomplete SSE streams, and one read timeout; these are OpenLux transport
  failures, not CUDA build failures. ASIC_680k is a backend capability gap.
  pwtk iteration 1's 5,345 mismatches came from requesting a direct/synchronous
  B path that the v7 semantic fusion still lowers as SM80 cp.async into SMEM.
- Scoped fixes to `run_v7_strategy_feedback.py`: v7 now exposes/normalizes only
  the implemented B-load async/materialization contract and retries only
  transient strategy transport errors (three attempts maximum with audit
  records). No shared codegen or non-v7 operator path changed.
- Validation: 14 focused unit tests passed, Python compilation passed, and
  `git diff --check` passed. No API/GPU rerun was started; prior outputs remain
  untouched and post-fix hardware confirmation is still required.

## 2026-10-14 — fix Opus 5.5 strategy launcher startup permissions

- Fixed `run_claude_opus_5_5_strategy_26.sh start`: the launcher is intended to
  be run with `bash`, but `setsid` previously tried to execute the non-executable
  script path directly. It now starts `setsid bash <script> _worker ...`, so no
  executable-bit change is needed and the existing process-group stop behavior
  is preserved.
- Confirmed the reported failure occurred before the Python worker/API/GPU
  stages; no campaign work was lost or needs a model-call retry. Validation:
  `bash -n` and `git diff --check`; no Agent or GPU work was started.

## 2026-10-14 — Claude Opus 5.5 fixed-format strategy launcher

- Added `scripts/run_claude_opus_5_5_strategy_26.sh` for the completed
  `20261014_002307_claude_opus_5_5_format26` artifacts: heuristic baseline plus
  three Claude Opus 5.5 strategy rounds, N=512, warmup=2, repeat=16, and three
  independent measurements, with NCU feedback enabled.
- The launcher provides local preflight, background start, live tail, status,
  and process-group-scoped stop actions. It preserves the saved format, puts
  strategy artifacts in a dataset-local isolated tuning directory, overlaps
  two agent/data pipelines while the v7 process lock serializes GPU operations,
  and keeps compiler jobs single-threaded.
- Read-only preflight verified the format-design batch is 26/26 accepted and
  the current backend supports 25/26. ASIC_680k's heavy-row partition design
  remains unsupported (`TCF-MCO backend requires axis X_s`); the driver will
  retain and report it as unsupported rather than replace the proposed format.
- Inspected current `run_v7_strategy_feedback.py --help` and source. No model
  call, compile, GPU correctness, benchmark, or NCU command was launched.

### 2026-10-13 follow-up — GPT-6.1 Sol request-envelope diagnosis

- Rechecked `20261013_131551_1057216_gpt-6.1-sol_format26`: effective output
  budget was already 8,192; the small preflight passed in 4.3 s, but all eight
  calls for the first four datasets disconnected after ~165–180 s. This rules
  out the previous claim that the output limit alone caused the failures.
- Fixed the GPT-6 request contract: use `max_completion_tokens`, explicit low
  reasoning, and streaming (with provider usage); omit temperature while
  reasoning is active. The Sol preflight now exercises that envelope.
- Transport fallback no longer launches a redundant corrective request; the
  quality-gate retry remains for actual model-generated invalid designs.
- Six focused tests, Python compilation, shell syntax, and diff checks pass.
  The later one-shot ASIC_680k probe completed over the corrected request in
  161.4 s (27,869 input / 4,706 output tokens), confirming end-to-end API
  transport. It did not pass the format quality gate: alternatives used the
  detailed physical-ledger schema rather than required `structure`/`pros`/`cons`;
  with max-attempts=1, no corrective request was sent. No format was exported
  and no operator ran. Details are in
  `docs/codex/gpt6_sol_gateway_disconnect_20261013.md`.

## 2026-10-13 — GPT-6.1 Sol long-request disconnect

- Diagnosed the first full format-design run: its one-word connectivity probe
  succeeded, but two ~83–84k-character prompts sent with a 32,768-token output
  budget were closed by the OpenLux gateway after roughly 2.5–3 minutes. The
  errors occurred before any model response, so these were transport failures,
  not quality-gate/DSL failures; failed-call token figures are local estimates.
- Reduced the GPT-6.1 Sol launcher completion default to 8,192 and instructed
  the format architect to avoid redundant JSON explanations. The global
  launcher default is unchanged. Existing in-flight processes are unaffected.
- Did not start another paid batch or add retries for ambiguous disconnects;
  see `docs/codex/gpt6_sol_gateway_disconnect_20261013.md` for the evidence,
  limits, and safe retry guidance.

## 2026-08-22 — nested Format-Agent/Strategy-Agent feedback campaign

- Added a two-level per-dataset loop: three complete Format-Agent designs, each
  followed by a deterministic strategy baseline and five NCU-guided Strategy
  Agent rounds. Final selection uses minimum median generated-kernel time across
  the 18-candidate budget, with full-output correctness as a hard gate.
- Added compact outer-loop feedback containing prior format signatures,
  correctness/normal timing, baseline-to-best strategy gain, physical padding,
  active-tile/metadata/restore costs, effective task distribution, and the best
  normalized strategy. Raw NCU remains private to the fixed-format inner loop.
- Added isolated run-owned directories per format candidate, resumable nested
  invocation of the existing v7 Strategy runner, dataset/global summaries, and
  publication of the best format/operator. Unsupported or failed formats are
  retained and reported as candidate-budget gaps rather than silently replaced.
- Added the initial three-dataset wrapper for ASIC_680k, ddi, and OVCAR-8H and
  reserved run directory `20260822_mask_joint_format3_strategy5_v1`.
- Local validation only: Python compilation, four focused unit tests, shell
  syntax, and a three-dataset dry-run all passed. No paid Agent, NVCC, GPU timing,
  or NCU call was launched in this development turn.

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
  output and `sparsene_llm_sog_deprecated/scripts/run_asic_strategy_feedback_12.sh` for baseline plus 12
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

## 2026-08-25 — 嵌套 Format/Strategy 全26数据集运行阻塞修复

- 定位 `ASIC_680k/format_02` 并非卡在模型/GPU，而是合法
  `ROW,ROW,COL,COL` 层次触发 RTS `_squeeze_new()` 无效同方向交换；循环不断前插 Swap，
  导致单核100% CPU和约4.5 GiB常驻内存。
- 将轴归并改为“相邻同方向优先、完全交错时才交换不同方向轴”，加入重复状态检测和
  确定性步数上限；保存的第三轮响应现在可完成 DSL normalization 和 RTS 推导。
- 保持 ME-TCF、HME-TCF、BIT-TCF、BIT-BSR、SR-BCRS 原有 RTS 序列不变，并新增三项
  收敛/兼容回归测试。
- 格式质量门重试现在携带前次提案摘要和精确 validator 错误，避免无反馈重复；新增
  `scripts/run_joint_format_strategy_26datasets.sh`，固定3个外层格式、每格式0--5策略、
  N=512 FP32、warmup=3、repeat=32和三次测量，GPU阶段保持串行。
- 未调用付费模型或GPU；Python编译、20项聚焦测试、保存响应重放、5种内置格式回归和
  26数据集 dry-run 均通过。详细证据见
  `docs/codex/joint_format_strategy_feedback_20260822.md`。

## 2026-08-25 — cuSPARSE/Sparsene/双智能体三方法论文统计

- 从 v7 run 的已审计正确候选中派生三方法26数据集 CSV、PNG/PDF柱图、逐行最佳加粗的
  LaTeX表和来源审计；cuSPARSE固定1.0，不再展示 Sparsene 2.0。
- 双智能体在16/26个数据集超过 Sparsene 1.0；胜出子集相对 Sparsene 的算术/几何平均
  加速为1.0723×/1.0709×，相对 cuSPARSE 的算术/几何平均为2.8862×/2.5056×。
- 在 run 内新增 `format_strategy_agent_optimization_summary.md`，用 `cant` 解释格式层
  16×8→16×16 的工作量压缩，用 `ddi` 解释32×32+重排格式和资源感知策略的协同收益，
  并如实记录总体 GeoMean 仍低于 Sparsene 1.0。
- 新增可复现入口 `python/test/build_three_method_agent_comparison.py`；Python编译、数据
  门禁、SHA审计、图像检查和LaTeX静态结构检查通过，环境无TeX引擎故未编译PDF表。
  详细记录见 `docs/codex/three_method_agent_comparison_20260825.md`。
- 按 run 内全部172个审计调用追加逐数据集 token 表：Format 28次、Strategy 144次，
  输入3,535,796、输出1,071,261、合计4,607,057 tokens；统计包含失败、重试、早期轮次
  与 ASIC 12轮探索，并注明25个 endpoint 失败请求的输入为本地估算而非供应商账单值。

## 2026-08-26 — 跨运行 correctness-gated 最优算子维护库

- 新增 `logs/e2e_runs/incumbents/a100_a800_n512_fp32`，按数据集维护当前最低 kernel 时间及
  对应的算子、Format DSL、Strategy、generation manifest、来源与 SHA256；旧 incumbent
  以内容寻址版本永久保留，更新使用文件锁和原子 current 切换。
- 门禁固定为单卡 SM80/108 SM、N=512 FP32/TF32、warmup=3、repeat=32、三次中位数与
  全部输出正确；只有 freshly generated 且严格更快的兼容候选允许替换。
- 按时间导入 v7 固定格式+策略运行和 20260825 联合 Format×Strategy 运行后，26/26 个
  数据集覆盖完整：16 个 current 来自前者、10 个来自后者，组合 GeoMean 为 1.9652×，
  24/26 快于 cuSPARSE；38 条历史事件保留了导入过程中的连续改进。
- 联合 runner 和 3/26 数据集 wrapper 现在默认读取此性能保底，把精简 incumbent 信息提供
  给 Format Agent，并在每个数据集结束后立即尝试更新；新一轮整体退化不会再丢失历史最好
  算子。
- 新增 updater CLI 和 focused tests，覆盖严格替换、慢候选保留、错误/协议拒绝、版本历史
  及联合选择保底。设计、维护命令和实验边界见
  `docs/codex/e2e_incumbent_registry_20260826.md`。
- 从论文 A100/N=512/FP32 vector PDF 可重复恢复26个 Sparsene 1.0 柱高，并加入维护库
  `summary.md/CSV`：算术平均2.1869×，四舍五入严格复现正文2.19×，几何平均1.9150×。
  原始日志逐项最近变体26/26均在5%内；九变体事后最快上包络为2.2324×，主要因
  `reddit`、`mip1` 选择不同而不能冒充论文逐项序列。新增逐项 baseline/validation CSV、
  audit provenance 及不改变 incumbent/history 的 summary-only 刷新入口。

## 2026-08-26 — 性能优化 Skill 与双智能体提示词维护

- 将 Sparsene 2.0、固定格式双智能体、ASIC_680k 策略实验和跨运行 incumbent 的经验整理
  为统一物理工作量账本：metadata、A/B movement、restore、实际 MMA/padding、store/
  reduction、task/grid imbalance 和 synchronization。
- Codex 开发 skill 新增 `performance-optimization.md`，四个运行时 Agent skill 同步加入
  语义保持跨层特化、格式物理等价、incumbent 保底、有效策略字段和可证伪实验规则。
- Format Agent 提示词现在必须建立“精确特征→表示机制→物理工作→预期效果→反证”的
  因果链；Strategy Agent 必须先分类瓶颈、通常只改1--3个有效字段并说明收益/代价/反证，
  representation-bound 或搜索饱和时允许空 patch。
- 修复 single-shot 策略请求完整 schema 与提示词错误要求 `strategy_patch` 的协议冲突；
  single-shot 现返回完整配置，只有反馈迭代返回紧凑增量。
- 未调用 GPU、NCU 或付费 Agent；两个模块编译、13项 Format 测试、9项 Strategy 测试和
  10项 joint/incumbent/RTS 测试通过。详见
  `docs/codex/performance_optimization_skill_and_prompt_20260826.md`。

## 2026-09-02 — 项目名称统一为 SparseKG

- 当前项目名称确定为 `SparseKG: A Matrix- and Architecture-Aware Agentic Compiler for
  SpMM Kernel Generation`；根 README、Codex/运行时 Skill、Format Agent prompt 和 e2e CLI
  已统一使用该身份。
- 未来联合运行默认目录改用 `sparsekg`，但历史 run ID、`/home/huoda/sparsene`、Python
  package、venv、Sparsene 论文/1.0/2.0 名称和性能契约保持不变，避免破坏复现与接口。
- incumbent 与新论文统计审计采用 `sparsekg.*` schema；用户可见榜单、三/四方法 PNG/PDF
  和 LaTeX 表头由 MASK/Sparsene-2Agent 更新为 SparseKG，内部历史 CSV key/文件名保留。
- Python 编译、20项 Format/incumbent/joint 测试和两个 wrapper 语法检查通过；在不调用
  Agent/GPU 的情况下重建图表并 summary-only 刷新 26 数据集 incumbent，性能数值未变。
  详细边界见 `docs/codex/sparsekg_project_rename_20260902.md`。

## 2026-09-02 — incumbent cuSPARSE/Sparsene/SparseKG 三方法柱状图

- 新增 `build_incumbent_three_method_comparison.py`，交叉校验 incumbent `summary.md` 和
  `summary.csv` 的26个数据集后，按既有论文风格生成 PNG/PDF、源数据 CSV、审计 JSON 和
  方法说明。
- cuSPARSE 固定1.0；Sparsene 使用论文 A100 逐数据集基线；SparseKG 使用当前跨运行
  correctness-gated incumbent。GeoMean 分别为1.0000×、1.9150×和1.9652×，SparseKG
  在24/26个数据集上超过 cuSPARSE、18/26个数据集上超过 Sparsene。
- incumbent README 记录了派生图和重建命令；脚本编译、26行数据/哈希/数值断言、PNG
  目视检查和 `git diff --check` 通过。未调用 Agent、GPU、NVCC 或 NCU。详见
  `docs/codex/incumbent_three_method_figure_20260902.md`。
- 同一生成器新增论文用 LaTeX 表格：26个数据集和 GeoMean 均按全精度值选出逐行最佳方法，
  三位小数展示并加粗；表格哈希、判定规则和显示精度写入审计 JSON。
- 按论文排版需要移除恒为1.0的 cuSPARSE 数值列，caption 保留归一化基线说明；表格仅展示
  Sparsene/SparseKG，并继续按两者全精度值加粗每行最佳结果。

## 2026-10-13 — OpenLux GPT-6 Luna 格式设计配置

- 在跟踪中的 `llm_models.yaml` 添加 OpenAI-compatible `gpt-6-luna` 模型项，endpoint 为
  OpenLux `/v1`，凭据只从 `OPENLUX_API_KEY` 环境变量解析；同时移除该文件中已有模型的
  明文 key，并取消对该配置文件的忽略规则，便于安全地提交模型元数据。
- 给出沿用格式设计批处理脚本的全数据集格式生成命令；此脚本阶段不会进入算子生成、编译
  或测量。配置解析和26个 feature 输入检查通过，未发起真实 API 请求。
- 用户在会话中粘贴的 key 应立即在代理网站撤销并轮换。旧 key 曾存在于 Git 跟踪文件的
  历史提交中；若已推送，当前文件清理不会改写历史，旧 key 仍需撤销。
- 详见 `docs/codex/openlux_gpt6_luna_format_design_20261013.md`。
- 新增 `scripts/run_gpt6_luna_format_design_26.sh`，要求 key 已在环境中导入，自动先做
  一次 GPT-6 Luna 连通性检查，成功后才后台启动纯格式设计批处理；详见
  `docs/codex/gpt6_luna_format_design_launcher_20261013.md`。

## 2026-10-13 — 格式设计质量门修复与有限纠错重试

- 修复 fallback 决策缺少 `agent_metadata.run_dir` 时报告渲染抛 `KeyError` 的问题，确保
  单数据集 API/解析失败会被记为失败结果并继续批次。
- 原有 `--max-attempts` 循环只会重复原提示。现在失败反馈会携带确定性质量问题和上一轮
  DSL 摘要，让格式智能体进行有针对性的修正；格式 prompt 明确列出 custom atomic 必填项
  和合法 layout。
- 通用格式设计批处理和 GPT-6 Luna wrapper 默认最多尝试2次（首次+一次纠错），只导出通过
  质量门的格式。未通过最终质量门的数据集继续被记录，不中断后续数据集。
- 未调用付费模型/GPU，未执行自动化测试；修复依据为新近运行中的 DD 异常栈与 fallback
  结果结构。细节见 `docs/codex/format_design_quality_gate_retry_20261013.md`。
- 用户改用 GPT-6.1 Sol 后，在 `llm_models.yaml` 增加对应 OpenLux 模型 ID，并将专用启动入口
  更新为 `sparsene_llm_sog_deprecated/scripts/run_gpt6_1_sol_format_design_26.sh`；仍要求环境变量提供 key，先进行一次
  连通性检查，再最多两次尝试完成每个数据集格式设计。此前 GPT-6 Luna 运行记录保持原样。
- ASIC_680k 的成功单次响应 DSL 有效，但门控仅因其备选方案采用 `levels`/`atomic`/
  `semantics_cn`/`physical_work_ledger` 而未写重复的 `structure`/`pros`/`cons` 被拒。
  提示词现在明确列举字段；门控同时识别简表和完整物理账本两种形式，不降低 DSL、证据、
  语义和实现要求。运行时 `.agent/skills/format-design/SKILL.md` 同步该合同。
- 新增回归测试覆盖探针风格的详细备选、缺失物理账本项和旧简表。首次测试因未设置
  `PYTHONPATH=../` 导致导入失败；设置后测试执行，仅提示词断言字符串不匹配，已修正并重跑。
  记录见 `docs/codex/format_quality_gate_alternative_schema_compatibility_20261013.md`。

## 2026-10-13 — GPT-6.1 Sol 策略智能体 26 数据集算子评测启动器

- 检查 `20261013_140947_1088479_gpt-6.1-sol_format26`：manifest 有 26 个数据集，实际通过质量门
  并导出的格式 JSON 为 25 个；`conf5_4-8x8-10` 因缺少 `rationale_cn` 未通过质量门。
- 将全部 25 个 artifact 加载并按 SM80 后端注册表检查：12 个格式当前可执行，另 13 个是合法
  研究 DSL 但没有匹配后端或其 ABI 不匹配；因此不能声称当前版本可以为全部 26 个生成并测量算子。
- 新增 `sparsene_llm_sog_deprecated/scripts/run_gpt6_1_sol_strategy_26.sh`，固定调用当前 v7 策略反馈流程：迭代0启发式基线+
  最多5次 GPT-6.1 Sol 策略迭代，N=512、FP32、warmup=3、repeat=32、每候选3次独立测量及正确候选
  NCU。默认两个 dataset workers 重叠非 GPU 阶段，现有驱动的锁保证 GPU correctness/timing/NCU 串行。
- 启动器支持 check/start/status/tail/stop；后台进程独占 session/process group，stop 仅向记录的本次
  PGID 发信号，30秒未退出才对该组强制结束。实时日志写到原格式 run 目录，重启使用 `--resume`。
- `check` 已真实运行并生成 `strategy_gpt6_1_sol_5iter_preflight.json`：确认12个可执行格式、14个
  缺失/后端不支持项；脚本 `bash -n` 通过。未调用策略 API、未编译、未运行 GPU/N CU。
- 详见 `docs/codex/gpt6_sol_strategy_26_operator_campaign_20261013.md`。

## 2026-10-13 — 策略预检排除项与 NCU profiler 版本修复

- 对14个预检排除项逐一对照 Format DSL、确定性 capability 与 backend registry：13个为当前后端
  尚未实现的 frontend-research 设计/ABI，不是格式 DSL 错误；conf5_4-8x8-10 则仅因说明性
  `rationale_cn` 缺失被质量门误拒。将该字段改成非阻塞 warning，并增加回归测试。
- 复现 NCU 失败来自自动选择版本：`/home/huoda/local/cuda/bin/ncu` wrapper 在当前机选择
  Nsight Compute 2026.3；同一已生成可执行文件改用 Toolkit 固定版 NCU 2025.1 后连接 A800
  成功并输出指标。将策略/benchmark 默认值及相关脚本固定到已验证的 2025.1 binary，保留显式覆盖。
- 验证：质量门专项 unittest 4项通过；修改的 Python 文件 `py_compile` 通过；campaign `check`
  仍报告12个已实现 backend、13个后端覆盖缺口及旧 conf5 artifact（旧实验目录未重写）；NCU
  2025.1 在现有 Ga41As41H72 binary 上完成 GPU profile，2026.3 wrapper 同条件失败。
- 未改变 kernel/backend 语义，未重跑26数据集策略任务；13个 backend 缺口需要分别补齐预处理、
  ABI、lowering/kernel 和正确性路径后才能执行。详见 `docs/codex/20261013_backend_coverage_and_ncu.md`。

## 2026-10-13 — 前端格式 canonical TCF 后端物理化扩展

- 在保留原生精确后端优先级的基础上，新增显式 frontend-canonical TCF 物理化 fallback：经 DSL
  校验、原子 ROW/COL 坐标映射明确的 COO/MCO/dense/CSR/ELL/DIA/custom 设计，可落到现有 MCO 或
  dense TCF host ABI，再经既有 RTS/Computent/OPIR/NVIR/CUDA 路径。Custom atomic 必须有明确逻辑映射
  和恢复算法；只允许已实现的单一外层行重排。原始 DSL 不覆盖，physical spec/hash、backend 和转换边界
  写入生成 manifest。此路径不是 CSR/ELL/custom 的原生 GPU 存储实现，kernel-only 计时排除 host packing。
- portfolio 不再对 canonical fallback 虚构原子/Tile sibling。旧 conf5 格式在 resume 中由当前质量门本地
  重验，缺少 rationale_cn 作为 warning，结构化 DSL 不变且不调用模型，26 个格式 artifact 全部恢复。
- 26 格式 preflight 现为 26/26；在独立 backend_codegen_smoke 目录对全部 26 格式生成 CUDA include 均成功，
  原先的算子目录未被覆盖。A800 N=512 FP32、warmup=2/repeat=16 代表性 e2e：ASIC_680k dense
  correctness PASS、0.169x cuSPARSE；DD MCO correctness PASS、1.139x；dc2 correctness 按 TF32 aggregate
  阈值 PASS、0.723x，严格逐元素统计仍报告36处偏差点。性能显示 coverage 不等于普遍加速，后续策略迭代/格式
  选择仍需关注 ASIC_680k 与 dc2。
- 策略启动脚本默认更新为启发式 baseline + 最多3次策略迭代、N=512 FP32、warmup=2、repeat=16；采用新 tuning
  名隔离旧 5-iteration 输出。GPU correctness/timing/NCU 仍由共享锁串行化，NCU 继续 pin 到当前已验证的
  Nsight Compute 2025.1。脚本 check 返回26个 backend 全部可执行；未启动模型 campaign。
- 13项格式设计 capability unittest 通过；修改 Python 文件 `py_compile`、shell `bash -n` 通过。完整详情、
  limitation 和执行指令见 `docs/codex/frontend_backend_physicalization_and_strategy_campaign_20261013.md`。
- 用户首次启动新脚本后 driver 立即退出，日志显示 v7 runner 仍硬编码拒绝 warmup=2/repeat=16（要求3/32）。
  移除旧采样值限制，只保留 N=512 与 warmup/repeat 合法性检查，并把真实 warmup/repeat 带入策略反馈上下文；
  launcher 新增启动后存活探测，失败时将 driver 日志尾部同步打印到终端。失败 campaign 尚未运行数据集，也未调用策略模型。
- 验证：runner `py_compile`、策略空间和格式设计 unittest 13项、shell `bash -n`、`git diff --check` 通过；未重新
  `start`，避免擅自调用模型/API。修复后可用同一命令重启。

## 2026-10-13 — GPT-6.1-Sol 未完成九数据集重跑脚本

- 基于 `strategy_gpt6_1_sol_3iter_backend_v1_summary.json` 将 9 个
  `completed_with_failures` case 单独列为重跑目标；这些 case 是部分策略轮次未完成，主要错误为暂态 API TLS EOF，
  并非判定已有算子错误。保留原 campaign 与成功算子结果。
- 新增 `sparsene_llm_sog_deprecated/scripts/rerun_gpt6_1_sol_failed9.sh`，使用新的 tuning 名称隔离 artifacts/summary，限制为 9 个 case，
  GPT-6.1-Sol、baseline+3 iter、N=512 FP32、warmup=2/repeat=16、NCU 2025.1；默认单 dataset worker 降低 API 并发。
  支持 check/start/status/tail/stop，停止操作只向该脚本记录的进程组发信号。
- `bash -n`、`git diff --check` 通过；`check` 确认9个 dataset 均存在于 manifest 且格式文件齐全。未启动 API、GPU、编译或 NCU。
  详见 `docs/codex/gpt6_sol_failed9_retry_20261013.md`。

## 2026-10-13 — 四版本 26 数据集速度比汇总

- 合并 GPT-6.1-Sol 原始17个完整数据集与重跑9个数据集的正确候选；重跑 summary 确认9/9 completed，策略迭代均通过 runner 验证。
- 对 DeepSeek V4 Pro，从 2026-07-22 全26正确初始算子出发，逐数据集采用可用的正确策略候选（含 ASIC 12-iter、YeastH/eu follow-up 与 iteration-0 recovery），缺失策略结果时保留初始正确算子。与 Sparsene 1.0 paper ratios、GPT 数据统一转换为 speedup over cuSPARSE。
- 生成 `four_method_26dataset_speedup_data.csv`、`four_method_26dataset_speedup_table.tex` 和 `four_method_26dataset_speedup_summary.md`。统计：SparseNe paper/SparseKG DS/GPT 几何平均分别为 1.915/1.858/1.616x；快于 cuSPARSE 为 26/23/18 个；逐行最快为 11/11/4 个。
- 数据来源为不同运行/计时协议：DeepSeek 3/32、GPT 2/16、Sparsene 来自论文 A100。报告明确为跨运行参考，不更新要求 3/32 的正式 incumbent。CSV 与 summary 26 行/计数复核、LaTeX 每行加粗冠军复核通过。

## 2026-10-13 — 一次性 campaign 启动器归档

- 将 `tmp.sh` 及7个已结束、与固定模型/旧 run 目录绑定的 strategy/format campaign shell 启动器移动到 `sparsene_llm_sog_deprecated/scripts/`，保留文件原内容；归档 README 记录范围和使用注意事项。
- 更新引用旧路径的历史 Codex 文档。通用特征、格式 batch、联合格式、benchmark 与可复现 Python 报告工具保留原位置。被 Python 代码引用的 joint 3-dataset launcher 不移动。
- 用户澄清后确认使用代理站点 OpenLux，而非 Anthropic 官方 API；新增 `claude-opus-5` 的 OpenAI-compatible 配置，模型请求发往 `https://api.openlux.ai/v1`，密钥仅从 `OPENLUX_API_KEY` 环境变量读取。
- 新增 `scripts/run_claude_opus_5_format_design_26.sh`，覆盖26个数据集的格式设计；含离线配置/特征文件预检、启动前 OpenLux 连通性探测、实时日志、状态查看和仅停止本 campaign 的功能。连通性探针会记录请求/响应和 token 用量，但绝不保存密钥。
- 修改归档目录中的说明，记录本次旧启动器归档范围。`check` 无需 API key，可离线检查模型路由、26个特征文件及续跑目录；`start` 才要求密钥并实际发送探针请求。未在开发环境发起 OpenLux API 调用或运行格式生成。
- 验证：模型配置解析为 `claude-opus-5` / `openai_compatible` / `https://api.openlux.ai/v1`；26个特征文件存在；shell 语法和 `git diff --check` 通过。使用方式见 `docs/codex/claude_opus5_format_design_20261013.md`。

## 2026-10-13 — Claude Opus 5 中断后的 token 统计与优雅停止

- 检查已停止的 `20261013_214732_claude-opus-5_format26`：DD、Ga41As41H72 两次响应包含 OpenLux 实际 usage；ASIC_680k、OVCAR-8H、Si41Ge41H72 三次请求以 `RemoteDisconnected` 失败。runtime 已把失败请求的输入字符估算写进各自 `audit.json`，但格式 agent 将 transport 异常转换为 fallback 时没有传递该 usage，故 dataset 的 `token_usage.json` 错误显示 0+0。失败调用没有响应，输出 token 无法从本地恢复，输入估算也不等于代理账单的准确 token 数。
- 修复格式 agent 从 invocation audit 恢复失败请求 usage，并标注来源为 `estimated_after_failed_request`；batch token rollup 同时纳入 OpenLux 连通性探针。summary/status 分开累计 API 实报与本地估算请求，防止把估算误当成代理账单，不再依赖 batch 最后写出的 summary。
- batch 收到 SIGTERM/SIGINT 后设置停止标记，完成/记录当前数据集后生成带 `completed`、`pending_datasets`、`interrupted` 字段的部分 summary；不再在停止等待30秒后 SIGKILL 当前请求。超时仍未结束时 `stop` 返回“停止请求已发出”，`status` 显示 STOPPING，可继续等待当前请求保存 usage。
- 验证：format-design 与 agent audit 测试共20项通过；py_compile、shell `bash -n`、`git diff --check` 通过。旧 campaign 的5次数据集请求中2次成功、3次 RemoteDisconnected；Yeast 仅有 dataset_started 事件，没有 agent audit，说明停止发生在它发出模型请求之前。连同探针的 rollup 为输入140356、输出27700、总计168056；3份失败调用输入是字符数/4估算，并非代理侧账单数据，输出无法从本地恢复。未重跑数据集或调用 OpenLux。

## 2026-10-13 — OpenLux TLS EOF 瞬断重试

- 新的 Opus campaign 在连通性探针阶段报 `[SSL: UNEXPECTED_EOF_WHILE_READING]`。这发生在 TLS 握手而非模型鉴权/HTTP 层；同环境对 OpenLux 无凭据探测能完成 TLS 并收到预期 HTTP 401，支持代理链路瞬时 EOF 而非模型名或 URL 配错的判断。
- `call_openai_compatible` 仅对 TLS 握手 EOF（请求尚未发出）进行最多3次尝试，退避1/2秒；HTTP错误、RemoteDisconnected 等不自动重放，避免重复提交已可能被模型端处理的请求。
- 加入对应单测；agent runtime 测试共21项通过，Python 编译和 `git diff --check` 通过。未用真实 key 发起模型请求、未启动格式批次。

## 2026-10-13 — Opus 长格式请求断连复查

- 复查 `20261013_221228_claude-opus-5_format26`：小型连通性探针成功（36 input + 2 output tokens），但 ASIC_680k、DD、Ga41As41H72、OVCAR-8H、Si41Ge41H72 的完整请求全部失败；日志包含2次 `RemoteDisconnected` 和3次 TLS EOF。五个用户 prompt 均约84KB，输出上限32768，且本次 manifest 显示 `stream=false`。因此简单 key/URL 错误不符合证据；更可能是代理对长时间非流式响应的连接稳定性问题，但仅凭客户端无法确认 OpenLux 内部原因。
- Claude Opus 5 配置切换为 OpenLux SSE streaming；启动探针采用相同 transport。OpenLux 文档描述 Chat Completions 支持 `stream=true` 并以 SSE partial deltas 返回。客户端新增流中断保护：审计已收到的部分文本及其 API 用量（若有），否则单独估算，不丢弃部分响应。resume 时在 manifest 的 `resumptions` 追加生效配置，保留之前 stream=false 的原始运行记录。
- 旧 run 本次 5/5 均失败，且 summary 标记 `interrupted=true`，没有格式成功产物。当前改动仅做本地测试（agent/runtime 与 format-design 22项通过、py_compile、bash -n、模型配置解析、status 与 diff-check），没有新的 OpenLux/LLM 请求。需先用一个数据集确认 SSE 响应可完整返回，再恢复全量任务；若 streaming 下依旧断连，应向 OpenLux 提供运行时间、model、请求时长和 HTTP/TLS 错误，请其查 relay/upstream 日志。
# 2026-10-13 — Claude Opus OpenLux smoke authentication diagnosis

- Inspected `logs/e2e_runs/20261013_claude-opus-5_stream_smoke_DD`: its request
  snapshot had `api_key_present=false`; OpenLux returned HTTP 401 `Invalid
  token`, with no generated response. This rules out streaming, output length,
  and format validation as the cause of this run's failure.
- Added Python-visible credential checks to the Claude wrapper preflight and
  the general format-design batch entry point, so an unset/unexported key fails
  before an API request rather than appearing as a failed format proposal.
- No API key was inspected or recorded. No authenticated retry was possible in
  this environment; rerun after exporting the valid OpenLux key.
# 2026-10-14 — Claude Opus empty streaming response diagnosis

- The second DD smoke request recorded `api_key_present=true`, unlike the prior
  401. After 100.33 s it yielded no content, no API usage, and no
  `finish_reason`; the wrapper then surfaced an empty-response JSON parse error.
  This indicates an empty/incomplete SSE response, not malformed model JSON.
  The existing small connectivity probe completed in 5.56 s, so it only proves
  the short streamed path, not this 84-KB format prompt.
- The client now accepts complete-message SSE variants, retains safe event-shape
  diagnostics, and raises `StreamingInterruptedError` if the stream ends before
  `[DONE]` or a finish reason. This distinguishes incomplete transport from
  parsing/quality-gate failure. OpenLux documents SSE deltas terminated by
  `[DONE]` (see the focused diagnosis note).
- Validation: 10 `test_agent_runtime_audit` tests pass; Python compilation and
  `git diff --check` pass. No authenticated request was available here, so the
  actual OpenLux long-request behavior remains to be retried by the user.
# 2026-10-14 — OpenLux Claude Opus 5.5 format-design launcher

- Added `claude-opus-5-5` as a separate OpenLux model entry; kept the Opus 5
  config and launcher behavior available. Generalized the existing launcher to
  accept an explicit model and optional dataset subset, then added a dedicated
  5.5 wrapper with a one-dataset DD smoke mode and the full 26-dataset mode.
- The launcher continues to check that the API key is visible to Python and
  uses the existing stream integrity checks. No credential was changed or
  written. OpenLux's public pricing page advertises the exact model alias
  `claude-opus-5-5`, but a live account-level availability check remains
  necessary.
- Validation: both launchers pass `bash -n`; a protected fake key was used only
  with the offline `check` command, which resolved the new model and found all
  26 feature files. `git diff --check` passes. No model API request was made by
  this development session.
# 2026-10-14 — Opus 5.5 interrupted campaign continuation

- Audited `20261014_002307_claude_opus_5_5_format26`: 4/9 processed datasets
  succeeded, 5/9 failed, and 17 remained pending when the stop marker ended the
  batch. Failures were transport-level (two TLS handshake EOFs; three
  incomplete SSE responses), not `finish_reason=length` or invalid generated
  DSL. Successful outputs stopped normally at 11.8K–16.9K tokens.
- Added bounded exponential retry for explicit transient transport fallbacks;
  retained non-retry for auth/parse errors. Resume now preserves prior attempt
  records and assigns unique attempt numbers. Added `resume RUN_DIR` to the
  Opus 5.5 launcher; existing accepted outputs are reused, failed and pending
  data sets are retried/processed.
- Validation: 15 focused batch-quality/runtime-audit tests pass; Python
  compilation, both launcher `bash -n` checks, and `git diff --check` pass. The
  execution environment has no OpenLux key, so the continuation was not sent.
