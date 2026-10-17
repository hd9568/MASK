python cluster_plan_search_curve.py   \
    --plans /workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/plans_6156.txt   \
    --results /workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/plan_result.json   \
    --metric time_ms   \
    --max-reps 1   \
    --group-keep-ratio 0.85   \
    --fine-mode ucb   \
    --ucb-beta 1.6   \
    --fine-budget-per-group 12   \
    --profile-cost-sec 2   \
    --output-json cluster_search_curve_slow.json \
    --output-csv cluster_search_curve_slow.csv   \
    --output-plot cluster_search_curve_slow.png

python naive_simulator.py --plans plans_6156.txt --output ./sim_scores_6156.json --k 5 --format list
python cluster_plan_search_curve_simulator.py \
    --plans plans_6156.txt \
    --results plan_result.json \
    --metric time_ms \
    --strategy sim-online-hybrid \
    --sim-results sim_scores_6156.json \
    --sim-metric sim_cost \
    --real-budget 64 \
    --sim-warmup-k 256 \
    --explore-ratio 0.2 \
    --output-json hybrid_report.json \
    --output-csv hybrid_curve.csv \
    --output-plot hybrid_curve.png




# 更新后脚本 非实测
/usr/bin/python cluster_plan_search_curve.py \
    --plans plans_6156.txt \
    --results plan_result.json \
    --metric time_ms \
    --max-reps 1 \
    --group-keep-ratio 0.85 \
    --fine-mode ucb \
    --ucb-beta 1.6 \
    --fine-budget-per-group 12 \
    --profile-cost-sec 2 \
    --output-json cluster_search_curve_oracle_online.json \
    --output-csv cluster_search_curve_oracle_online.csv \
    --output-plot cluster_search_curve_oracle_online.png


/usr/bin/python cluster_plan_search_curve.py \
    --plans plans_6156.txt \
    --results plan_result.json \
    --metric time_ms \
    --max-reps 1 \
    --group-keep-ratio 0.85 \
    --fine-mode ucb \
    --ucb-beta 1.6 \
    --fine-budget-per-group 12 \
    --use-real-timing \
    --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build \
    --target-template "acc-plan_{plan_id}-tf32" \
    --make-jobs 48 \
    --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
    --output-json cluster_search_curve_real_online.json \
    --output-csv cluster_search_curve_real_online.csv \
    --output-plot cluster_search_curve_real_online.png

/usr/bin/python cluster_plan_search_curve.py \
    --plans plans_6156.txt \
    --results plan_result.json \
    --metric time_ms \
    --max-reps 1 \
    --group-keep-ratio 0.85 \
    --fine-mode ucb \
    --ucb-beta 1.6 \
    --fine-budget-per-group 12 \
    --use-real-timing \
    --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
    --target-template "acc-plan_{plan_id}-tf32" \
    --make-jobs 48 \
    --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
    --output-json cluster_search_curve_real_online_2.json \
    --output-csv cluster_search_curve_real_online_2.csv \
    --output-plot cluster_search_curve_real_online_2.png



# 更新了simulator+聚类模拟算法脚本，执行逻辑
python3 /workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/cluster_plan_search_curve_simulator.py \
  --plans <filtered_plans.txt> \
  --results <real_results.json> \
  --sim-results <sim_results.json> \
  --strategy sim-cluster-hybrid \
  --group-keep-ratio 0.35 \
  --real-budget 64 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --explore-ratio 0.2 \
  --ucb-beta 0.8


#! 将simulator+聚类算法与聚类算法合并
python3 cluster_plan_search_curve_simulator.py \
    --strategy cluster-search \
    --plans plans_6156.txt \
    --results plan_result.json \
    --metric time_ms \
    --output-json run_cluster_search.json \
    --output-csv run_cluster_search.csv \
    --output-plot run_cluster_search.png 


python3 cluster_plan_search_curve_simulator.py \
    --strategy sim-cluster-hybrid \
    --plans plans_6156.txt \
    --results plan_result.json \
    --sim-results sim_scores_6156.json \
    --sim-metric sim_cost \
    --metric time_ms \
    --real-budget 64 \
    --explore-ratio 0.2 \
    --sim-cluster-max-reps 3 \
    --sim-cluster-beta 0.5 \
    --output-json run_sim_cluster_hybrid.json \
    --output-csv run_sim_cluster_hybrid.csv \
    --output-plot run_sim_cluster_hybrid.png



# 是否使用真实测试 
    --use-real-timing \
    --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
    --target-template acc-plan_{plan_id}-tf32 \
    --make-jobs 48 \
    --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0"


#! 优先bad result

python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --output-json run_sim_cluster_hybrid_badstart.json \
  --output-csv run_sim_cluster_hybrid_badstart.csv \
  --output-plot run_sim_cluster_hybrid_badstart.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --bad-start-k 8 \
  --bad-start-source distance \
  --output-json run_cluster_search_badstart.json \
  --output-csv run_cluster_search_badstart.csv \
  --output-plot run_cluster_search_badstart.png

#! 渐进式提升
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --output-json run_sim_cluster_hybrid_badstart_ramp24.json \
  --output-csv run_sim_cluster_hybrid_badstart_ramp24.csv \
  --output-plot run_sim_cluster_hybrid_badstart_ramp24.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --bad-start-k 8 \
  --bad-start-source distance \
  --bad-start-ramp-steps 24 \
  --output-json run_cluster_search_badstart_ramp24.json \
  --output-csv run_cluster_search_badstart_ramp24.csv \
  --output-plot run_cluster_search_badstart_ramp24.png


#> 实测脚本
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_sim_cluster_hybrid_badstart_ramp24_real.json \
  --output-csv run_sim_cluster_hybrid_badstart_ramp24_real.csv \
  --output-plot run_sim_cluster_hybrid_badstart_ramp24_real.png


python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --bad-start-k 8 \
  --bad-start-source distance \
  --bad-start-ramp-steps 24 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_badstart_ramp24_real.json \
  --output-csv run_cluster_search_badstart_ramp24_real.csv \
  --output-plot run_cluster_search_badstart_ramp24_real.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --threshold 0.12 \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0 \
  --fine-budget-per-group 1 \
  --bad-start-k 128 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1024 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_slower_oracle_real.json \
  --output-csv run_cluster_search_slower_oracle_real.csv \
  --output-plot run_cluster_search_slower_oracle_real.png

#> 更差的起点，设置预算
python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --threshold 0.12 \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0 \
  --fine-budget-per-group 1 \
  --bad-start-k 128 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1024 \
  --cluster-real-budget 512 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_slower_oracle_real_budget512.json \
  --output-csv run_cluster_search_slower_oracle_real_budget512.csv \
  --output-plot run_cluster_search_slower_oracle_real_budget512.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --threshold 0.12  \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0  \
  --fine-budget-per-group 1 \
  --bad-start-k 128 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1024 \
  --cluster-real-budget 512 \
  --suppress-early-good-window 24 \
  --suppress-gap-history 6 \
  --suppress-gap-ratio 0.78 \
  --suppress-gap-min-history 3 \
  --suppress-early-good-threshold-ms -1 \
  --suppress-early-good-ratio-vs-best -1 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_dynamic_suppress_real_budget512.json \
  --output-csv run_cluster_search_dynamic_suppress_real_budget512.csv \
  --output-plot run_cluster_search_dynamic_suppress_real_budget512.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --threshold 0.12 \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0 \
  --fine-budget-per-group 1 \
  --bad-start-k 192 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1536 \
  --cluster-real-budget 384 \
  --suppress-early-good-window 96 \
  --suppress-gap-history 12 \
  --suppress-gap-ratio 0.92 \
  --suppress-gap-min-history 2 \
  --suppress-early-good-threshold-ms -1 \
  --suppress-early-good-ratio-vs-best 1.35 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_more_suppressed.json \
  --output-csv run_cluster_search_more_suppressed.csv \
  --output-plot run_cluster_search_more_suppressed.png

python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --results plan_result.json \
  --metric time_ms \
  --threshold 0.12 \
  --cluster-cache-file /workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/cache/cluster_6156_t012_a4_b2_g1.json \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0 \
  --fine-budget-per-group 1 \
  --bad-start-k 192 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1536 \
  --cluster-real-budget 256 \
  --suppress-early-good-window 96 \
  --suppress-gap-history 12 \
  --suppress-gap-ratio 0.92 \
  --suppress-gap-min-history 2 \
  --suppress-early-good-threshold-ms -1 \
  --suppress-early-good-ratio-vs-best 1.35 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json run_cluster_search_more_suppressed_cache.json \
  --output-csv run_cluster_search_more_suppressed_cache.csv \
  --output-plot run_cluster_search_more_suppressed_cache.png

#! 看起来还是不太行
python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt \
  --metric time_ms \
  --threshold 0.12 \
  --cluster-cache-file /workspace/sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/cache/cluster_6156_t012_a4_b2_g1.json \
  --max-reps 1 \
  --fine-mode ucb \
  --ucb-beta 2.0 \
  --fine-budget-per-group 1 \
  --bad-start-k 192 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1536 \
  --cluster-real-budget 256 \
  --suppress-early-good-window 96 \
  --suppress-gap-history 12 \
  --suppress-gap-ratio 0.92 \
  --suppress-gap-min-history 2 \
  --suppress-early-good-threshold-ms -1 \
  --suppress-early-good-ratio-vs-best 1.35 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_cluster_search_1.json \
  --output-csv results/run_cluster_search_1.csv \
  --output-plot results/run_cluster_search_1.png


python3 cluster_plan_search_curve_simulator.py \
  --strategy cluster-search \
  --plans plans_6156.txt  \
  --metric time_ms  \
  --threshold 0.12  \
  --cluster-cache-file cache/cluster_6156_t012_a4_b2_g1.json \
  --max-reps 1  \
  --fine-budget-per-group 0 \
  --bad-start-k 192 \
  --bad-start-source distance \
  --bad-start-ramp-steps 1536 \
  --cluster-real-budget 256 \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test  \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48  \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0"  \
  --output-json results/run_cluster_search_2.json  \
  --output-csv results/run_cluster_search_2.csv  \
  --output-plot results/run_cluster_search_2.png 

#! 设置threshold重新测试sim+search
#! 结果和之前相同
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --threshold 0.12 \
  --cluster-cache-file cache/cluster_6156_t012_a4_b2_g1.json \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_sim_cluster_hybrid_1.json \
  --output-csv results/run_sim_cluster_hybrid_1.csv \
  --output-plot results/run_sim_cluster_hybrid_1.png

#! 尝试一下sim全常数的方案（模拟没有sim的情况）
#> bad-start没有发挥效果
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results results/sim_scores_6156_const.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --threshold 0.12 \
  --cluster-cache-file cache/cluster_6156_t012_a4_b2_g1.json \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_sim_cluster_hybrid_constsim_1.json \
  --output-csv results/run_sim_cluster_hybrid_constsim_1.csv \
  --output-plot results/run_sim_cluster_hybrid_constsim_1.png

#! 设置bad-start逻辑是distance
python cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results results/sim_scores_6156_const.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source distance \
  --bad-start-ramp-steps 24 \
  --threshold 0.12 \
  --cluster-cache-file cache/cluster_6156_t012_a4_b2_g1.json \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_sim_cluster_hybrid_constsim_badstart_distance_1.json \
  --output-csv results/run_sim_cluster_hybrid_constsim_badstart_distance_1.csv \
  --output-plot results/run_sim_cluster_hybrid_constsim_badstart_distance_1.png


python plot_sim_vs_nosim_curve.py \
  --sim-json results/run_sim_cluster_hybrid_1.json \
  --nosim-json results/run_sim_cluster_hybrid_constsim_badstart_distance_1.json \
  --sim-label sim \
  --nosim-label no-sim \
  --output-plot results/plot_sim_vs_nosim_curve2.png \
  --output-csv results/plot_sim_vs_nosim_curve2.csv

#! 2026/04/04 重新测试sim+search与no-sim + search，重新设置阈值匹配模拟器的clusters总数
python3 cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results sim_scores_6156.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source sim \
  --bad-start-ramp-steps 24 \
  --threshold 0.20 \
  --cluster-cache-file cache/cluster_6156_t020_a4_b2_g1.json \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_sim_cluster_hybrid_2.json \
  --output-csv results/run_sim_cluster_hybrid_2.csv \
  --output-plot results/run_sim_cluster_hybrid_2.png

python cluster_plan_search_curve_simulator.py \
  --strategy sim-cluster-hybrid \
  --plans plans_6156.txt \
  --results plan_result.json \
  --sim-results results/sim_scores_6156_const.json \
  --sim-metric sim_cost \
  --metric time_ms \
  --real-budget 64 \
  --explore-ratio 0.2 \
  --sim-cluster-max-reps 3 \
  --sim-cluster-beta 0.5 \
  --bad-start-k 8 \
  --bad-start-source distance \
  --bad-start-ramp-steps 24 \
  --threshold 0.20 \
  --cluster-cache-file cache/cluster_6156_t020_a4_b2_g1.json \
  --use-real-timing \
  --build-dir /workspace/sparsene/examples/src_fp32/acc/testbed/build_test \
  --target-template acc-plan_{plan_id}-tf32 \
  --make-jobs 48 \
  --run-args "-N 64 -M 1024 -K 1024 -mtx_flag 0" \
  --output-json results/run_sim_cluster_hybrid_constsim_badstart_distance_2.json \
  --output-csv results/run_sim_cluster_hybrid_constsim_badstart_distance_2.csv \
  --output-plot results/run_sim_cluster_hybrid_constsim_badstart_distance_2.png

python plot_sim_vs_nosim_curve.py \
  --sim-json results/run_sim_cluster_hybrid_2.json \
  --nosim-json results/run_sim_cluster_hybrid_constsim_badstart_distance_2.json \
  --sim-label sim \
  --nosim-label no-sim \
  --output-plot results/plot_sim_vs_nosim_curve2.png \
  --output-csv results/plot_sim_vs_nosim_curve2.csv