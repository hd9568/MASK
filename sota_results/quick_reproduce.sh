#! T1: kernel
#> Sparsene
# sparsene_fp32_N128.log -> results
# sparsene_fp32_N256.log -> results
# sparsene_fp32_N512.log -> results
cp sparsene_fp32_N128.log ../results/
cp sparsene_fp32_N256.log ../results/
cp sparsene_fp32_N512.log ../results/
#> DTC-SpMM
#TODO: running
# DTCSpMM_exe_time_and_throughput.csv -> results
cp DTCSpMM_exe_time_and_throughput.csv ../results/
#> Acc-SpMM
#TODO: running
# acc-result.csv -> results
cp acc-result.csv ../results/
#> SparseTIR
# sparsetir.log -> SparseTIR-exp/examples/spmm/
cp sparsetir.log ../SparseTIR-exp/examples/spmm/
#> FlashSparse
# spmm_tf32_128.csv -> FlashSparse/result/FlashSparse/spmm
# spmm_tf32_256.csv -> FlashSparse/result/FlashSparse/spmm
# spmm_tf32_512.csv -> FlashSparse/result/FlashSparse/spmm
cp spmm_tf32_128.csv ../FlashSparse/result/FlashSparse/spmm/
cp spmm_tf32_256.csv ../FlashSparse/result/FlashSparse/spmm/
cp spmm_tf32_512.csv ../FlashSparse/result/FlashSparse/spmm/
#> cuSPARSE
# cusparse_fp32_N128.log -> results
# cusparse_fp32_N256.log -> results
# cusparse_fp32_N512.log -> results
cp cusparse_fp32_N128.log ../results/
cp cusparse_fp32_N256.log ../results/
cp cusparse_fp32_N512.log ../results/
#> Sputnik
# rode_spmm_f32_n128.csv -> FlashSparse/result/Baseline/spmm
# rode_spmm_f32_n256.csv -> FlashSparse/result/Baseline/spmm
# rode_spmm_f32_n512.csv -> FlashSparse/result/Baseline/spmm
cp rode_spmm_f32_n128.csv ../FlashSparse/result/Baseline/spmm/
cp rode_spmm_f32_n256.csv ../FlashSparse/result/Baseline/spmm/
cp rode_spmm_f32_n512.csv ../FlashSparse/result/Baseline/spmm/

#! T2: end2end
#> Sparsene
# sc-ad-gcn_e2e_dtc_base_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_dtc_multibind_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_srbcrs_base_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_srbcrs_16x8_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_srbcrs_16x8_multibind_26_128-512.csv -> sparsene/end2end/result
# sc-ad-gcn_e2e_srbcrs_16x8_strict_lb_26_128-512.csv -> sparsene/end2end/result
cp sc-ad-gcn_e2e_dtc_base_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_dtc_multibind_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_dtc_strict_lb_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_srbcrs_base_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_srbcrs_16x8_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_srbcrs_16x8_multibind_26_128-512.csv  ../sparsene/end2end/result
cp sc-ad-gcn_e2e_srbcrs_16x8_strict_lb_26_128-512.csv  ../sparsene/end2end/result

#> FlashSparse
# sc-ad-gcn_e2e_flashsparse_26_128-512.csv -> sparsene/end2end/result
cp sc-ad-gcn_e2e_flashsparse_26_128-512.csv ../sparsene/end2end/result
#> DTC-SpMM
# sc-ad-gcn_e2e_dtc_origin_26_128-512.csv -> sparsene/end2end/result
cp sc-ad-gcn_e2e_dtc_origin_26_128-512.csv ../sparsene/end2end/result
#> DGL
# sc-ad-gcn_e2e_dgl_26_128-512.csv -> sparsene/end2end/result
cp sc-ad-gcn_e2e_dgl_26_128-512.csv ../sparsene/end2end/result
#> PyG
# sc-ad-gcn_e2e_pyg_26_128-512.csv -> sparsene/end2end/result
cp sc-ad-gcn_e2e_pyg_26_128-512.csv ../sparsene/end2end/result

#! T3: load balance
# mip1_dtc_load_balance.log -> results
# mip1_dtc_multi_binding_load_balance.log -> results
# ddi_dtc_load_balance.log -> results
# ddi_dtc_strict_lb_load_balance.log -> results
cp mip1_dtc_load_balance.log ../results/
cp mip1_dtc_multi_binding_load_balance.log ../results/
cp ddi_dtc_load_balance.log ../results/
cp ddi_dtc_strict_lb_load_balance.log ../results/


#! T4: pipeline simulator
# acc_combined_results.json -> sparsene/simulator/results/a100/acc/combined_results.json
# bitbsr_combined_results.json -> sparsene/simulator/results/a100/bitbsr/combined_results.json
cp acc_combined_results.json ../sparsene/simulator/results/a100/acc/combined_results.json
cp bitbsr_combined_results.json ../sparsene/simulator/results/a100/bitbsr/combined_results.json

#! T5: search convergence
# run_sim_cluster_hybrid.json -> sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/results
# run_sim_cluster_hybrid_constsim.json -> sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/results
cp run_sim_cluster_hybrid.json ../sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/results/
cp run_sim_cluster_hybrid_constsim.json ../sparsene/examples/src_fp32/acc/testbed/scripts/motivation_scripts/sc_search_6156plans/results/