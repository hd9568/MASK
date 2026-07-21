SUMMARY_MD="/home/huoda/sparsene/.tc_block_k_sweep_20260715_124737/sweep_summary.md"
BEST_CSV="/home/huoda/sparsene/.tc_block_k_sweep_20260715_124737/best_per_dataset.csv"
ok_count=0
fail_count=0
total_speedup=0.0

while IFS=',' read -r ds M K_dim N_dim best_k best_time best_cu best_speedup; do
    [[ "$ds" == "dataset" ]] && continue
    if [[ "$best_k" == "0" || "$best_k" == "" ]]; then
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | — | — | — | — |" >> "${SUMMARY_MD}"
        fail_count=$((fail_count + 1))
    else
        echo "| ${ds} | ${M} | ${K_dim} | ${N_dim} | **${best_k}** | ${best_time} | ${best_cu} | ${best_speedup} |" >> "${SUMMARY_MD}"
        ok_count=$((ok_count + 1))
        total_speedup="$(python3 -c "print(${total_speedup} + ${best_speedup})" 2>/dev/null || echo "${total_speedup}")"
    fi
done < "${BEST_CSV}"

if [[ "$ok_count" -gt 0 ]]; then
    avg_speedup="$(python3 -c "print(round(float(${total_speedup}) / ${ok_count}, 4))" 2>/dev/null || echo "N/A")"
    beating_count=0
    beating_speedup_sum=0.0
    while IFS=',' read -r ds M K_dim N_dim best_k best_time best_cu best_speedup; do
        [[ "$ds" == "dataset" ]] && continue
        if [[ "$best_k" != "0" && "$best_k" != "" ]]; then
            if python3 -c "exit(0 if float('${best_speedup}') > 1.0 else 1)" 2>/dev/null; then
                beating_count=$((beating_count + 1))
                beating_speedup_sum="$(python3 -c "print(${beating_speedup_sum} + ${best_speedup})" 2>/dev/null || echo "${beating_speedup_sum}")"
            fi
        fi
    done < "${BEST_CSV}"
    beating_avg=""
    if [[ "$beating_count" -gt 0 ]]; then
        beating_avg="$(python3 -c "print(round(float(${beating_speedup_sum}) / ${beating_count}, 4))" 2>/dev/null || echo "N/A")"
    fi
    {
        echo ""
        echo "**Summary**: ${ok_count}/${#DATASET_PATHS[@]} datasets OK"
        echo "- Average best speedup: ${avg_speedup}x"
        echo ""
        echo "> **统计**（共 ${ok_count} 个数据集）："
        echo "> - speedup_vs_cusparse 平均值：**${avg_speedup}**"
        if [[ -n "$beating_avg" && "$beating_count" -gt 0 ]]; then
            echo "> - 超过 cusparse 的数据集：**${beating_count}** 个，平均 speedup：**${beating_avg}**"
        else
            echo "> - 超过 cusparse 的数据集：**0** 个"
        fi
    } >> "${SUMMARY_MD}"
fi
if [[ "$fail_count" -gt 0 ]]; then
    echo "- Failed: ${fail_count} datasets" >> "${SUMMARY_MD}"
fi

# ── 7. Done ───────────────────────────────────────────────────────────────
# log "============================================================"
# log "Sweep complete."
# log "  Per-dataset best:   ${BEST_CSV}"
# log "  Full raw data:      ${CSV}"
# log "  Markdown summary:   ${SUMMARY_MD}"
# log "  Per-dataset logs:   ${RESULT_ROOT}/logs/<dataset>/"
# log "============================================================"

# cat "${BEST_CSV}"
