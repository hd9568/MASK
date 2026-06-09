我直接跑两轮多数据集 sweep：一轮 `generated`，一轮 `reference`，沿用现有无 checkpoint、纯 GPU 计时口径，然后把两份汇总表直接整理给你。
toolName: run_command
            
status: success
          
command: MODE=generated RESULT_ROOT=/mnt/lustre-client/zhangkaige/sparsene_perf_logs SPARSENE_DISABLE_MTX_CACHE=1 CUDA_ARCH=sm_80 bash scripts/benchmark_inc_multi_dataset.sh undefined
          
`generated` 全量已经跑完了。我接着跑 `reference` 全量，用同一批数据和同一计时口径，确保两份表可以直接对照。
        
            
toolName: run_command
            
status: success
          
command: MODE=reference RESULT_ROOT=/mnt/lustre-client/zhangkaige/sparsene_perf_logs SPARSENE_DISABLE_MTX_CACHE=1 CUDA_ARCH=sm_80 bash scripts/benchmark_inc_multi_dataset.sh undefined
          
两轮 sweep 都完成了。我现在直接读取 `summary.csv/summary.md`，把 `generated` 和 `reference` 两份表按你要的形式整理出来。
        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.md
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.md
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.csv
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.csv
          
我再补一组简短汇总指标，方便你快速判断两份表的整体差距，比如中位 `speedup_vs_cusparse` 和跑赢 cuSPARSE 的数据集个数。
        
            
toolName: run_command
            
status: success
          
command: python - <<'PY'
import csv, statistics
paths = {
 'generated':'/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.csv',
 'reference':'/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.csv',
}
for name,path in paths.items():
    rows=list(csv.DictReader(open(path)))
    speeds=[float(r['speedup_vs_cusparse']) for r in rows if r['status']=='ok' and r['speedup_vs_cusparse']]
    my=[float(r['mykernel_ms']) for r in rows if r['status']=='ok' and r['mykernel_ms']]
    faster=sum(1 for s in speeds if s>1.0)
    print(name)
    print('ok', len(speeds))
    print('median_speedup', f'{statistics.median(speeds):.4f}')
    print('mean_speedup', f'{statistics.mean(speeds):.4f}')
    print('faster_than_cusparse', faster)
    best=max(rows, key=lambda r: float(r['speedup_vs_cusparse']))
    worst=min(rows, key=lambda r: float(r['speedup_vs_cusparse']))
    print('best', best['dataset'], best['speedup_vs_cusparse'])
    print('worst', worst['dataset'], worst['speedup_vs_cusparse'])
    print()
PY undefined
          

        
            
toolName: todo_write
            
status: success
          
Todos updated: 20 items
          
**结果文件**
- `generated` 汇总：[/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.csv](file:///mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.csv) 和 [summary.md](file:///mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_generated_20260512_005036/summary.md)
- `reference` 汇总：[/mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.csv](file:///mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.csv) 和 [summary.md](file:///mnt/lustre-client/zhangkaige/sparsene_perf_logs/multi_dataset_reference_20260512_005854/summary.md)

**快速结论**
- 两轮都是 `26/26` 数据集成功，无 correctness error。
- `generated` 中位 `speedup_vs_cusparse = 0.6708x`，共有 `8/26` 个数据集快于 cuSPARSE。
- `reference` 中位 `speedup_vs_cusparse = 1.0964x`，共有 `16/26` 个数据集快于 cuSPARSE。
- 当前这版 `generated` 相比之前已经明显进步，但全量上还没有整体追平 `reference`。

**Generated 表**
| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |
|---|---:|---:|---:|---:|---:|---:|---|
| ASIC_680k.mtx | 682862 | 682862 | 512 | 94.085215 | 4.954757 | 0.0527 | ok |
| DD.mtx | 334925 | 334925 | 512 | 2.486815 | 1.604117 | 0.6450 | ok |
| Ga41As41H72.mtx | 268096 | 268096 | 512 | 13.577523 | 10.889072 | 0.8020 | ok |
| OVCAR-8H.mtx | 1889542 | 1889542 | 512 | 12.327126 | 7.755602 | 0.6291 | ok |
| Si41Ge41H72.mtx | 185639 | 185639 | 512 | 10.278748 | 7.996918 | 0.7780 | ok |
| Yeast.mtx | 1710902 | 1710902 | 512 | 8.168632 | 7.193989 | 0.8807 | ok |
| YeastH.mtx | 3138114 | 3138114 | 512 | 20.000870 | 12.838380 | 0.6419 | ok |
| cant.mtx | 62451 | 62451 | 512 | 1.343089 | 1.874104 | 1.3954 | ok |
| conf5_4-8x8-10.mtx | 49152 | 49152 | 512 | 0.868925 | 1.034363 | 1.1904 | ok |
| consph.mtx | 83334 | 83334 | 512 | 1.903217 | 3.076188 | 1.6163 | ok |
| cop20k_A.mtx | 121192 | 121192 | 512 | 2.108191 | 3.261010 | 1.5468 | ok |
| dc2.mtx | 116835 | 116835 | 512 | 26.892903 | 0.800051 | 0.0297 | ok |
| ddi.mtx | 4267 | 4267 | 512 | 1.438976 | 0.895826 | 0.6225 | ok |
| eu-2005.mtx | 862664 | 862664 | 512 | 12.970045 | 9.439693 | 0.7278 | ok |
| mac_econ_fwd500.mtx | 206500 | 206500 | 512 | 1.942047 | 1.099182 | 0.5660 | ok |
| mc2depi.mtx | 525825 | 525825 | 512 | 4.352788 | 2.431939 | 0.5587 | ok |
| mip1.mtx | 66463 | 66463 | 512 | 17.212498 | 4.289802 | 0.2492 | ok |
| pdb1HYS.mtx | 36417 | 36417 | 512 | 0.980408 | 2.073109 | 2.1145 | ok |
| protein.mtx | 132534 | 132534 | 512 | 67.132773 | 38.868745 | 0.5790 | ok |
| pwtk.mtx | 217918 | 217918 | 512 | 2.857175 | 5.639854 | 1.9739 | ok |
| reddit.mtx | 232965 | 232965 | 512 | 116.981846 | 67.972905 | 0.5811 | ok |
| rma10.mtx | 46835 | 46835 | 512 | 0.751811 | 1.190666 | 1.5837 | ok |
| scircuit.mtx | 170998 | 170998 | 512 | 1.503089 | 0.982272 | 0.6535 | ok |
| shipsec1.mtx | 140874 | 140874 | 512 | 1.998735 | 4.136448 | 2.0695 | ok |
| web-BerkStan.mtx | 685230 | 685230 | 512 | 6.299433 | 4.335298 | 0.6882 | ok |
| webbase-1M.mtx | 1000005 | 1000005 | 512 | 7.205458 | 4.283023 | 0.5944 | ok |

> **统计**（共 26 个数据集）：
> - speedup_vs_cusparse 平均值：**0.9142**
> - 超过 cusparse 的数据集：**8** 个，平均 speedup：**1.6863**

**Reference 表**
| dataset | M | K | N | mykernel_ms | cusparse_ms | speedup_vs_cusparse | status |
|---|---:|---:|---:|---:|---:|---:|---|
| ASIC_680k.mtx | 682862 | 682862 | 512 | 49.317622 | 4.954901 | 0.1005 | ok |
| DD.mtx | 334925 | 334925 | 512 | 1.775473 | 1.605867 | 0.9045 | ok |
| Ga41As41H72.mtx | 268096 | 268096 | 512 | 7.479675 | 10.346609 | 1.3833 | ok |
| OVCAR-8H.mtx | 1889542 | 1889542 | 512 | 8.979600 | 7.755735 | 0.8637 | ok |
| Si41Ge41H72.mtx | 185639 | 185639 | 512 | 5.604761 | 7.994645 | 1.4264 | ok |
| Yeast.mtx | 1710902 | 1710902 | 512 | 7.906652 | 7.193497 | 0.9098 | ok |
| YeastH.mtx | 3138114 | 3138114 | 512 | 14.842245 | 12.838390 | 0.8650 | ok |
| cant.mtx | 62451 | 62451 | 512 | 0.778824 | 1.876787 | 2.4098 | ok |
| conf5_4-8x8-10.mtx | 49152 | 49152 | 512 | 0.572027 | 1.035960 | 1.8110 | ok |
| consph.mtx | 83334 | 83334 | 512 | 1.099735 | 3.075482 | 2.7966 | ok |
| cop20k_A.mtx | 121192 | 121192 | 512 | 1.186970 | 3.824732 | 3.2223 | ok |
| dc2.mtx | 116835 | 116835 | 512 | 13.540465 | 0.792781 | 0.0585 | ok |
| ddi.mtx | 4267 | 4267 | 512 | 0.701379 | 0.902349 | 1.2865 | ok |
| eu-2005.mtx | 862664 | 862664 | 512 | 7.359990 | 9.459261 | 1.2852 | ok |
| mac_econ_fwd500.mtx | 206500 | 206500 | 512 | 1.179412 | 1.099418 | 0.9322 | ok |
| mc2depi.mtx | 525825 | 525825 | 512 | 4.297216 | 2.432379 | 0.5660 | ok |
| mip1.mtx | 66463 | 66463 | 512 | 8.332667 | 4.303217 | 0.5164 | ok |
| pdb1HYS.mtx | 36417 | 36417 | 512 | 0.541051 | 2.081004 | 3.8462 | ok |
| protein.mtx | 132534 | 132534 | 512 | 37.638606 | 38.897910 | 1.0335 | ok |
| pwtk.mtx | 217918 | 217918 | 512 | 1.896182 | 5.643305 | 2.9761 | ok |
| reddit.mtx | 232965 | 232965 | 512 | 60.753501 | 67.925215 | 1.1180 | ok |
| rma10.mtx | 46835 | 46835 | 512 | 0.421263 | 1.194168 | 2.8347 | ok |
| scircuit.mtx | 170998 | 170998 | 512 | 0.940677 | 0.980849 | 1.0427 | ok |
| shipsec1.mtx | 140874 | 140874 | 512 | 1.544243 | 4.003021 | 2.5922 | ok |
| web-BerkStan.mtx | 685230 | 685230 | 512 | 3.937393 | 4.232499 | 1.0749 | ok |
| webbase-1M.mtx | 1000005 | 1000005 | 512 | 6.635643 | 4.284232 | 0.6456 | ok |

> **统计**（共 26 个数据集）：
> - speedup_vs_cusparse 平均值：**1.4808**
> - 超过 cusparse 的数据集：**16** 个，平均 speedup：**2.0087**

**补充观察**
- `generated` 优势最明显的数据集是 `pdb1HYS.mtx`、`shipsec1.mtx`、`pwtk.mtx`，都明显快于 cuSPARSE。
- `generated` 目前最差的是 `dc2.mtx` 和 `ASIC_680k.mtx`，这两个点与 `reference` 也都明显偏慢，说明不完全是 generated 独有问题。
- `reference` 整体仍然更稳，尤其在 `cop20k_A.mtx`、`pwtk.mtx`、`rma10.mtx`、`consph.mtx` 上优势更明显。

如果你要，我下一步可以直接把这两份表再合并成一张 `generated vs reference vs cusparse` 对照总表，方便你直接写报告。