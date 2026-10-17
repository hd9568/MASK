"""Generated from /home/huoda/sparsene/logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/formats/dc2.format.json; do not edit the embedded agent proposal."""

from sparsene.formats.dsl import build_format_from_spec

FORMAT_SPEC = {'dataset': 'dc2',
 'format_name': 'BIT-BSR-MCO-16x16',
 'levels': [{'level': 1,
             'name': 'active_fixed_blocks',
             'axes': [{'name': 'X_o',
                       'type': 'df',
                       'direction': 'ROW',
                       'length': 'M',
                       'reorder': None,
                       'purpose': '按 BLK_M 对行域分块行'},
                      {'name': 'Y_o_s',
                       'type': 'sv',
                       'direction': 'COL',
                       'length': 'K',
                       'reorder': None,
                       'purpose': '每个块行中活跃列块的选择；变长元数据记录实际活跃列块索引'},
                      {'name': 'X_i',
                       'type': 'df',
                       'direction': 'ROW',
                       'length': 'BLK_M',
                       'reorder': None,
                       'purpose': '块内固定行坐标'},
                      {'name': 'Y_i',
                       'type': 'df',
                       'direction': 'COL',
                       'length': 'BLK_K',
                       'reorder': None,
                       'purpose': '块内固定列坐标'}]}],
 'atomic': {'type': 'mco',
            'x_axis': 'X_i',
            'y_axis': 'Y_i',
            'layout': 'row_major',
            'rationale': '16×16 块内 occupancy 不超过 110，Gini 0.296，mask 存储最小 (5.84 MB)，bitmask 比 COO '
                         '坐标节省 2.9 MB，restore 仅需按 mask 位 scatter values，无全局列号映射。',
            'custom_spec': None},
 'params': {'BLK_M': 16, 'BLK_K': 16},
 'rationale': '行列极不平衡、大量空列窗，MCO 结合直接活跃列块选择避免了凝聚列数组的间接映射和额外的全局列号加载，kernel 执行路径更短更规整，在现有特征证据下预计具有最低的 '
              'kernel 内执行时间。',
 'provenance': {'run_id': '20260718_165146_deepseek_v4_pro_format26_v6',
                'source': '/home/huoda/sparsene/logs/e2e_runs/20260718_165146_deepseek_v4_pro_format26_v6/formats/dc2.format.json',
                'schema_version': 'sparsene.format_design.v6',
                'request_id': '774384f4-df13-48db-9f72-a917b3d57d48'}}
FORMAT = build_format_from_spec(FORMAT_SPEC)
PARAMS = FORMAT_SPEC["params"]
PROVENANCE = FORMAT_SPEC["provenance"]
