from collections import OrderedDict
from operator import length_hint

from sparsene.format.format import Direction, Format, coo_atomic_format, df_axis, sv_axis, Symbol

# TODO(ZKG):这里使用上下两种格式的时候,生成的RTS不稳定

# HME_TCF_FORMAT = Format(
#     name="HME-TCF",
#     axes=[
#         df_axis("X_o", Direction.ROW, length="M"),
#         sv_axis("Y_s", Direction.COL, length="K"),
#         df_axis("X_i", Direction.ROW, length="BLK_M"),
#     ],
#     child=Format(
#         axes=[
#             df_axis("Y_so", Direction.COL, length=Symbol("K")/Symbol("BLK_K")),
#             df_axis("Y_si", Direction.COL, length="BLK_K"),
#             df_axis("X_i", Direction.ROW, length="BLK_M"),
#         ],
#         child=Format(
#             axes=[
#                 df_axis("X_io", Direction.ROW, length=Symbol("BLK_M")/Symbol("BLK_M_i")),
#                 sv_axis("Y_sis", Direction.COL, length="BLK_K"),
#                 df_axis("X_ii", Direction.ROW, length="BLK_M_i"),
#             ],
#             child=Format(
#                 axes=[
#                     df_axis("Y_siso", Direction.COL, length=Symbol("BLK_K")/Symbol("BLK_K_i")),
#                     df_axis("Y_sisi", Direction.COL, length="BLK_K_i"),
#                     df_axis("X_ii", Direction.ROW, length="BLK_M_i"),
#                 ],
#                 child=coo_atomic_format("X_ii", "Y_sisi"),
#             ),
#         ),
#     ),
# )

HME_TCF_FORMAT = Format(
    name="HME-TCF",
    axes=[
        df_axis("X_o", Direction.ROW, length="M"),
        sv_axis("Y_s", Direction.COL, length="K"),
        df_axis("X_i", Direction.ROW, length="BLK_M"),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", Direction.COL, length=Symbol("K")/Symbol("BLK_K")),
            df_axis("Y_si", Direction.COL, length="BLK_K"),
            df_axis("X_i", Direction.ROW, length="BLK_M"),
        ],
        child=Format(
            axes=[
                sv_axis("Y_sis", Direction.COL, length="BLK_K"),
                df_axis("X_i", Direction.ROW, length="BLK_M"),
            ],
            child=Format(
                axes=[
                    df_axis("X_io", Direction.ROW, length=Symbol("BLK_M")/Symbol("BLK_M_I")),
                    df_axis("Y_sis", Direction.COL, length="BLK_K"),
                    df_axis("X_ii", Direction.ROW, length="BLK_M_I"),
                ],
                child=Format(
                    axes=[
                        df_axis("Y_siso", Direction.COL, length=Symbol("BLK_K")/Symbol("BLK_K_I")),
                        df_axis("Y_sisi", Direction.COL, length="BLK_K_I"),
                        df_axis("X_ii", Direction.ROW, length="BLK_M_I"),
                    ],
                    child=coo_atomic_format("X_ii", "Y_sisi"),
                )
            )
        )
    )
)