from collections import OrderedDict

from sparsene.format.format import Direction, Format, dense_atomic_format, df_axis, sv_axis

ROW_REORDER_SR_BCRS_FORMAT = Format(
    name="ROW-REORDER-SR-BCRS",
    axes=[
        sv_axis("X_s", Direction.ROW),
        df_axis("Y", Direction.COL),
    ],
    child=Format(
        axes=[
            df_axis("X_so", Direction.ROW),
            df_axis("Y_s", Direction.COL),
            df_axis("X_si", Direction.ROW),
        ],
        child=Format(
            axes=[
                df_axis("Y_so", Direction.COL),
                df_axis("Y_si", Direction.COL),
                df_axis("X_i", Direction.ROW),
            ],
            child=dense_atomic_format("X_i", "Y_si"),
        ),
    ),
)

# ROW_REORDER_SR_BCRS_FORMAT = Format(
#     name="ROW-REORDER-SR-BCRS",
#     axes=[
#         df_axis("X_o", Direction.ROW),
#         df_axis("Y", Direction.COL),
#         df_axis("X_i", Direction.ROW),
#     ],
#     child=Format(
#         axes=[
#             sv_axis("Y_s", Direction.COL),
#             df_axis("X_i", Direction.ROW),
#         ],
#         child=Format(
#             axes=[
#                 df_axis("Y_so", Direction.COL),
#                 df_axis("Y_si", Direction.COL),
#                 df_axis("X_i", Direction.ROW),
#             ],
#             child=dense_atomic_format("X_i", "Y_si"),
#         ),
#     ),
# )


# ROW_REORDER_SR_BCRS_FORMAT = Format(
#     name="ROW-REORDER-SR-BCRS",
#     axes=[
#         df_axis("X_o", Direction.ROW),
#         sv_axis("Y_s", Direction.COL),
#         df_axis("X_i", Direction.ROW),
#     ],
#     child=Format(
#         axes=[
#             df_axis("Y_so", Direction.COL),
#             df_axis("Y_si", Direction.COL),
#             df_axis("X_i", Direction.ROW),
#         ],
#         child=dense_atomic_format("X_i", "Y_si"),
#     ),
# )