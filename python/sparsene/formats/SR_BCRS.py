from collections import OrderedDict

from sparsene.format.format import Direction, Format, dense_atomic_format, df_axis, sv_axis

# df_axis: dense fixed-length axis
# sv_axis: sparse variable-length axis
SR_BCRS_FORMAT = Format(
    name="SR-BCRS",
    axes=[
        df_axis("X_o", Direction.ROW),
        sv_axis("Y_s", Direction.COL),
        df_axis("X_i", Direction.ROW),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", Direction.COL),
            df_axis("Y_si", Direction.COL),
            df_axis("X_i", Direction.ROW),
        ],
        child=dense_atomic_format("X_i", "Y_si"),
    ),
)
