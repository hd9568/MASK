from collections import OrderedDict

from sparsene.format.format import Direction, Format, dense_atomic_format, df_axis, sv_axis, Symbol

SR_BCRS_FORMAT = Format(
    name="SR-BCRS",
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
        child=dense_atomic_format("X_i", "Y_si"),
    ),
)
