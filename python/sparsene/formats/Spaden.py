from collections import OrderedDict

from sparsene.format.format import Direction, Format, mco_atomic_format, df_axis, sv_axis

BIT_BSR_FORMAT = Format(
    name="BIT-BSR",
    axes=[
        df_axis("X_o", Direction.ROW, length="M"),
        sv_axis("Y_o_s", Direction.COL, length="K"),
        df_axis("X_i", Direction.ROW, length="BLK_M"),
        df_axis("Y_i", Direction.COL, length="BLK_K"),
    ],
    child=mco_atomic_format("X_i", "Y_i"),
)
