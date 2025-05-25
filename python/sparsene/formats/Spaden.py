from collections import OrderedDict

from sparsene.format.format import Direction, Format, mco_atomic_format, df_axis, sv_axis

BIT_BSR_FORMAT = Format(
    name="BIT-BSR",
    axes=[
        df_axis("X_o", Direction.ROW),
        sv_axis("Y_o_s", Direction.COL),
        df_axis("X_i", Direction.ROW),
        df_axis("Y_i", Direction.COL),
    ],
    child=mco_atomic_format("X_i", "Y_i"),
)
