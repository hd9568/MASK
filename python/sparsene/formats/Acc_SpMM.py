from collections import OrderedDict

from sparsene.format.format import Direction, Format, mco_atomic_format, df_axis, sv_axis

BIT_TCF_FORMAT = Format(
    name="DTC-SpMM",
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
        child=mco_atomic_format("X_i", "Y_si"),
    ),
)
