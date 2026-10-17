from collections import OrderedDict

from sparsene.format.format import (
    Direction,
    Format,
    ell_atomic_format,
    df_axis,
    sv_axis,
    Symbol,
)

ROW = Direction.ROW
COL = Direction.COL

ELL_TCF_FORMAT = Format(
    name="ELL-TCF",
    axes=[
        df_axis("X_o", ROW, length="M"),
        sv_axis("Y_s", COL, length="K"),
        df_axis("X_i", ROW, length="BLK_M"),
    ],
    child=Format(
        axes=[
            df_axis("Y_so", COL, length=Symbol("K") / Symbol("BLK_K")),  # type: ignore
            df_axis("Y_si", COL, length="BLK_K"),
            df_axis("X_i", ROW, length="BLK_M"),
        ],
        child=ell_atomic_format("X_i", "Y_si"),
    ),
)
