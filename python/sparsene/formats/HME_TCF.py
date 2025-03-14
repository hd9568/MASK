from collections import OrderedDict

from sparsene.format.format import Direction, Format, coo_atomic_format, df_axis, sv_axis

# df_axis: dense fixed-length axis
# sv_axis: sparse variable-length axis
HME_TCF_FORMAT = Format(
    name="HME-TCF",
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
        child=Format(
            axes=[
                df_axis("X_io", Direction.ROW),
                sv_axis("Y_sis", Direction.COL),
                df_axis("X_ii", Direction.ROW),
            ],
            child=Format(
                axes=[
                    df_axis("Y_siso", Direction.COL),
                    df_axis("Y_sisi", Direction.COL),
                    df_axis("X_ii", Direction.ROW),
                ],
                child=coo_atomic_format("X_ii", "Y_sisi"),
            ),
        ),
    ),
)
