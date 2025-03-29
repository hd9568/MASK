from sparsene.format.format import Direction, Format, df_axis
from copy import deepcopy

DENSE_FORMAT = Format(
    name="dense",
    axes=[
        df_axis("X", Direction.ROW, length="M"),
        df_axis("Y", Direction.COL, length="K"),
    ],
)

DENSE_T_FORMAT = Format(
    name="dense_T",
    axes=[
        df_axis("Y", Direction.COL, length="K"),
        df_axis("X", Direction.ROW, length="M"),
    ],
)


def get_dense_format(direction: Direction = Direction.ROW) -> Format:
    if direction == Direction.ROW:
        return deepcopy(DENSE_FORMAT)
    else:
        return deepcopy(DENSE_T_FORMAT)
