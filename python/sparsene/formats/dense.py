from sparsene.model.format import Direction, Format, df_axis
from copy import deepcopy

DENSE_FORMAT = Format(
    name="dense",
    axes=[
        df_axis("X", Direction.ROW, length="m"),
        df_axis("Y", Direction.COL, length="n"),
    ],
)

DENSE_T_FORMAT = Format(
    name="dense_T",
    axes=[
        df_axis("Y", Direction.COL, length="n"),
        df_axis("X", Direction.ROW, length="m"),
    ],
)


def get_dense_format(direction: Direction = Direction.ROW) -> Format:
    if direction == Direction.ROW:
        return deepcopy(DENSE_FORMAT)
    else:
        return deepcopy(DENSE_T_FORMAT)
