from sparsene.model.format import Direction, Format, df_axis
from copy import deepcopy

DENSE_FORMAT = Format(
    name="dense",
    axes=[
        df_axis("X", Direction.ROW),
        df_axis("Y", Direction.COL),
    ],
)

DENSE_T_FORMAT = Format(
    name="dense_T",
    axes=[
        df_axis("Y", Direction.COL),
        df_axis("X", Direction.ROW),
    ],
)


def get_dense_format(direction: Direction = Direction.ROW) -> Format:
    if direction == Direction.ROW:
        return deepcopy(DENSE_FORMAT)
    else:
        return deepcopy(DENSE_T_FORMAT)
