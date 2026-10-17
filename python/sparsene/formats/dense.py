from sparsene.format.format import Direction, Format, df_axis
from copy import deepcopy
from sympy import Expr, Symbol, Number

DENSE_FORMAT = Format(
    name="dense",
    axes=[
        df_axis("X", Direction.ROW, length=Symbol("M")),
        df_axis("Y", Direction.COL, length=Symbol("K")),
    ],
)

DENSE_T_FORMAT = Format(
    name="dense_T",
    axes=[
        df_axis("Y", Direction.COL, length=Symbol("K")),
        df_axis("X", Direction.ROW, length=Symbol("M")),
    ],
)


def get_dense_format(direction: Direction = Direction.ROW) -> Format:
    if direction == Direction.ROW:
        return deepcopy(DENSE_FORMAT)
    else:
        return deepcopy(DENSE_T_FORMAT)
