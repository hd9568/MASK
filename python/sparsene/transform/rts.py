from typing import List
from collections import OrderedDict
from sparsene.model.format import (
    Direction,
    Format,
    AtomicFormat,
    df_axis,
    sv_axis,
    Axis,
)
from sparsene.model.op import Op
from sparsene.model.param import Params
from sparsene.model.transformation import (
    TransformationSequence,
    GoUpTransformation,
    MergeTransformation,
    SplitTransformation,
    SpawnTransformation,
    SwapTransformation,
    SparsifyTransformation,
    DensifyTransformation,
    CooizeTransformation,
    McoizeTransformation,
    Transformation,
)

from sparsene.logging import get_logger

logger = get_logger("rts")


def print_axes(axes: List[Axis]) -> None:
    for axis in axes:
        logger.debug(axis)


def derive_rts(format: Format) -> TransformationSequence:
    seq = TransformationSequence()

    def _bubble_up(axes: List[Axis], parent_major_direction: Direction) -> None:
        if len(axes) == 0:
            return

        minor_direction_axis_count = 0
        bubble_seq: List[Transformation] = []
        for i in range(len(axes)):
            if axes[i].direction == parent_major_direction:
                for j in range(minor_direction_axis_count, i):
                    bubble_seq.append(SwapTransformation(j, j + 1))
                minor_direction_axis_count += 1

        for trans in reversed(bubble_seq):
            seq.prepend(trans)
            i, j = trans.axes
            axes[i], axes[j] = axes[j], axes[i]

    def _densify(axes: List[Axis]) -> None:
        densify_seq: List[Transformation] = []
        for i in range(len(axes)):
            if axes[i].is_sparse:
                densify_seq.append(SparsifyTransformation(i))

        for trans in reversed(densify_seq):
            seq.prepend(trans)
            axes[trans.axes[0]].is_sparse = False

    def _squeeze(axes: List[Axis]) -> None:
        for i in reversed(range(len(axes) - 1)):
            if axes[i].direction == axes[i + 1].direction:
                seq.prepend(SplitTransformation(i, axes[i + 1].length))
                axes.pop(i)

    def _derive_rts(format: Format, parent_major_direction: Direction) -> None:
        if format.child is not None:
            if isinstance(format.child, AtomicFormat):
                match format.child.atomic_type:
                    case "coo":
                        seq.prepend(CooizeTransformation())
                    case "mco":
                        seq.prepend(McoizeTransformation())
                    case "dense":
                        pass
                    case _:
                        raise ValueError(
                            f"Unknown atomic type: {format.child.atomic_type}"
                        )
                seq.prepend(SpawnTransformation())
            elif isinstance(format.child, Format):
                _derive_rts(format.child, format.axes[-2].direction)
                seq.prepend(SpawnTransformation())

        axes = format.axes.copy()

        # densify
        _densify(axes)
        logger.debug("densify:")
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)
        # bubble up
        _bubble_up(axes, parent_major_direction)
        logger.debug("bubble up:")
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)
        # squeeze
        _squeeze(axes)
        logger.debug("squeeze:")
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)

    _derive_rts(format, Direction.ROW)

    return seq
