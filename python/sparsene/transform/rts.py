from typing import List
from sparsene.format.format import (
    Direction,
    Format,
    AtomicFormat,
    Axis,
)
from sparsene.transform.transformation import (
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
    CsrizeTransformation,
    EllizeTransformation,
    DiaizeTransformation,
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

    def _squeeze_new(axes: List[Axis]) -> None:
        """Reduce one level to a row/column axis pair with guaranteed progress.

        A split is the inverse of merging two adjacent axes with the same
        direction.  If no such pair exists, the directions must alternate, so
        swapping the final *different-direction* pair creates an adjacent
        merge candidate on the next step.  The previous implementation always
        inspected ``axes[-3:-1]`` but swapped ``axes[-2:]``; for a layout such
        as ROW, ROW, COL, COL that swaps two COL axes, leaves the state
        unchanged, and grows ``seq`` forever.
        """
        initial_axis_count = len(axes)
        max_steps = max(1, 2 * max(0, initial_axis_count - 2))
        seen_states: set[tuple[Direction, ...]] = set()
        steps = 0

        while len(axes) > 2:
            state = tuple(axis.direction for axis in axes)
            if state in seen_states:
                rendered = ",".join(direction.name for direction in state)
                raise RuntimeError(
                    f"RTS axis squeeze made no progress for directions [{rendered}]"
                )
            seen_states.add(state)
            steps += 1
            if steps > max_steps:
                rendered = ",".join(axis.direction.name for axis in axes)
                raise RuntimeError(
                    "RTS axis squeeze exceeded its deterministic step budget "
                    f"({max_steps}) for directions [{rendered}]"
                )

            merge_index = next(
                (
                    index
                    for index in reversed(range(len(axes) - 1))
                    if axes[index].direction == axes[index + 1].direction
                ),
                None,
            )
            if merge_index is not None:
                seq.prepend(
                    SplitTransformation(merge_index, axes[merge_index + 1].length)
                )
                axes.pop(merge_index)
                continue

            # With only ROW/COL directions, no adjacent equal pair means the
            # sequence alternates.  The final pair is therefore different and
            # swapping it necessarily creates a mergeable neighbor.
            swap_i = len(axes) - 2
            swap_j = len(axes) - 1
            if axes[swap_i].direction == axes[swap_j].direction:
                raise RuntimeError(
                    "RTS axis squeeze could neither merge nor find a productive swap"
                )
            seq.prepend(SwapTransformation(swap_i, swap_j))
            axes[swap_i], axes[swap_j] = axes[swap_j], axes[swap_i]


    def _derive_rts(format: Format, parent_major_direction: Direction) -> None:
        if format.child is not None:
            if isinstance(format.child, AtomicFormat):
                match format.child.atomic_type:
                    case "coo":
                        seq.prepend(CooizeTransformation())
                    case "mco":
                        seq.prepend(McoizeTransformation())
                    case "csr":
                        seq.prepend(CsrizeTransformation())
                    case "ell":
                        seq.prepend(EllizeTransformation())
                    case "dia":
                        seq.prepend(DiaizeTransformation())
                    case "dense":
                        pass
                    case _:
                        raise ValueError(
                            f"Unknown atomic type: {format.child.atomic_type}"
                        )
                if format.child.axes[1].direction == parent_major_direction:
                    seq.prepend(SwapTransformation(0, 1))
                seq.prepend(SpawnTransformation())
            elif isinstance(format.child, Format):
                _derive_rts(format.child, format.axes[-2].direction)
                seq.prepend(SpawnTransformation())

        axes = format.axes.copy()
        logger.debug("begin print axes")
        print_axes(axes)
        logger.debug("end print axes")

        #! densify
        _densify(axes)
        logger.debug("=" * 10 + "densify:" + "=" * 10)
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)
        #! squeese new
        _squeeze_new(axes)
        logger.debug("=" * 10 + "squeeze:" + "=" * 10)
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)
        #! bubble up
        _bubble_up(axes, parent_major_direction)
        logger.debug("=" * 10 + "bubble up:" + "=" * 10)
        logger.debug(f"ts:\n{seq}")
        print_axes(axes)
        #! squeeze
        # _squeeze(axes)
        # logger.debug("=" * 10 + "squeeze:" + "=" * 10)
        # logger.debug(f"ts:\n{seq}")
        # print_axes(axes)

    _derive_rts(format, Direction.ROW)

    return seq
