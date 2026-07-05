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
        """
        把相邻相同方向的 Axis 合并成一个。
        在不能直接合并时，通过 swap 把能合并的轴移到一起。
        """
        while len(axes) > 2:
            merged = False
            # 倒数第二个轴（候选 merge 目标）
            last = axes[-2]
            prev = axes[-3]

            if last.direction == prev.direction:
                # ✅ 直接合并
                seq.prepend(SplitTransformation(len(axes) - 3, last.length))
                axes.pop(-2)
                merged = True
            else:
                # ❌ 不行: swap 最后两个轴
                seq.prepend(SwapTransformation(len(axes) - 2, len(axes) - 1))
                axes[-1], axes[-2] = axes[-2], axes[-1]
                # 再下一轮继续尝试 merge

            if not merged:
                # 如果 swap 后还不能 merge，下轮继续循环
                continue


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
