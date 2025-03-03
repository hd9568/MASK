from sparsene.op_gen.rts_parser import computent_from_rts, Computent
from sparsene.model.op import Op, SequentialOp, MetaOp
from sparsene.model.stage import Stage
from typing import List, Optional
from sparsene.op_gen.rts_parser import (
    Schedule,
    DenseAxisIterator,
    SparseAxisIterator,
    SparseAxisSlicer,
    AtomicFormatOp,
    AtomicFormatType,
)


def generate_from_computent(computent: Computent) -> List:
    def _handle_dense_axis_iter(schedule: DenseAxisIterator, op_list: List) -> None:
        pass

    def _handle_sparse_axis_iter(schedule: SparseAxisIterator, op_list: List) -> None:
        op_list.append("[G2R] load soff")
        op_list.append("[G2S] load AtoB")
        op_list.append("[S2R] load AtoB")

    def _handle_sparse_axis_slicer(schedule: SparseAxisSlicer, op_list: List) -> None:
        op_list.append("[G2R] load soff")
        op_list.append("[G2S] load AtoB")
        op_list.append("[S2R] load AtoB")

    def _handle_atomic_format_op(schedule: AtomicFormatOp, op_list: List) -> None:
        match schedule.type:
            case AtomicFormatType.DENSE:
                op_list.append("[G2S] load A")
                op_list.append("[S2R] load A")
                op_list.append("[G2S] load B")
                op_list.append("[S2R] load B")
                op_list.append("mma")
            case AtomicFormatType.COO:
                op_list.append("[G2R] load A coo_off")
                op_list.append("[G2R] load A coo_idx")
                op_list.append("[R2S] zero A buffer")
                op_list.append("[G2R] extract A coo_val")
                op_list.append("[S2R] load A coo_val")
                op_list.append("[G2S] load B")
                op_list.append("[S2R] load B")
                op_list.append("mma")
            case AtomicFormatType.MCO:
                op_list.append("load soff")

    def _handle_epilogue(op_list: List) -> None:
        op_list.append("store C")

    def _handle(schedule: Optional[Schedule], op_list: List) -> None:
        if schedule is None:
            _handle_epilogue(op_list)
            return
        elif isinstance(schedule, DenseAxisIterator):
            _handle_dense_axis_iter(schedule, op_list)
        elif isinstance(schedule, SparseAxisIterator):
            _handle_sparse_axis_iter(schedule, op_list)
        elif isinstance(schedule, SparseAxisSlicer):
            _handle_sparse_axis_slicer(schedule, op_list)
        elif isinstance(schedule, AtomicFormatOp):
            _handle_atomic_format_op(schedule, op_list)
        else:
            raise ValueError(
                f"Unknown schedule type: {type(schedule)}. schedule: {schedule}"
            )

        _handle(schedule.get_child(), op_list)

    # TODO: IMPLEMENT
    root_schedule = computent.schedule
    op_list = []
    while root_schedule is not None:
        op_list.append(root_schedule)
        root_schedule = root_schedule.get_child()
    return op_list
