from sparsene.model.format import Direction, Axis, Format
from sparsene.formats.DTC_SpMM import OP_S, OP_R, OP_MMA, OP_WB

dtc_spmm_fmt = Format(
    name="DTC-SpMM",
    axes=[
        Axis(
            name="X_o",
            direction=Direction.ROW,
            is_sparse=False,
            is_varlen=False,
        ),
        Axis(
            name="Y_s",
            direction=Direction.COL,
            is_sparse=True,
            is_varlen=True,
        ),
        Axis(
            name="X_i",
            direction=Direction.ROW,
            is_sparse=False,
            is_varlen=False,
        ),
    ],
    child=(
        (1, 2),
        Format(
            axes=[
                Axis(
                    name="Y_so",
                    direction=Direction.ROW,
                    is_sparse=False,
                    is_varlen=False,
                ),
                Axis(
                    name="Y_si",
                    direction=Direction.ROW,
                    is_sparse=False,
                    is_varlen=False,
                ),
                Axis(
                    name="X_i",
                    direction=Direction.ROW,
                    is_sparse=False,
                    is_varlen=False,
                ),
            ],
            child=(
                (1, 2),
                Format(
                    axes=[
                        Axis(
                            name="X",
                            direction=Direction.ROW,
                            is_sparse=True,
                            is_varlen=True,
                        ),
                        Axis(
                            name="Y",
                            direction=Direction.COL,
                            is_sparse=True,
                            is_varlen=True,
                        ),
                    ],
                    child=None,
                    impl="coo",
                ),
            ),
        ),
    ),
)

if __name__ == "__main__":
    shifts = [2, 1]
    reg_ids = [0, 1, 2]
    print("=" * 50)
    print(OP_S.gen_func(shifts))
    print("=" * 50)
    print(OP_R.gen_func(shifts))
    print("=" * 50)
    print(OP_MMA.gen_func(shifts))
    print("=" * 50)
    print(OP_WB.gen_func(shifts))
    print("=" * 50)
    print(OP_S.gen_func_call(shifts, reg_ids,
                             lane_id=0,
                             block_idx_x=1,
                             i=2,
                             TCStart_offset="TCStart_offset",
                             tileA_size="tileA_size",
                             A_TCFValue="A_TCFValue",
                             A_TCOffset="A_TCOffset",
                             A_TCFColIdx="A_TCFColIdx",
                             X_Val="X_Val",
                             Y_Val="Y_Val",
                             tileA_block="tileA_block",
                             sparse_index="sparse_index",
                             A="A",
                             B="B",
                             C="nullptr", ))
    print("=" * 50)

    print(dtc_spmm_fmt)
