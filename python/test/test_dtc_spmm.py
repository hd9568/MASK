import python.sparsene.nvir.op
from sparsene.format import stage
from sparsene.formats.DTC_SpMM import OP_S, OP_R, OP_MMA, OP_WB, DTC_SPMM_FORMAT

dtc_spmm_fmt = DTC_SPMM_FORMAT

if __name__ == "__main__":
    shifts = [2, 1]
    num_res_groups = max(shifts) + 1
    reg_ids = [0, 1, 2]
    num_stages = 2
    filename = "./pipeline_generated.cuh"
    with open(filename, "w") as f:
        f.write(
            "#pragma once\n"
            "#include <cstdint>\n"
            "#include <cuda.h>\n"
            "#include <cuda_fp16.h>\n"
            "#include <cuda_pipeline_primitives.h>\n\n"
        )
        f.write(OP_S.gen_func_def(num_stages) + "\n")
        f.write(OP_R.gen_func_def(num_stages) + "\n")
        f.write(OP_MMA.gen_func_def(num_stages) + "\n")
        f.write(OP_WB.gen_func_def(num_stages) + "\n")
        f.write(
            OP_S.gen_func_call(
                shifts,
                num_res_groups,
                reg_ids,
                wait_num=0,
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
                C="nullptr",
            )
            + "\n"
        )

        op_seq = python.sparsene.nvir.op.SequentialOp(
            name="seq", type="k", ops=[OP_S, OP_R, OP_MMA]
        )
        monolithic_stage = stage.Stage(name="one-stage", seq=op_seq)
        f.write("=" * 50 + "\n")
        f.write(monolithic_stage.gen_kernel(num_res_groups))
        f.write("=" * 50 + "\n")
        f.write(str(dtc_spmm_fmt))
