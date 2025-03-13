#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

template <int _v_shift0, int _v_shift1, int _v_num_res_groups, int _v_r0, int _v_r1, int _v_r2, int _v_wait_num>
__device__ void s(
  int32_t _H_block_idx_x,
  int32_t _H_lane_id,
  int32_t _I_i,
  int32_t _I_TCStart_offset,
  int32_t _C_tileA_size,
  half _G_A_TCFValue,
  int32_t _G_A_TCOffset,
  int32_t _G_A_TCFColIdx,
  half _G_X_Val,
  half _G_Y_Val,
  half _S_tileA_block,
  int32_t _S_sparse_index,
  uint32_t _R_A,
  half _R_B,
  float _R_C
) {
    const int smem_st_ptr = _I_i % (_v_shift0 + 1);

    float4* src = (float4*)(&_G_A_TCFValue[(_I_TCStart_offset + _I_i) * 16 * 16 + _H_lane_id * 8]);
    float4* dst = (float4*)(&_S_tileA_block[smem_st_ptr * _C_tileA_size + _H_lane_id * 8]);
    __pipeline_memcpy_async(dst, src, sizeof(float4));
    if (_H_lane_id < 4) {
        int4* src_idx = (int4*)(&_G_A_TCFColIdx[(_I_TCStart_offset + _I_i) * 16 + _H_lane_id * 4]);
        int4* dst_idx = (int4*)(&_S_sparse_index[smem_st_ptr * 16 + _H_lane_id * 4]);
        __pipeline_memcpy_async(dst_idx, src_idx, sizeof(int4));
    }
    __pipeline_commit();
}
template <int _v_shift0, int _v_shift1, int _v_num_res_groups, int _v_r0, int _v_r1, int _v_r2, int _v_wait_num>
__device__ void r(
  int32_t _H_block_idx_x,
  int32_t _H_lane_id,
  int32_t _I_i,
  int32_t _I_TCStart_offset,
  int32_t _C_tileA_size,
  half _G_A_TCFValue,
  int32_t _G_A_TCOffset,
  int32_t _G_A_TCFColIdx,
  half _G_X_Val,
  half _G_Y_Val,
  half _S_tileA_block,
  int32_t _S_sparse_index,
  uint32_t _R_A,
  half _R_B,
  float _R_C
) {
    const int smem_ld_ptr = _I_i % (_v_shift0 + 1);

    __pipeline_wait_prior(_v_wait_num);
    __syncthreads();

    if (_H_lane_id < 4) {
        _R_B[_v_r2 * 4 + 0] = __ldg(&_G_X_Val[_S_sparse_index[smem_ld_ptr * 16 + _H_lane_id * 2 + 0]]);
        _R_B[_v_r2 * 4 + 1] = __ldg(&_G_X_Val[_S_sparse_index[smem_ld_ptr * 16 + _H_lane_id * 2 + 1]]);
        _R_B[_v_r2 * 4 + 2] = __ldg(&_G_X_Val[_S_sparse_index[smem_ld_ptr * 16 + _H_lane_id * 2 + 8 + 0]]);
        _R_B[_v_r2 * 4 + 3] = __ldg(&_G_X_Val[_S_sparse_index[smem_ld_ptr * 16 + _H_lane_id * 2 + 8 + 1]]);
    }
    ldmatrix_m8n8k8_x4(&_R_A[_v_r2 * 4], &_S_tileA_block[smem_ld_ptr * _C_tileA_size + _H_lane_id * 8]);
}
template <int _v_shift0, int _v_shift1, int _v_num_res_groups, int _v_r0, int _v_r1, int _v_r2, int _v_wait_num>
__device__ void mma(
  int32_t _H_block_idx_x,
  int32_t _H_lane_id,
  int32_t _I_i,
  int32_t _I_TCStart_offset,
  int32_t _C_tileA_size,
  half _G_A_TCFValue,
  int32_t _G_A_TCOffset,
  int32_t _G_A_TCFColIdx,
  half _G_X_Val,
  half _G_Y_Val,
  half _S_tileA_block,
  int32_t _S_sparse_index,
  uint32_t _R_A,
  half _R_B,
  float _R_C
) {
    mma_m16n8k16_fp32(_R_C, &_R_A[_v_r2 * 4], &_R_B[_v_r2 * 4]);
}
template <int _v_shift0, int _v_shift1, int _v_num_res_groups, int _v_r0, int _v_r1, int _v_r2, int _v_wait_num>
__device__ void wb(
  int32_t _H_block_idx_x,
  int32_t _H_lane_id,
  int32_t _I_i,
  int32_t _I_TCStart_offset,
  int32_t _C_tileA_size,
  half _G_A_TCFValue,
  int32_t _G_A_TCOffset,
  int32_t _G_A_TCFColIdx,
  half _G_X_Val,
  half _G_Y_Val,
  half _S_tileA_block,
  int32_t _S_sparse_index,
  uint32_t _R_A,
  half _R_B,
  float _R_C
) {
    if (_H_lane_id % 4 == 0) {
        _G_Y_Val[_H_block_idx_x * 16 + _H_lane_id / 4 + 0] = (half)_R_C[0];
        _G_Y_Val[_H_block_idx_x * 16 + _H_lane_id / 4 + 8] = (half)_R_C[2];
    }
}
s<2, 1, 3, 0, 1, 2, 0>(
    1, 0, 2, TCStart_offset, tileA_size, A_TCFValue, A_TCOffset, A_TCFColIdx, X_Val, Y_Val, tileA_block, sparse_index, A, B, nullptr
);
==================================================
template <int _v_shift0, int _v_shift1, int _v_shift2, int _v_num_res_groups, int _v_r0, int _v_r1, int _v_r2, int _v_r3, int _v_wait_num>
__device__ void seq(
  int32_t _H_block_idx_x,
  int32_t _H_lane_id,
  int32_t _I_i,
  int32_t _I_TCStart_offset,
  int32_t _C_tileA_size,
  half _G_A_TCFValue,
  int32_t _G_A_TCOffset,
  int32_t _G_A_TCFColIdx,
  half _G_X_Val,
  half _G_Y_Val,
  half _S_tileA_block,
  int32_t _S_sparse_index,
  uint32_t _R_A,
  half _R_B,
  float _R_C
) {
    s<_v_shift0, _v_shift1, _v_shift2, _v_num_res_groups, _v_r0, _v_r1, _v_r2, _v_r3, _v_wait_num>(
        _H_block_idx_x, _H_lane_id, _I_i, _I_TCStart_offset, _C_tileA_size, _G_A_TCFValue, _G_A_TCOffset, _G_A_TCFColIdx, _G_X_Val, _G_Y_Val, _S_tileA_block, _S_sparse_index, _R_A, _R_B, _R_C
    );
    r<_v_shift0, _v_shift1, _v_shift2, _v_num_res_groups, _v_r0, _v_r1, _v_r2, _v_r3, _v_wait_num>(
        _H_block_idx_x, _H_lane_id, _I_i, _I_TCStart_offset, _C_tileA_size, _G_A_TCFValue, _G_A_TCOffset, _G_A_TCFColIdx, _G_X_Val, _G_Y_Val, _S_tileA_block, _S_sparse_index, _R_A, _R_B, _R_C
    );
    mma<_v_shift0, _v_shift1, _v_shift2, _v_num_res_groups, _v_r0, _v_r1, _v_r2, _v_r3, _v_wait_num>(
        _H_block_idx_x, _H_lane_id, _I_i, _I_TCStart_offset, _C_tileA_size, _G_A_TCFValue, _G_A_TCOffset, _G_A_TCFColIdx, _G_X_Val, _G_Y_Val, _S_tileA_block, _S_sparse_index, _R_A, _R_B, _R_C
    );
}==================================================
Format(
    axes=(
        Axis(name=X_o, direction=ROW, length=M, is_sparse=False, is_varlen=False, reorder=None)
        Axis(name=Y_s, direction=COL, length=<function N at 0x155157329b20>, is_sparse=True, is_varlen=True, reorder=None)
        Axis(name=X_i, direction=ROW, length=16, is_sparse=False, is_varlen=False, reorder=None)
    )
    child=Format(
        axes=(
            Axis(name=Y_so, direction=COL, length=N/8, is_sparse=False, is_varlen=False, reorder=None)
            Axis(name=Y_si, direction=COL, length=8, is_sparse=False, is_varlen=False, reorder=None)
            Axis(name=X_i, direction=ROW, length=16, is_sparse=False, is_varlen=False, reorder=None)
        )
        child=CooAtomicFormat(
            axes=(
                Axis(name=X_i, direction=ROW, length=_, is_sparse=False, is_varlen=False, reorder=None)
                Axis(name=Y_si, direction=COL, length=_, is_sparse=False, is_varlen=False, reorder=None)
            )
            atomic_type=coo
        )
    )
)