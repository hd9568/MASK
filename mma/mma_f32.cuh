struct MmaM16N8K4RowColF32Tf32 {
    __device__ static void mma(uint32_t (&frag_d)[4], uint32_t (&frag_a)[2], uint32_t (&Frag_b)[1], uint32_t (&frag_c)[4]) {
        asm volatile(
            "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
            : "=f"(frag_D[0]), "=f"(frag_D[1]), "=f"(frag_D[2]), "=f"(frag_D[3])
            : "r"(frag_A[2]), "r"(frag_A[3]),
              "r"(frag_B[2]),
              "f"(frag_D[0]), "f"(frag_D[1]), "f"(frag_D[2]), "f"(frag_D[3]));
    }
}
