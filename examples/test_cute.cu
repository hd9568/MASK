#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/pointer.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "layouts.hpp"

// Generic 2D Layout to console table
template <class Layout>
CUTE_HOST_DEVICE
void
print_layout_local(Layout const& layout)  // (m,n) -> idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;
  const char* delim = "+-----------------------";

  print(layout); print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("  %*d ", idx_width-2, n); }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) { printf("| %*d ", idx_width-2, int(layout(m,n)) % 8); }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) { printf("%.*s", idx_width+1, delim); }
  printf("+\n");
}

int main() {

    constexpr Layout A_layout = make_layout(make_shape(_16{}, _16{}), LayoutRight{});
    constexpr Layout B_layout = make_layout(make_shape(_16{}, _8{}));
    constexpr Layout C_layout = make_layout(make_shape(_16{}, _8{}), LayoutRight{});
    constexpr int A_tr = 8, A_tc = 4, A_vr = 2, A_vc = 2, A_pr = 1,
                  A_pc = 2; // packed 2 with different col (in a row)
    constexpr int B_tr = 4, B_tc = 8, B_vr = 2, B_vc = 1, B_pr = 2,
                  B_pc = 1; // packed 2 with different row (in a col)
    constexpr int C_tr = 8, C_tc = 4, C_vr = 2, C_vc = 1, C_pr = 1,
                  C_pc = 2; // packed 2 with different col (in a row)

    using A_universal_layout =
      UniversalTVLayout<A_pr, A_pc, A_tr, A_tc, A_vr, A_vc, Prow, Trow, Vcol>;
    printf("A_universal_layout::TVLayout: ");
    print(A_universal_layout::TVLayout{});
    printf("\n");
    using B_universal_layout =
      UniversalTVLayout<B_pr, B_pc, B_tr, B_tc, B_vr, B_vc, Pcol, Tcol, Vcol>;
    printf("B_universal_layout::TVLayout: ");
    print(B_universal_layout::TVLayout{});
    printf("\n");
    using C_universal_layout =
      UniversalTVLayout<C_pr, C_pc, C_tr, C_tc, C_vr, C_vc, Prow, Trow, Vcol>;
    printf("C_universal_layout::TVLayout: ");
    print(C_universal_layout::TVLayout{});
    printf("\n");
    printf("A_universal_layout::TrVrRowLayout: ");
    print(A_universal_layout::TrVrRowLayout{});
    printf("\n");
    printf("A_universal_layout::TVRowLayout: ");
    print(A_universal_layout::TVRowLayout{});
    printf("\n");
    printf("B_universal_layout::TrVrRowLayout: ");
    print(B_universal_layout::TrVrRowLayout{});
    printf("\n");
    printf("B_universal_layout::TVRowLayout: ");
    print(B_universal_layout::TVRowLayout{});
    printf("\n");
    printf("C_universal_layout::TrVrRowLayout: ");
    print(C_universal_layout::TrVrRowLayout{});
    printf("\n");
    printf("C_universal_layout::TVRowLayout: ");
    print(C_universal_layout::TVRowLayout{});
    printf("\n");
    
    

    printf("A TVLayout: ");
    print(composition(A_layout, A_universal_layout::TVLayout{}));
    printf("\n");
    printf("B TVLayout: ");
    print(composition(B_layout, B_universal_layout::TVLayout{}));
    printf("\n");
    printf("C TVLayout: ");
    print(composition(C_layout, C_universal_layout::TVLayout{}));
    printf("\n");


    printf("Testing if cute would copy tensors created without ptr\n");
    auto tensor_A = make_tensor<float>(make_shape(_16{}));
    for (int i = 0; i < 16; i++) {
        tensor_A(i) = i;
    }
    auto tensor_B = tensor_A;
    printf("After tensor_B = tensor_A, tensor_B: ");
    for (int i = 0; i < 16; i++) {
        printf("%f ", tensor_B(i));
    }
    printf("\n");
    tensor_A(0) = 100.0f;
    printf("After tensor_A(0) = 100.0f, tensor_B: ");
    for (int i = 0; i < 16; i++) {
        printf("%f ", tensor_B(i));
    }
    printf("\n");

    printf("Testing if cute would copy tensors created with ptr\n");
    auto ptr = new float[16];
    auto tensor_pA = make_tensor(ptr, make_shape(_16{}));
    for (int i = 0; i < 16; i++) {
        tensor_pA(i) = i;
    }
    auto tensor_pB = tensor_pA;
    printf("After tensor_pB = tensor_pA, tensor_pB: ");
    for (int i = 0; i < 16; i++) {
        printf("%f ", tensor_pB(i));
    }
    printf("\n");
    tensor_pA(0) = 100.0f;
    printf("After tensor_pA(0) = 100.0f, tensor_pB: ");
    for (int i = 0; i < 16; i++) {
        printf("%f ", tensor_pB(i));
    }
    printf("\n");

    auto tensor_0d = make_tensor<float>(Shape<_1>{});
    tensor_0d(0) = 101.0f;
    printf("tensor_0d: %f\n", tensor_0d(0));


    // auto C0 = new float[C_tr * C_tc * C_vc * C_vr * C_p];
    // auto C1 = new float[C_tr * C_tc * C_vc * C_vr * C_p];
    // Tensor C_tensor_pt2v2_0 = make_tensor(C0, C_pt2v2_layout);
    // Tensor C_tensor_pt2v2_1 = make_tensor(C1, C_pt2v2_layout);
    // Tensor C_tensor_ptv_0 = make_tensor(C0, C_ptv_layout);
    // Tensor C_tensor_ptv_1 = make_tensor(C1, C_ptv_layout);

    // for (int tx = 0; tx < C_tr; tx++) {
    //     for (int ty = 0; ty < C_tc; ty++) {
    //         for (int vx = 0; vx < C_vr; vx++) {
    //             for (int vy = 0; vy < C_vc; vy++) {
    //                 C_tensor_pt2v2_0(0, make_coord(ty, tx), make_coord(vy, vx)) = 10.0f;
    //                 C_tensor_pt2v2_0(1, make_coord(ty, tx), make_coord(vy, vx)) = 11.0f;
    //             }
    //         }
    //     }
    // }
    // for (int t = 0; t < C_tr * C_tc; t++) {
    //     if (t % 3 == 0) {
    //         copy(C_tensor_ptv_0(_, t, _), C_tensor_ptv_1(_, t, _));
    //         printf("copy %d size %d vs. %d\n", t, int(size(C_tensor_ptv_0(_, t, _))),
    //                int(size(C_tensor_ptv_1(_, t, _))));
    //         print(C_tensor_ptv_0(_, t, _));
    //         printf("||");
    //         print(C_tensor_ptv_1(_, t, _));
    //         printf("\n");
    //     }
    // }

    // print(C_tensor_ptv_1);
    // for (int i = 0; i < C_tr * C_tc * C_vc * C_vr * C_p; i++) {
    //     if (i % (C_tc * C_vc * C_p) == 0) {
    //         printf("\n");
    //     }
    //     printf("%6.2f ", C0[i]);
    // }
    // printf("\n\n");
    // for (int i = 0; i < C_tr * C_tc * C_vc * C_vr * C_p; i++) {
    //     if (i % (C_tc * C_vc * C_p) == 0) {
    //         printf("\n");
    //     }
    //     printf("%6.2f ", C1[i]);
    // }
    // printf("\n");

    // printf("C_pt2v2_layout: ");
    // print(C_pt2v2_layout);
    // printf("\n");
    // printf("C_ptv_layout: ");
    // print(C_ptv_layout);
    // printf("\n");

    // constexpr Layout B_pt2v2_layout =
    //   make_pt2v2_pcol_tcol_vcol_layout<B_tc, B_vc, B_tr, B_vr, B_p>(B_layout);
    // constexpr Layout B_ptv_layout =
    //   make_ptv_pcol_tcol_vcol_layout<B_tc, B_vc, B_tr, B_vr, B_p>(B_layout);

    // auto B0 = new float[B_tr * B_tc * B_vc * B_vr * B_p];
    // auto B1 = new float[B_tr * B_tc * B_vc * B_vr * B_p];
    // Tensor B_tensor_pt2v2_0 = make_tensor(B0, B_pt2v2_layout);
    // Tensor B_tensor_pt2v2_1 = make_tensor(B1, B_pt2v2_layout);
    // Tensor B_tensor_ptv_0 = make_tensor(B0, B_ptv_layout);
    // Tensor B_tensor_ptv_1 = make_tensor(B1, B_ptv_layout);

    // for (int tx = 0; tx < B_tr; tx++) {
    //     for (int ty = 0; ty < B_tc; ty++) {
    //         for (int vx = 0; vx < B_vr; vx++) {
    //             for (int vy = 0; vy < B_vc; vy++) {
    //                 B_tensor_pt2v2_0(0, make_coord(ty, tx), make_coord(vy, vx)) =
    //                   float(tx + ty * B_tr) + 0.0f;
    //                 B_tensor_pt2v2_0(1, make_coord(ty, tx), make_coord(vy, vx)) =
    //                   float(tx + ty * B_tr) + 0.5f;
    //             }
    //         }
    //     }
    // }

    // for (int t = 0; t < B_tr * B_tc; t++) {
    //     if (t % 3 == 0) {
    //         auto tmp0 = B_tensor_ptv_0(_, t, _);
    //         auto tmp1 = B_tensor_ptv_1(_, t, _);
    //         printf("copy %d size %d vs. %d\n", t, int(size(tmp0)), int(size(tmp1)));
    //         copy(tmp0, tmp1);
    //         printf("||");
    //         print(tmp0);
    //         printf("||");
    //         print(tmp1);
    //         printf("\n");
    //     }
    // }

    // for (int i = 0; i < B_tr * B_tc * B_vc * B_vr * B_p; i++) {
    //     if (i % (B_tr * B_vr * B_p) == 0) {
    //         printf("\n");
    //     }
    //     printf("%6.2f ", B0[i]);
    // }
    // printf("\n\n");
    // for (int i = 0; i < B_tr * B_tc * B_vc * B_vr * B_p; i++) {
    //     if (i % (B_tr * B_vr * B_p) == 0) {
    //         printf("\n");
    //     }
    //     printf("%6.2f ", B1[i]);
    // }
    // printf("\n");

    // printf("B_pt2v2_layout: ");
    // print(B_pt2v2_layout);
    // printf("\n");
    // printf("B_ptv_layout: ");
    // print(B_ptv_layout);
    // printf("\n");

    // auto rank_1_layout = make_layout((make_shape(_16{}, _1{})));
    // static_assert(rank_1_layout.rank == 1);

	{
		auto a = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
		auto b = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
		auto b = composition(Swizzle<2, 0, 4>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
		auto b = composition(Swizzle<3, 0, 4>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _16>, Stride<_16, _1>>{};
		auto b = composition(Swizzle<4, 0, 4>{}, Layout<Shape<_32, _16>, Stride<_16, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<2, 0, 3>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<1, 0, 3>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<4, 0, 4>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<3, 0, 4>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{
		auto a = Layout<Shape<_32, _8>, Stride<_8, _1>>{};
		auto b = composition(Swizzle<3, 0, 4>{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{	// suppose tile N = 32 (无冲突)
		auto a = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
		auto b = composition(Swizzle<2, 0, 3>{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{	// suppose tile N = 32 （有冲突）
		auto a = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
		auto b = composition(Swizzle<2, 0, 2>{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{	// suppose tile N = 32 (无冲突)
		auto a = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
		auto b = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{	// suppose tile N = 32 (无冲突)
		auto a = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
		auto b = composition(Swizzle<1, 0, 3>{}, Layout<Shape<_32, _4>, Stride<_4, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
	}
	{	// suppose tile N = 16 (无冲突)
		auto a = Layout<Shape<_16, _2>, Stride<_2, _1>>{};
		auto b = composition(Swizzle<3, 0, 3>{}, Layout<Shape<_16, _2>, Stride<_2, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}
	{	// suppose tile N = 16 (无冲突)
		auto a = Layout<Shape<_16, _2>, Stride<_2, _1>>{};
		auto b = composition(Swizzle<1, 0, 3>{}, Layout<Shape<_16, _2>, Stride<_2, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}
	{	// suppose tile N = 16 (无冲突)
		auto a = Layout<Shape<_16, _2>, Stride<_2, _1>>{};
		auto b = composition(Swizzle<1, 0, 1>{}, Layout<Shape<_16, _2>, Stride<_2, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}
	{	// suppose tile N = 16 (无冲突)
		auto a = Layout<Shape<_16, _2>, Stride<_2, _1>>{};
		auto b = composition(Swizzle<2, 0, 2>{}, Layout<Shape<_16, _2>, Stride<_2, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}
	{	// suppose tile N = 128 (无冲突)
		auto a = Layout<Shape<_16, _16>, Stride<_16, _1>>{};
		auto b = composition(Swizzle<3, 0, 4>{}, Layout<Shape<_16, _16>, Stride<_16, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}

    return 0;
}