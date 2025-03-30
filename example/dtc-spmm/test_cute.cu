#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/pointer.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

class A {
    using Shape = cute::Shape<cute::Int<16>, cute::Int<16>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor = cute::make_tensor<float>(Shape{});
};

class B {
    using Shape = cute::Shape<cute::Int<16>, cute::Int<1>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor = cute::make_tensor<float>(Shape{});
};

class C {
    using Shape = cute::Shape<cute::Int<1>, cute::Int<16>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor = cute::make_tensor<float>(Shape{});
};

class D {
    using Shape = cute::Shape<cute::Int<1>, cute::Int<1>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor = cute::make_tensor<float>(Shape{});
};

class E {
    using Shape = cute::Shape<cute::Int<16>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor[3];
};

class F {
    using Shape = cute::Shape<cute::Int<3>>;
    using Tensor = decltype(cute::make_tensor<float>(Shape{}));
    Tensor tensor[16];
};

class G {
  public:
    using Shape = cute::Shape<cute::Int<16>>;
    using Tensor = decltype(cute::make_tensor(cute::make_gmem_ptr((int*)nullptr), Shape{}));
    Tensor tensor;

    G(int* dval_sidx) : tensor(cute::make_tensor(cute::make_gmem_ptr(dval_sidx), Shape{})) {}
};

using namespace cute;

template <class Tc, class Vc, class Tr, class Vr, class P, class TensorLayout>
auto make_pt2v2_prow_trow_vcol_layout(Tc tc, Vc vc, Tr tr, Vr vr, P p, TensorLayout tensor_layout) {
    auto ptcvc_layout = make_layout(make_shape(p, tc, vc));
    auto trvr_layout = make_layout(make_shape(tr, vr));
    auto ptcvctrvr_tiler = make_tile(trvr_layout, ptcvc_layout);

    // ((tr, vr), (p, tc, vc)) -> elem_idx
    auto trvrptcvc_layout = composition(tensor_layout, ptcvctrvr_tiler);

    // (p, (tc, tr), (vc, vr)) -> elem_idx
    auto ptctrvcvr_layout = make_layout(
      layout<1, 0>(trvrptcvc_layout),
      make_layout(layout<1, 1>(trvrptcvc_layout), layout<0, 0>(trvrptcvc_layout)),
      make_layout(layout<1, 2>(trvrptcvc_layout), layout<0, 1>(trvrptcvc_layout)));
	printf("trvrptcvc_layout: ");
	print(trvrptcvc_layout);
	printf("\n");
	printf("ptctrvcvr_layout: ");
	print(ptctrvcvr_layout);
	printf("\n");

    // (p, (tc, tr), (vc, vr)) -> elem_idx
	return ptctrvcvr_layout;
}

template <class Tc, class Vc, class Tr, class Vr, class P, class TensorLayout>
auto make_pt2v2_pcol_tcol_vcol_layout(Tc tc, Vc vc, Tr tr, Vr vr, P p, TensorLayout tensor_layout) {
    auto tcvc_layout = make_layout(make_shape(tc, vc));
    auto ptrvr_layout = make_layout(make_shape(p, tr, vr));
    auto tcvcptrvr_tiler = make_tile(ptrvr_layout, tcvc_layout);

    // ((p, tr, vr), (tc, vc)) -> elem_idx
    auto ptrvrtcvc_layout = composition(tensor_layout, tcvcptrvr_tiler);

    // (p, (tc, tr), (vc, vr)) -> elem_idx
    auto ptctrvcvr_layout = make_layout(
      layout<0, 0>(ptrvrtcvc_layout),
      make_layout(layout<1, 0>(ptrvrtcvc_layout), layout<0, 1>(ptrvrtcvc_layout)),
      make_layout(layout<1, 1>(ptrvrtcvc_layout), layout<0, 2>(ptrvrtcvc_layout)));
	printf("ptrvrtcvc_layout: ");
	print(ptrvrtcvc_layout);
	printf("\n");
	printf("ptctrvcvr_layout: ");
	print(ptctrvcvr_layout);
	printf("\n");

    // (p, (tc, tr), (vc, vr)) -> elem_idx
	return ptctrvcvr_layout;
}

template <class Tc, class Vc, class Tr, class Vr, class P, class PTcTrVcVrLayout>
CUTE_HOST_DEVICE auto make_ptv_prow_trow_vcol_layout(Tc tc, Vc vc, Tr tr, Vr vr, P p, PTcTrVcVrLayout ptctrvcvr_layout) {
    // (tc, tr) -> tid
    // (vc, vr) -> vid
    auto tctr_to_tid_layout = make_layout(make_shape(tc, tr));
    auto vcvr_to_vid_layout = make_layout(make_shape(vc, vr), LayoutRight{});

    // tid -> (tc, tr)
    // vid -> (vc, vr)
    auto tid_to_tctr_layout = right_inverse(tctr_to_tid_layout);
    auto vid_to_vcvr_layout = right_inverse(vcvr_to_vid_layout);

    // p -> p (identical)
    auto p_layout = make_layout(make_shape(p));

    // (p, (tid, vid)) -> (p, (tc, tr), (vc, vr))
    auto tv_to_tctrvcvr_layout = logical_product(tid_to_tctr_layout, vid_to_vcvr_layout);
    printf("tv_to_tctrvcvr_layout: ");
    print(tv_to_tctrvcvr_layout);
    printf("\n");
    auto _ptv_to_ptctrvcvr_layout = logical_product(p_layout, tv_to_tctrvcvr_layout);
    auto ptv_to_ptctrvcvr_layout =
      make_layout(layout<0, 0>(_ptv_to_ptctrvcvr_layout), layout<1, 0>(_ptv_to_ptctrvcvr_layout),
                  layout<1, 1>(_ptv_to_ptctrvcvr_layout));
    printf("ptv_to_ptctrvcvr_layout: ");
    print(ptv_to_ptctrvcvr_layout);
    printf("\n");

    auto ptv_layout = composition(ptctrvcvr_layout, ptv_to_ptctrvcvr_layout);
	// (p, tid, vid) -> elem_idx
    return ptv_layout;
}

template <class Tc, class Vc, class Tr, class Vr, class P, class PTcTrVcVrLayout>
CUTE_HOST_DEVICE auto make_ptv_pcol_tcol_vcol_layout(Tc tc, Vc vc, Tr tr, Vr vr, P p, PTcTrVcVrLayout ptctrvcvr_layout) {
    // (tc, tr) -> tid
    // (vc, vr) -> vid
    auto tctr_to_tid_layout = make_layout(make_shape(tc, tr), LayoutRight{});
    auto vcvr_to_vid_layout = make_layout(make_shape(vc, vr));

    // tid -> (tc, tr)
    // vid -> (vc, vr)
    auto tid_to_tctr_layout = right_inverse(tctr_to_tid_layout);
    auto vid_to_vcvr_layout = right_inverse(vcvr_to_vid_layout);

    // p -> p (identical)
    auto p_layout = make_layout(make_shape(p));

    // (p, (tid, vid)) -> (p, (tc, tr), (vc, vr))
    auto tv_to_tctrvcvr_layout = logical_product(tid_to_tctr_layout, vid_to_vcvr_layout);
    printf("tv_to_tctrvcvr_layout: ");
    print(tv_to_tctrvcvr_layout);
    printf("\n");
    auto _ptv_to_ptctrvcvr_layout = logical_product(p_layout, tv_to_tctrvcvr_layout);
    auto ptv_to_ptctrvcvr_layout =
      make_layout(layout<0, 0>(_ptv_to_ptctrvcvr_layout), layout<1, 0>(_ptv_to_ptctrvcvr_layout),
                  layout<1, 1>(_ptv_to_ptctrvcvr_layout));
    printf("ptv_to_ptctrvcvr_layout: ");
    print(ptv_to_ptctrvcvr_layout);
    printf("\n");

    auto ptv_layout = composition(ptctrvcvr_layout, ptv_to_ptctrvcvr_layout);
	// (p, tid, vid) -> elem_idx
    return ptv_layout;
}

int main() {

    Layout C_layout = make_layout(make_shape(_16{}, _8{}), LayoutRight{});
    auto C_tr = _8{};
    auto C_vr = _2{};
    auto C_tc = _4{};
    auto C_vc = _1{};
    auto C_p = _2{}; // packed 2

	auto C_pt2v2_layout = make_pt2v2_prow_trow_vcol_layout(C_tc, C_vc, C_tr, C_vr, C_p, C_layout);
    auto C_ptv_layout = make_ptv_prow_trow_vcol_layout(C_tc, C_vc, C_tr, C_vr, C_p, C_pt2v2_layout);

	auto C0 = new float[C_tr * C_tc * C_vc * C_vr * C_p];
	auto C1 = new float[C_tr * C_tc * C_vc * C_vr * C_p];
	Tensor C_tensor_pt2v2_0 = make_tensor(C0, C_pt2v2_layout);
	Tensor C_tensor_pt2v2_1 = make_tensor(C1, C_pt2v2_layout);
	Tensor C_tensor_ptv_0 = make_tensor(C0, C_ptv_layout);
	Tensor C_tensor_ptv_1 = make_tensor(C1, C_ptv_layout);

	for (int tx = 0; tx < C_tr; tx++) {
		for (int ty = 0; ty < C_tc; ty++) {
			for (int vx = 0; vx < C_vr; vx++) {
				for (int vy = 0; vy < C_vc; vy++) {
					C_tensor_pt2v2_0(0, make_coord(ty, tx), make_coord(vy, vx)) = 10.0f;
					C_tensor_pt2v2_0(1, make_coord(ty, tx), make_coord(vy, vx)) = 11.0f;
				}
			}
		}
	}

	for (int t = 0; t < C_tr * C_tc; t++) {
		if (t % 3 == 0) {
			copy(C_tensor_ptv_0(_, t, _), C_tensor_ptv_1(_, t, _));
			printf("copy %d size %d vs. %d\n", t, 
				int(size(C_tensor_ptv_0(_, t, _))),
				int(size(C_tensor_ptv_1(_, t, _))));
			print(C_tensor_ptv_0(_, t, _));
			printf("||");
			print(C_tensor_ptv_1(_, t, _));
			printf("\n");
		}
	}
	
	print(C_tensor_ptv_1);
	for (int i = 0; i < C_tr * C_tc * C_vc * C_vr * C_p; i++) {
		if (i % (C_tc * C_vc * C_p) == 0) {
			printf("\n");
		}
		printf("%6.2f ", C0[i]);
	}
	printf("\n\n");
	for (int i = 0; i < C_tr * C_tc * C_vc * C_vr * C_p; i++) {
		if (i % (C_tc * C_vc * C_p) == 0) {
			printf("\n");
		}
		printf("%6.2f ", C1[i]);
	}
	printf("\n");

	printf("C_pt2v2_layout: ");
	print(C_pt2v2_layout);
	printf("\n");
    printf("C_ptv_layout: ");
    print(C_ptv_layout);
    printf("\n");

	Layout B_layout = make_layout(make_shape(_16{}, _8{}));
	auto B_tr = _4{};
	auto B_vr = _2{};
	auto B_tc = _8{};
	auto B_vc = _1{};
	auto B_p = _2{};

	auto B_pt2v2_layout = make_pt2v2_pcol_tcol_vcol_layout(B_tc, B_vc, B_tr, B_vr, B_p, B_layout);
	auto B_ptv_layout = make_ptv_pcol_tcol_vcol_layout(B_tc, B_vc, B_tr, B_vr, B_p, B_pt2v2_layout);

	auto B0 = new float[B_tr * B_tc * B_vc * B_vr * B_p];
	auto B1 = new float[B_tr * B_tc * B_vc * B_vr * B_p];
	Tensor B_tensor_pt2v2_0 = make_tensor(B0, B_pt2v2_layout);
	Tensor B_tensor_pt2v2_1 = make_tensor(B1, B_pt2v2_layout);
	Tensor B_tensor_ptv_0 = make_tensor(B0, B_ptv_layout);
	Tensor B_tensor_ptv_1 = make_tensor(B1, B_ptv_layout);

	for (int tx = 0; tx < B_tr; tx++) {
		for (int ty = 0; ty < B_tc; ty++) {
			for (int vx = 0; vx < B_vr; vx++) {
				for (int vy = 0; vy < B_vc; vy++) {
					B_tensor_pt2v2_0(0, make_coord(ty, tx), make_coord(vy, vx)) = float(tx + ty * B_tr) + 0.0f;
					B_tensor_pt2v2_0(1, make_coord(ty, tx), make_coord(vy, vx)) = float(tx + ty * B_tr) + 0.5f;
				}
			}
		}
	}

	for (int t = 0; t < B_tr * B_tc; t++) {
		if (t % 3 == 0) {
			auto tmp0 = B_tensor_ptv_0(_, t, _);
			auto tmp1 = B_tensor_ptv_1(_, t, _);
			printf("copy %d size %d vs. %d\n", t, int(size(tmp0)), int(size(tmp1)));
			copy(tmp0, tmp1);
			printf("||");
			print(tmp0);
			printf("||");
			print(tmp1);
			printf("\n");
		}
	}

	for (int i = 0; i < B_tr * B_tc * B_vc * B_vr * B_p; i++) {
		if (i % (B_tr * B_vr * B_p) == 0) {
			printf("\n");
		}
		printf("%6.2f ", B0[i]);
	}
	printf("\n\n");
	for (int i = 0; i < B_tr * B_tc * B_vc * B_vr * B_p; i++) {
		if (i % (B_tr * B_vr * B_p) == 0) {
			printf("\n");
		}
		printf("%6.2f ", B1[i]);
	}
	printf("\n");

	printf("B_pt2v2_layout: ");
	print(B_pt2v2_layout);
	printf("\n");
	printf("B_ptv_layout: ");
	print(B_ptv_layout);
	printf("\n");

    return 0;
}