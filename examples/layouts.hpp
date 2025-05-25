#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/pointer.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

using namespace cute;

template <bool is_row> struct Contiguous {
    constexpr static auto row_major = is_row;
    constexpr static auto col_major = !is_row;
};

using RowMajor = Contiguous<true>;
using ColMajor = Contiguous<false>;
using Prow = RowMajor;
using Pcol = ColMajor;
using Trow = RowMajor;
using Tcol = ColMajor;
using Vrow = RowMajor;
using Vcol = ColMajor;

template <int Pr, int Pc, int Tr, int Tc, int Vr, int Vc, class Pmajor, class Tmajor, class Vmajor>
class UniversalTVLayout {
  public:
    // (Pr, Pc) -> Pid
    using PLayout =
      std::conditional_t<Pmajor::row_major,
                         decltype(make_layout(make_shape(Int<Pr>{}, Int<Pc>{}), LayoutRight{})),
                         decltype(make_layout(make_shape(Int<Pr>{}, Int<Pc>{})))>;
    // (Pr, Pc) -> Pr
    using PrPc2PrLayout = Layout<Shape<Int<Pr>, Int<Pc>>, Stride<_1, _0>>;
    // (Pr, Pc) -> Pc
    using PrPc2PcLayout = Layout<Shape<Int<Pr>, Int<Pc>>, Stride<_0, _1>>;
    // Pid -> (Pr, Pc)
    using Pid2PrPcLayout =
      decltype(right_inverse(PLayout{}).with_shape(make_shape(size(PLayout{}))));
    // Pid -> Pr
    using Pid2PrLayout = decltype(composition(PrPc2PrLayout{}, Pid2PrPcLayout{}));
    // Pid -> Pc
    using Pid2PcLayout = decltype(composition(PrPc2PcLayout{}, Pid2PrPcLayout{}));

    // (Tr, Tc) -> Tid
    using TLayout =
      std::conditional_t<Tmajor::row_major,
                         decltype(make_layout(make_shape(Int<Tr>{}, Int<Tc>{}), LayoutRight{})),
                         decltype(make_layout(make_shape(Int<Tr>{}, Int<Tc>{})))>;
    // (Tr, Tc) -> Tr
    using TrTc2TrLayout = Layout<Shape<Int<Tr>, Int<Tc>>, Stride<_1, _0>>;
    // (Tr, Tc) -> Tc
    using TrTc2TcLayout = Layout<Shape<Int<Tr>, Int<Tc>>, Stride<_0, _1>>;
    // Tid -> (Tr, Tc)
    using Tid2TrTcLayout =
      decltype(right_inverse(TLayout{}).with_shape(make_shape(size(TLayout{}))));
    // Tid -> Tr
    using Tid2TrLayout = decltype(composition(TrTc2TrLayout{}, Tid2TrTcLayout{}));
    // Tid -> Tc
    using Tid2TcLayout = decltype(composition(TrTc2TcLayout{}, Tid2TrTcLayout{}));

    // (Vr, Vc) -> Vid
    using VLayout =
      std::conditional_t<Vmajor::row_major,
                         decltype(make_layout(make_shape(Int<Vr>{}, Int<Vc>{}), LayoutRight{})),
                         decltype(make_layout(make_shape(Int<Vr>{}, Int<Vc>{})))>;
    // (Vr, Vc) -> Vr
    using VrVc2VrLayout = Layout<Shape<Int<Vr>, Int<Vc>>, Stride<_1, _0>>;
    // (Vr, Vc) -> Vc
    using VrVc2VcLayout = Layout<Shape<Int<Vr>, Int<Vc>>, Stride<_0, _1>>;
    // Vid -> (Vr, Vc)
    using Vid2VrVcLayout =
      decltype(right_inverse(VLayout{}).with_shape(make_shape(size(VLayout{}))));
    // Vid -> Vr
    using Vid2VrLayout = decltype(composition(VrVc2VrLayout{}, Vid2VrVcLayout{}));
    // Vid -> Vc
    using Vid2VcLayout = decltype(composition(VrVc2VcLayout{}, Vid2VrVcLayout{}));

    // (packed) (m, n) -> (tid, packed_vid)
    using PackedMN2TVLayout = decltype(raked_product(VLayout{}, TLayout{}));
    // (m, n) -> (p, (tid, vid))
    using MN2PTVLayout = decltype(raked_product(PackedMN2TVLayout{}, PLayout{}));
    // (tid, packed_vid) -> (packed) (m, n)
    // This is not commonly used, but for illustration only
    using PackedTVLayout = decltype(select<1, 0>(
      right_inverse(PackedMN2TVLayout{}).with_shape(make_shape(size(VLayout{}), size(TLayout{})))));
    // (p, tid, packed_vid) -> (m, n)
    using PTVLayout = decltype(select<2, 1, 0>(
      right_inverse(MN2PTVLayout{})
        .with_shape(make_shape(size(VLayout{}), size(TLayout{}), size(PLayout{})))));
    // (tid, vid) -> (m, n)
    using TVLayout = decltype(make_layout(layout<1>(PTVLayout{}), select<0, 2>(PTVLayout{})));

    // (Pr, Tr, Vr) -> m
    using PrTrVrLayout = decltype(make_layout(make_shape(Int<Pr>{}, Int<Tr>{}, Int<Vr>{})));
    // (Tr, (Pr, Vr)) -> m
    using TrVrRowLayout =
      decltype(make_layout(layout<1>(PrTrVrLayout{}), select<0, 2>(PrTrVrLayout{})));
    // (Tid, (Pr, Vr)) -> m
    using TVRowLayout = decltype(make_layout(
      composition(layout<0>(TrVrRowLayout{}), Tid2TrLayout{}), layout<1>(TrVrRowLayout{})));

    // (Pc, Tc, Vc) -> n
    using PcTcVcLayout = decltype(make_layout(make_shape(Int<Pc>{}, Int<Tc>{}, Int<Vc>{})));
    // (Tc, (Pc, Vc)) -> n
    using TcVcColLayout =
      decltype(make_layout(layout<1>(PcTcVcLayout{}), select<0, 2>(PcTcVcLayout{})));
    // (Tid, (Pc, Vc)) -> n
    using TVColLayout = decltype(make_layout(
      composition(layout<0>(TcVcColLayout{}), Tid2TcLayout{}), layout<1>(TcVcColLayout{})));
};
