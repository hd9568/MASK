#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>
#include <cute/pointer.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

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

class E{
	using Shape = cute::Shape<cute::Int<16>>;
	using Tensor = decltype(cute::make_tensor<float>(Shape{}));
	Tensor tensor[3];
};

class F{
	using Shape = cute::Shape<cute::Int<3>>;
	using Tensor = decltype(cute::make_tensor<float>(Shape{}));
	Tensor tensor[16];
};

class G{
	public:
	using Shape = cute::Shape<cute::Int<16>>;
	using Tensor = decltype(cute::make_tensor(cute::make_gmem_ptr((int*)nullptr), Shape{}));
	Tensor tensor;

	G(int* dval_sidx) : tensor(cute::make_tensor(cute::make_gmem_ptr(dval_sidx), Shape{})) {}
};

using namespace cute;

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
	// auto a = cute::Int<16>{};
	// // std::cout << a * a << std::endl;
	// // using shape = cute::Shape<cute::Int<16>, int>;
	// // std::cout << sizeof(shape) << std::endl;
	// std::cout << sizeof(A) << std::endl;
	// std::cout << sizeof(B) << std::endl;
	// std::cout << sizeof(C) << std::endl;
	// std::cout << sizeof(D) << std::endl;
	// std::cout << sizeof(E) << std::endl;
	// std::cout << sizeof(F) << std::endl;
	// G g = G((int*)nullptr);
	// std::cout << sizeof(g) << std::endl;

	// auto scalar_tensor = cute::make_tensor<float>(cute::make_shape(cute::Int<3>{}, cute::Int<4>{}), cute::make_stride(cute::Int<10>{}, cute::Int<1>{}));
	// for (int i = 0; i < 6; i++) {
	// 	scalar_tensor[i] = i;
	// }
	// std::cout << sizeof(scalar_tensor) << std::endl;
	// std::cout << size(scalar_tensor) << std::endl;
	// std::cout << cosize(layout(scalar_tensor)) << std::endl;
	// cute::print(scalar_tensor);
	// std::cout << scalar_tensor(0, 0) << std::endl;
	// std::cout << scalar_tensor(1, 0) << std::endl;
	// std::cout << scalar_tensor(2, 0) << std::endl;
	// std::cout << scalar_tensor(0, 1) << std::endl;
	// std::cout << scalar_tensor(1, 1) << std::endl;
	// std::cout << scalar_tensor(2, 1) << std::endl;
	// cute::print(rank(make_tuple(_1{}, make_tuple(_2{}, _3{}), _4{})));
	// cute::print(get<1>(make_tuple(_1{}, make_tuple(_2{}, _3{}), _4{})));

	// auto tensor0 = make_tensor<float>(make_layout(
	// 	make_shape(_2{}, make_shape(_3{}, _4{}), _2{}),
	// 	make_stride(_48{}, make_stride(_8{}, _2{}), _1{})
	// ));
	// print(tensor0);
	// putchar('\n');
	// for (int i = 0; i < 3*4*2; i++) {
	// 	tensor0(i) = ((float)i) * 0.1 + 0.1;
	// }
	// tensor0(0, 1, 2) = 100;
	// print(tensor0);
	// putchar('\n');
	// float *ptr = tensor0.data();
	// for (int i = 0; i < cosize(layout(tensor0)); i++) {
	// 	printf("%f ", ptr[i]);
	// }
	// auto tA = make_tensor<float>(make_shape(make_shape(_2{}, _4{}), _3{}));
	// auto tB = make_tensor<float>(make_shape(_2{}, _4{}));
	// print(tA); putchar('\n');
	// print(tA(_, 0)); putchar('\n');
	// print(tA((_, _), 0)); putchar('\n');
	// print(tB); putchar('\n');
	// // printf("%d\n", a());
	// auto tensor = make_tensor<float>(make_shape(_16{}));
    // decltype(make_shape(Int<4>{})) thr_tiler;
	// for (int i = 0; i < 5; i++) {
	// 	auto thr_coord = make_coord(i);
	// 	print(local_tile(tensor, thr_tiler, thr_coord)); putchar('\n');
	// }
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
		auto b = composition(Swizzle<2, 0, 2>{}, Layout<Shape<_16, _2>, Stride<_2, _1>>{});
		printf("Layout a: "); print(a); printf("\n");
		printf("Layout b: "); print(b); printf("\n");
		
		print_layout_local(b);
		printf("%14s: %d\n", "cosize(layout)", (int)cosize(b));
		// printf("size = %d\n", cosize(b));
	}

    return 0;
}