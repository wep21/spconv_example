#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32tnt_m64n128k32m32n64k32A1_200_S1 {
namespace mma {
namespace mma_ns_wmma {
namespace instmma {
struct InstMma {
  __forceinline__ __device__ void operator()(tv::array<int32_t, 1, 0>& d, tv::array<int8_t, 4, 0> const & a, tv::array<int8_t, 4, 0> const & b, tv::array<int32_t, 1, 0> const & c)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610))
    unsigned const &A = reinterpret_cast<unsigned const &>(a);
    unsigned const &B = reinterpret_cast<unsigned const &>(b);
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
                : "=r"(d[0])
                : "r"(A), "r"(B), "r"(c[0]));
    #else
        d[0] = c[0];
        TV_PRAGMA_UNROLL
        for (int k = 0; k < 4; ++k) {
            d[0] += a[k] * b[k];
        }
    #endif
  }
};
} // namespace instmma
} // namespace mma_ns_wmma
} // namespace mma
} // namespace SimtDP4A_s8s8s8s32s32tnt_m64n128k32m32n64k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib