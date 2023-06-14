#pragma once
#include <tensorview/gemm/arch/memory_sm75.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SKD_S8 {
namespace mma {
namespace mma_ns_wb {
namespace ldsm {
struct LdMatrix {
  __forceinline__ __device__ static void run(tv::array<unsigned, 4, 0> & D, void const* ptr)   {
    
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
      unsigned addr = tv::gemm::get_smem_pointer(ptr);
      int x, y, z, w;
      asm volatile ("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(addr));
      reinterpret_cast<int4 &>(D) = make_int4(x, y, z, w);
    #else
      assert(0);
    #endif
  }
};
} // namespace ldsm
} // namespace mma_ns_wb
} // namespace mma
} // namespace Ampere_s8s8s8s32f16tnt_m128n64k64m64n32k64A1T16832_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib