#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace Turing_s8s8s8s32s32tnt_m256n128k32m64n64k32A1T8816_200_S1 {
namespace inpiterb {
namespace maskiter {
struct GlobalLoad {
  template <typename Frag>
  __forceinline__ __device__ static void run(Frag & frag, void const* ptr, bool pred)   {
    
    uint32_t* frag_ptr = reinterpret_cast<uint32_t*>(&frag);
    #if (CUDA_VERSION >= 11040)
      asm volatile (
          "{\n"
          "  mov.b32 %0,%0;\n"
          "  mov.b32 %1,%1;\n"
          "  mov.b32 %2,%2;\n"
          "  mov.b32 %3,%3;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #else
      asm volatile (
          "{\n"
          "  mov.b32 %0,%0;\n"
          "  mov.b32 %1,%1;\n"
          "  mov.b32 %2,%2;\n"
          "  mov.b32 %3,%3;\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p,%4,0;\n"
          "  @p ld.global.v4.b32 {%0, %1, %2, %3},[%5];\n"
          "}\n"
          : "+r"(frag_ptr[0]), "+r"(frag_ptr[1]), "+r"(frag_ptr[2]), "+r"(frag_ptr[3])
          : "r"((int)pred), "l"(ptr)
      );
    #endif
  }
};
} // namespace maskiter
} // namespace inpiterb
} // namespace Turing_s8s8s8s32s32tnt_m256n128k32m64n64k32A1T8816_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib