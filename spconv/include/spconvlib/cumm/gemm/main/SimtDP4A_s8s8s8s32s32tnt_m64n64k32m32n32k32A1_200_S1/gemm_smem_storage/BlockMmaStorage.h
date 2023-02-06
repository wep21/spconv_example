#pragma once
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
namespace SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1 {
namespace gemm_smem_storage {
struct BlockMmaStorage {
  tv::alignedarray<int8_t, 4352, 16> smem_A;
  tv::alignedarray<int8_t, 4352, 16> smem_B;
};
} // namespace gemm_smem_storage
} // namespace SimtDP4A_s8s8s8s32s32tnt_m64n64k32m32n32k32A1_200_S1
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib